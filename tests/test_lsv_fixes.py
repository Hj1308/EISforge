"""
Regression tests for the L1/L2 fixes (numerical bugs + scientific validity).

Covers:
  Fix 1 — carbon_standards oxidised_carbon mF→μF unit bug + relaxed noise cut
  Fix 2 — rising-branch interpolation (non-monotonic AOR curves)
  Fix 3 — E@j reported always; η gated behind an equilibrium potential
  Fix 4 — mass/specific activity at AOR peak (not at onset)
  Fix 5 — capacitive-contamination warning for high-C_dl carbons
  Fix 6 — neutral rating language (no publishability claims)
  Fix 7 — apparent-Tafel labelling + reference-frame sanity warning
"""

import numpy as np
import pytest

from eisforge.analysis.lsv_analyzer import LSVAnalyzer
from eisforge.analysis.cv_analyzer import CATALYST_METAL_FREE
from eisforge.standards.carbon_standards import CDL_RANGES, CarbonValidator


# ── Synthetic AOR-like LSV: exponential foot → peak → decline ────────────────

def make_aor_lsv(n=800, e_half_rise=0.70, e_peak=1.05, j_peak=25.0,
                 slope_v_per_dec=0.120, e_max=1.40, noise=0.0, seed=0):
    """Smooth Tafel-like foot (logistic in log-current), soft saturation to a
    peak, then decline (poisoning / second-wave valley) — the canonical
    non-monotonic AOR shape with several decades on the activation branch."""
    rng = np.random.default_rng(seed)
    E = np.linspace(0.0, e_max, n)
    x = 10 ** ((E - e_half_rise) / slope_v_per_dec)
    j = j_peak * x / (1.0 + x)
    decline = np.where(E > e_peak,
                       np.exp(-((E - e_peak) ** 2) / (2 * 0.12 ** 2)), 1.0)
    j = j * decline
    if noise:
        j = j + rng.normal(0.0, noise, size=n)
    return E, j


def analyzer(**kw):
    defaults = dict(scan_rate=5.0, electrode_area=1.0,
                    electrolyte="KOH", electrolyte_concentration=1.0,
                    catalyst_type=CATALYST_METAL_FREE)
    defaults.update(kw)
    return LSVAnalyzer(**defaults)


# ═══ Fix 1 — carbon_standards unit bug ═══════════════════════════════════════

class TestCarbonStandardsUnits:
    def test_oxidised_carbon_range_is_mF_regime(self):
        ref = CDL_RANGES["oxidised_carbon"]
        # literature values were 4.84–11.44 mF/cm² → 4840–11440 μF/cm²
        assert ref.cdl_min_uF == pytest.approx(4840.0)
        assert ref.cdl_max_uF == pytest.approx(11440.0)

    def test_literature_value_now_validates_ok(self):
        # 8 mF/cm² is squarely inside the published range and must pass
        res = CarbonValidator.validate_cdl(cdl_mF_cm2=8.0,
                                           material_key="oxidised_carbon")
        assert res.passed and res.severity == "ok"

    def test_generic_noise_threshold_still_active_for_low_cdl_materials(self):
        # 8 mF/cm² against plain graphene (10–50 μF) must still hard-error
        res = CarbonValidator.validate_cdl(cdl_mF_cm2=8.0,
                                           material_key="graphene")
        assert not res.passed and res.severity == "error"


# ═══ Fix 2 — rising-branch interpolation ═════════════════════════════════════

class TestRisingBranchInterpolation:
    def test_e_at_j_on_rising_branch_only(self):
        E, j = make_aor_lsv(j_peak=25.0)
        e10 = LSVAnalyzer._e_at_j_rising(E, j, 10.0)
        e_peak = float(E[int(np.argmax(j))])
        assert np.isfinite(e10)
        assert e10 < e_peak                      # must be BEFORE the peak
        # j at that potential really is ~10 on the rising branch
        assert np.interp(e10, E, np.maximum.accumulate(
            np.where(E <= e_peak, j, j.max()))) == pytest.approx(10.0, rel=0.05)

    def test_target_above_peak_returns_nan(self):
        E, j = make_aor_lsv(j_peak=25.0)
        assert np.isnan(LSVAnalyzer._e_at_j_rising(E, j, 100.0))

    def test_half_wave_is_finite_and_below_peak_despite_decline(self):
        E, j = make_aor_lsv(j_peak=25.0)
        e_half = LSVAnalyzer._half_wave_potential(E, j, float(np.max(j)))
        e_peak = float(E[int(np.argmax(j))])
        assert np.isfinite(e_half) and e_half < e_peak

    def test_old_full_curve_interp_would_have_been_wrong(self):
        """Documents WHY the fix matters: full-curve np.interp on the
        non-monotonic j gives a different (wrong) potential."""
        E, j = make_aor_lsv(j_peak=25.0)
        wrong = float(np.interp(10.0, j, E))     # ill-defined, silently wrong
        right = LSVAnalyzer._e_at_j_rising(E, j, 10.0)
        assert abs(wrong - right) > 0.01         # they genuinely differ


# ═══ Fix 3 — η gated behind E_eq ═════════════════════════════════════════════

class TestOverpotentialGating:
    def test_without_eeq_eta_invalid_but_e_at_j_reported(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert not r.eta_is_valid
        assert np.isnan(r.overpotential_10)
        assert np.isfinite(r.e_at_j10)

    def test_with_eeq_eta_equals_e_at_j_minus_eeq(self):
        E, j = make_aor_lsv()
        e_eq = 0.10
        r = analyzer(equilibrium_potential=e_eq).analyze(E, j)
        assert r.eta_is_valid
        assert r.overpotential_10 == pytest.approx(r.e_at_j10 - e_eq, abs=1e-9)

    def test_summary_mentions_eta_requirement_when_missing(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert "equilibrium potential" in r.summary()


# ═══ Fix 4 — activities at the AOR peak ══════════════════════════════════════

class TestActivityDefinition:
    def test_mass_activity_uses_peak_not_onset(self):
        E, j = make_aor_lsv(j_peak=25.0)
        loading = 0.5  # mg/cm²
        r = analyzer(catalyst_loading=loading).analyze(E, j)
        assert r.mass_activity == pytest.approx(float(np.max(j)) / loading,
                                                rel=0.05)
        # old (onset-based) definition gives a materially different, smaller value
        old_definition = r.j_at_onset / loading
        assert r.mass_activity > old_definition
        assert "peak" in r.activity_reference

    def test_user_activity_potential_respected(self):
        E, j = make_aor_lsv()
        e_user = 0.80
        r = analyzer(catalyst_loading=1.0,
                     activity_potential=e_user).analyze(E, j)
        assert r.mass_activity == pytest.approx(float(np.interp(e_user, E, j)),
                                                rel=0.02)
        assert f"{e_user:.3f}" in r.activity_reference


# ═══ Fix 5 — capacitive contamination warning ════════════════════════════════

class TestCapacitanceCheck:
    def test_high_cdl_triggers_warning(self):
        E, j = make_aor_lsv()
        # 20 mF/cm² at 5 mV/s → j_cap = 0.1 mA/cm², inside the metal-free
        # Tafel window foot (0.02 mA/cm²) → must warn
        r = analyzer(cdl_uF_per_cm2=20000.0).analyze(E, j)
        assert any("capacitance-inflated" in w or "j_cap" in w
                   for w in r.tafel_warnings)

    def test_blank_subtracted_suppresses_warning(self):
        E, j = make_aor_lsv()
        r = analyzer(cdl_uF_per_cm2=20000.0, blank_subtracted=True).analyze(E, j)
        assert not any("capacitance-inflated" in w for w in r.tafel_warnings)

    def test_metal_free_without_cdl_gets_soft_note(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert any("blank" in w.lower() for w in r.tafel_warnings)

    def test_tiny_cdl_no_warning(self):
        E, j = make_aor_lsv()
        # 20 μF/cm² at 5 mV/s → j_cap = 1e-4 mA/cm² ≪ window foot
        r = analyzer(cdl_uF_per_cm2=20.0).analyze(E, j)
        assert not any("capacitance-inflated" in w for w in r.tafel_warnings)


# ═══ Fix 6 — neutral rating language ═════════════════════════════════════════

class TestNeutralLanguage:
    def test_no_publishability_claims(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        text = (r.performance_rating + r.mechanism_interpretation).lower()
        for banned in ("publishable", "high-if", "journal"):
            assert banned not in text

    def test_interpretation_uses_hypothesis_language(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert ("consistent with" in r.mechanism_interpretation
                or "apparent" in r.mechanism_interpretation.lower())


# ═══ Fix 7 — apparent label + frame sanity ═══════════════════════════════════

class TestFrameAndLabels:
    def test_summary_labels_tafel_as_apparent(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert "Apparent Tafel slope" in r.summary()

    def test_negative_frame_without_offset_warns(self):
        # simulate data recorded vs Ag/AgCl in alkaline (mostly negative E)
        E, j = make_aor_lsv()
        E_agcl = E - 1.0
        r = analyzer().analyze(E_agcl, j)
        assert any("e_ref_vs_rhe" in w for w in r.tafel_warnings)

    def test_rhe_frame_does_not_warn(self):
        E, j = make_aor_lsv()
        r = analyzer().analyze(E, j)
        assert not any("e_ref_vs_rhe" in w for w in r.tafel_warnings)


# ═══ Integration sanity: Tafel slope itself is recovered ═════════════════════

class TestTafelRecovery:
    def test_known_slope_recovered_on_clean_curve(self):
        E, j = make_aor_lsv(slope_v_per_dec=0.120)
        r = analyzer(blank_subtracted=True).analyze(E, j)
        assert np.isfinite(r.tafel_slope)
        assert r.tafel_slope == pytest.approx(120.0, rel=0.15)
