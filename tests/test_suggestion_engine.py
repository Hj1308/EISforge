"""
Tests for eisforge.catalogs.suggestion_engine — the multi-candidate,
AICc-ranked circuit suggestion feature (replaces the single fixed
suggestion from suggest_eec/lookup_circuit).
"""

import warnings

import numpy as np
import pytest
from impedance.models.circuits import CustomCircuit

from eisforge.catalogs.circuit_models import CircuitCatalog
from eisforge.catalogs.suggestion_engine import (
    candidate_models,
    default_p0,
    suggest_circuits,
    _aicc,
)

FREQ = np.logspace(-2, 5, 50)


def simulate(notation, params, freq=FREQ, noise=0.0, seed=0):
    c = CustomCircuit(circuit=notation, initial_guess=list(params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = c.predict(freq)
    if noise:
        rng = np.random.default_rng(seed)
        Z = Z + rng.normal(0, noise * np.abs(Z)) + 1j * rng.normal(0, noise * np.abs(Z))
    return Z


# ═══ candidate_models ═════════════════════════════════════════════════════

class TestCandidateModels:
    def test_returns_multiple_models_for_carbon(self):
        models = candidate_models("carbon_material")
        assert len(models) >= 3

    def test_no_duplicate_notations(self):
        models = candidate_models("carbon_material", inductive_loop=True)
        notations = [m.notation for m in models]
        assert len(notations) == len(set(notations))

    def test_inductive_hint_adds_pseudoinductive_not_replaces(self):
        base = candidate_models("carbon_material")
        with_ind = candidate_models("carbon_material", inductive_loop=True)
        assert CircuitCatalog.AOR_PSEUDOINDUCTIVE in with_ind
        # family defaults still present alongside it
        for m in base:
            assert m in with_ind

    def test_negative_resistance_hint_adds_ndr(self):
        models = candidate_models("noble_metal", negative_resistance=True)
        assert CircuitCatalog.AOR_NDR in models

    def test_ndr_takes_precedence_over_plain_inductive(self):
        models = candidate_models("noble_metal", inductive_loop=True,
                                  negative_resistance=True)
        assert CircuitCatalog.AOR_NDR in models
        assert CircuitCatalog.AOR_PSEUDOINDUCTIVE not in models


# ═══ default_p0 ═══════════════════════════════════════════════════════════

class TestDefaultP0:
    def test_p0_length_matches_circuit(self):
        Z = simulate("R0-p(R1,CPE1)", [10, 60, 2e-4, 0.9])
        p0 = default_p0(CircuitCatalog.RANDLES_SIMPLE, FREQ, Z.real, -Z.imag)
        assert len(p0) == 4

    def test_series_r_scaled_to_high_frequency_real_part(self):
        Z = simulate("R0-p(R1,CPE1)", [15, 60, 2e-4, 0.9])
        p0 = default_p0(CircuitCatalog.RANDLES_SIMPLE, FREQ, Z.real, -Z.imag)
        hf_idx = int(np.argmax(FREQ))
        assert p0[0] == pytest.approx(Z.real[hf_idx], rel=0.3)

    def test_cpe_exponent_in_valid_range(self):
        Z = simulate("R0-p(R1,CPE1)", [10, 60, 2e-4, 0.9])
        p0 = default_p0(CircuitCatalog.RANDLES_SIMPLE, FREQ, Z.real, -Z.imag)
        assert 0.0 < p0[3] <= 1.0

    def test_p0_produces_finite_impedance(self):
        Z = simulate("R0-p(CPE1,R1-p(R2,L2))", [8, 2e-4, 0.9, 60, 40, 30])
        p0 = default_p0(CircuitCatalog.AOR_PSEUDOINDUCTIVE, FREQ, Z.real, -Z.imag)
        c = CustomCircuit(circuit=CircuitCatalog.AOR_PSEUDOINDUCTIVE.notation,
                          initial_guess=p0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Zp = c.predict(FREQ)
        assert np.all(np.isfinite(Zp))


# ═══ AICc arithmetic ══════════════════════════════════════════════════════

class TestAICc:
    def test_more_params_penalised_at_equal_rss(self):
        n = 100
        rss = 10.0
        a_simple = _aicc(rss, n, k_params=4)
        a_complex = _aicc(rss, n, k_params=8)
        assert a_complex > a_simple

    def test_lower_rss_wins_at_equal_k(self):
        n = 100
        a_good = _aicc(1.0, n, k_params=5)
        a_bad = _aicc(10.0, n, k_params=5)
        assert a_good < a_bad

    def test_small_sample_correction_increases_aic(self):
        # AICc >= AIC always (for n > k+1)
        n, k, rss = 20, 6, 5.0
        aic = 2 * k + n * np.log(rss / n)
        aicc = _aicc(rss, n, k)
        assert aicc >= aic


# ═══ suggest_circuits: end-to-end ranking ═══════════════════════════════════

class TestSuggestCircuitsRanking:
    def test_two_rc_data_ranks_two_rc_first(self):
        Z = simulate("R0-p(R1,CPE1)-p(R2,CPE2)",
                    [10, 60, 2e-4, 0.9, 800, 1e-4, 0.8], noise=0.01, seed=1)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal")
        assert results[0].model.notation == "R0-p(R1,CPE1)-p(R2,CPE2)"
        assert results[0].converged
        assert results[0].delta_aicc == 0.0

    def test_inductive_data_with_hint_ranks_inductive_first(self):
        Z = simulate("R0-p(CPE1,R1-p(R2,L2))", [8, 2e-4, 0.9, 60, 40, 30],
                    noise=0.008, seed=2)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="carbon_material",
                                   inductive_loop=True)
        assert results[0].model is CircuitCatalog.AOR_PSEUDOINDUCTIVE
        assert results[0].delta_aicc == 0.0
        # a plain 2-RC model should be decisively worse
        two_rc = next(r for r in results
                     if r.model.notation == "R0-p(R1,CPE1)-p(R2,CPE2)")
        assert two_rc.delta_aicc > 10

    def test_without_hint_inductive_not_offered(self):
        Z = simulate("R0-p(R1,CPE1)", [10, 60, 2e-4, 0.9], noise=0.01, seed=3)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal")
        assert all(r.model is not CircuitCatalog.AOR_PSEUDOINDUCTIVE
                  for r in results)

    def test_results_sorted_ascending_aicc(self):
        Z = simulate("R0-p(R1,CPE1)", [10, 60, 2e-4, 0.9], noise=0.01, seed=4)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal")
        aiccs = [r.aicc for r in results if np.isfinite(r.aicc)]
        assert aiccs == sorted(aiccs)

    def test_ndr_requires_hint_and_uses_negative_bounds(self):
        Z = simulate("R0-p(CPE1,R1-p(R2,L2))", [8, 2e-4, 0.9, 60, -90, 30],
                    noise=0.005, seed=5)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal",
                                   negative_resistance=True)
        best = results[0]
        assert best.model is CircuitCatalog.AOR_NDR
        assert best.converged
        r2 = best.fit_result.parameters["R2"]
        assert r2 < 0

    def test_support_label_thresholds(self):
        Z = simulate("R0-p(R1,CPE1)-p(R2,CPE2)",
                    [10, 60, 2e-4, 0.9, 800, 1e-4, 0.8], noise=0.01, seed=6)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal")
        assert results[0].support_label() == "essentially equivalent support"
        assert any(r.support_label() == "no support" for r in results[1:])

    def test_str_repr_is_informative(self):
        Z = simulate("R0-p(R1,CPE1)", [10, 60, 2e-4, 0.9], noise=0.01, seed=7)
        results = suggest_circuits(FREQ, Z.real, -Z.imag,
                                   catalyst_type="noble_metal")
        s = str(results[0])
        assert "AICc" in s and results[0].model.name in s
