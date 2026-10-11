"""
patch39 — LSV E_onset only when a sustained anodic current exists (B6), and the
LSV tab's vs-RHE conversion no longer adds the reference offset twice (B7).

Real-file tests run only when EISFORGE_LOCAL_DATA points to the private data folder.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CATALYST_METAL_FREE
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")


def _analyzer(**kw):
    d = dict(scan_rate=5.0, electrode_area=1.0, electrolyte="KOH",
             electrolyte_concentration=1.0, catalyst_type=CATALYST_METAL_FREE)
    d.update(kw)
    return LSVAnalyzer(**d)


def _aor(n=800):
    """Same AOR-like curve as tests/test_lsv_fixes.make_aor_lsv (foot, peak 25 mA/cm2)."""
    E = np.linspace(0.0, 1.4, n)
    x = 10 ** ((E - 0.70) / 0.120)
    j = 25.0 * x / (1.0 + x)
    return E, j * np.where(E > 1.05, np.exp(-((E - 1.05) ** 2) / (2 * 0.12 ** 2)), 1.0)


def _no_wave(n=200):
    """Metal-free-like scan without faradaic wave: cathodic O2-reduction-like
    current at low E (-15 uA/cm2) relaxing to a few uA/cm2 (all < 0.01 mA/cm2)."""
    E = np.linspace(0.0, 1.3, n)
    j = -0.015 * np.exp(-E / 0.25) + 0.002 * E
    return E, j


# ── library ──────────────────────────────────────────────────────────────────
def test_real_wave_still_has_onset():
    r = _analyzer().analyze(*_aor())
    assert r.onset_found
    assert np.isfinite(r.e_onset)
    assert "not reached" not in r.e_onset_method
    assert not r.performance_rating.startswith("Not rated")


def test_no_wave_gives_no_onset():
    r = _analyzer().analyze(*_no_wave())
    assert not r.onset_found
    assert np.isnan(r.e_onset)
    assert "not reached" in r.e_onset_method
    assert r.performance_rating.startswith("Not rated")
    assert any("No sustained anodic current" in w for w in r.tafel_warnings)


def test_cathodic_current_is_not_flipped_into_an_onset():
    # the old magnitude rule flipped this curve (|j| larger at low E) and found a "rise"
    E = np.linspace(0.0, 1.3, 200)
    j = -0.75 * np.exp(-E / 0.25) + 0.002 * E   # -0.75 mA/cm2 cathodic, anodic < 0.003
    r = _analyzer().analyze(E, j)
    assert not r.onset_found


def test_polarographic_convention():
    E, j = _aor()
    assert _analyzer(current_convention="polarographic").analyze(E, -j).onset_found
    assert not _analyzer().analyze(E, -j).onset_found          # IUPAC default


def test_threshold_is_user_settable():
    E, j = _aor()
    assert not _analyzer(onset_min_j=50.0).analyze(E, j).onset_found    # peak is 25


def test_single_spike_is_not_sustained():
    E = np.linspace(0, 1, 50)
    j = np.zeros(50)
    j[20] = 1.0
    assert np.isnan(LSVAnalyzer._sustained_rise(E, j, 0.01, 5))
    j[30:36] = 1.0
    assert LSVAnalyzer._sustained_rise(E, j, 0.01, 5) == pytest.approx(E[30])


# ── app (AppTest) ────────────────────────────────────────────────────────────
def _idf(E, i_amp):
    rows = "\n".join(f"{a:.5E} {b:.5E} {a:.5E}" for a, b in zip(E, i_amp))
    head = ("Method=LinearSweep\nScanrate=0.005\nCurrent Range=1mA\nApply wrt OCP=false\n")
    return (head + f"primary_data\n3\n{len(E)}\n{rows}\n").encode("latin-1")


def _run_lsv(content, ref=None):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert len(at.exception) == 0
    if ref is not None:
        at.selectbox[[s.label for s in at.selectbox].index("Reference electrode")].select(ref)
        at.run()
    at.file_uploader(key="lsv_up").upload("lsv.idf", content).run()
    assert len(at.exception) == 0, at.exception
    return at


def test_app_no_wave_shows_warning_and_dash():
    E, j = _no_wave()
    at = _run_lsv(_idf(E, j * 1e-3))            # area 1 cm2: mA/cm2 -> A
    assert any("No sustained anodic current" in w.value for w in at.warning)
    m = {x.label: x.value for x in at.metric}
    assert m["E_onset"] == "—"


def test_app_rhe_help_adds_reference_once():
    # exponential anodic rise, recorded vs Ag/AgCl (3 M KCl): raw -0.8 .. 0.9 V
    E = np.round(np.arange(-0.8, 0.9001, 0.005), 4)
    j = 1e-4 * np.exp((E - 0.0) / 0.12)                        # mA/cm2
    at = _run_lsv(_idf(E, j * 1e-3), ref="Ag/AgCl (3M KCl)")
    r = at.session_state["lsv_r"]
    ph = None                                                  # sidebar caption "pH = **x**"
    for c in at.caption:
        if c.value.startswith("pH = **"):
            ph = float(c.value.split("**")[1])
    assert ph is not None, "pH caption not found"
    assert r.onset_found
    for x in at.metric:
        if x.label == "E_onset (j-threshold)":
            # value is in the analysis frame (E_meas + 0.210); RHE adds only 0.059 pH
            # (the caption pH is rounded to 2 decimals -> 3 mV tolerance)
            e_help = float(x.proto.help.split("=")[1].split("V")[0])
            assert e_help == pytest.approx(r.e_onset_threshold + 0.059 * ph, abs=0.003)
            assert abs(e_help - (r.e_onset_threshold + 0.210 + 0.059 * ph)) > 0.2   # B7
            break
    else:
        pytest.fail("E_onset (j-threshold) metric not shown")


# ── real data (private) ──────────────────────────────────────────────────────
@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
def test_real_metal_free_lsv_has_no_onset():
    from eisforge.data.cv_data import CVData
    s, b = Path(LOCAL) / "koh_ipa_lsv_5mVs.idf", Path(LOCAL) / "koh_lsv_5mVs.idf"
    if not (s.exists() and b.exists()):
        pytest.skip("private files not found")
    S, B = CVData.from_ivium(s), CVData.from_ivium(b)
    la = _analyzer(electrode_area=0.1256, e_ref_vs_rhe=0.210)
    assert not la.analyze(S.potential_V, S.current_mA).onset_found           # raw: max 2 uA/cm2
    En, In = LSVAnalyzer.subtract_blank(S.potential_V, S.current_mA,
                                        B.potential_V, B.current_mA, e_min=-0.1)
    r = la.analyze(En, In)                                                   # net < 0 everywhere
    assert not r.onset_found          # was E_onset = 1.024 V_RHE (tangent) before patch39
