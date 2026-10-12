"""
patch40 — one RHE-shift helper in app.py (B8: no pH term when the data are already vs
RHE), LSV E_onset line drawn in the display frame (B10), and no zero-cross onset
without a sustained anodic current.

Reference model (idea from unitpackage/echemdb): every reference has a value vs SHE,
RHE is at -0.059*pH vs SHE, so E_RHE = E_meas + E_ref + 0.059*pH, and + 0 for RHE.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CATALYST_METAL_FREE
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")


# ── static guard: no hand-written RHE shift left in app.py ──────────────────
def test_no_hand_written_rhe_shift_left():
    src = APP.read_text(encoding="utf-8")
    left = re.findall(r"e_ref_val\s*\+\s*(?:0\.059|_nernst\w*)\s*\*\s*ph_value", src)
    assert left == [], f"use _to_rhe_shift() instead of: {left}"
    assert "def _to_rhe_shift(" in src


# ── zero-cross onset (library) ───────────────────────────────────────────────
def _lsv(**kw):
    d = dict(scan_rate=5.0, electrode_area=1.0, electrolyte="KOH",
             electrolyte_concentration=1.0, catalyst_type=CATALYST_METAL_FREE)
    d.update(kw)
    return LSVAnalyzer(**d)


def test_zero_cross_needs_a_sustained_current():
    E = np.linspace(0.0, 1.3, 200)
    j = -0.015 * np.exp(-E / 0.25) + 0.002 * E      # turns positive, stays < 0.01 mA/cm2
    r = _lsv().analyze(E, j)
    assert not r.onset_found
    assert np.isnan(r.e_onset_zerocross)


def test_zero_cross_kept_for_a_real_wave():
    E = np.linspace(0.0, 1.4, 800)
    x = 10 ** ((E - 0.70) / 0.120)
    j = 25.0 * x / (1.0 + x) - 0.01                 # small cathodic offset at low E
    r = _lsv().analyze(E, j)
    assert r.onset_found
    assert np.isfinite(r.e_onset_zerocross)


# ── app (AppTest) ────────────────────────────────────────────────────────────
def _apptest():
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _idf(E, i_amp, method="CyclicVoltammetry", sr="0.05"):
    rows = "\n".join(f"{a:.5E} {b:.5E} {a:.5E}" for a, b in zip(E, i_amp))
    head = (f"Method={method}\nScanrate={sr}\nCurrent Range=1mA\nApply wrt OCP=false\n")
    return (head + f"primary_data\n3\n{len(E)}\n{rows}\n").encode("latin-1")


def _sidebar(at, ref, ph=None):
    at.selectbox[[s.label for s in at.selectbox].index("Reference electrode")].select(ref)
    at.run()
    if ph is not None:
        at.checkbox[[c.label for c in at.checkbox].index("Override pH manually")].check()
        at.run()
        at.number_input[[n.label for n in at.number_input].index("pH (manual)")].set_value(ph)
        at.run()


def _cv_with_peak():
    up = np.linspace(-0.2, 1.0, 241)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) <= 240
    i = np.where(fwd, 0.02, -0.02) + np.where(
        fwd, 0.5 * np.exp(-((e - 0.6) / 0.08) ** 2), 0.3 * np.exp(-((e - 0.4) / 0.06) ** 2))
    return e, i * 1e-3                                   # mA -> A for the .idf file


@pytest.mark.parametrize("ref, shift", [("RHE", 0.0), ("Ag/AgCl (3M KCl)", 0.210 + 0.059159 * 13.0)])
def test_cv_tab_e_onset_vs_rhe(ref, shift):
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    _sidebar(at, ref, ph=13.0)
    at.file_uploader(key="cv_up").upload("pt.idf", _idf(*_cv_with_peak())).run()
    assert len(at.exception) == 0, at.exception
    r = at.session_state["cv_r"]
    assert np.isfinite(r.e_onset)
    m = {x.label: x.value for x in at.metric}
    assert float(m["E_onset (vs RHE)"].split()[0]) == pytest.approx(r.e_onset + shift, abs=2e-4)


def _lsv_chart_vline_x(at):
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        for shp in spec.get("layout", {}).get("shapes", []):
            if shp.get("line", {}).get("dash") == "dash" and shp.get("x0") == shp.get("x1"):
                return float(shp["x0"])
    return None


def test_lsv_onset_line_in_display_frame():
    E = np.round(np.arange(-0.8, 0.9001, 0.005), 4)              # vs Ag/AgCl
    j = 1e-4 * np.exp(E / 0.12)                                  # mA/cm2, area 1
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    _sidebar(at, "Ag/AgCl (3M KCl)")
    at.file_uploader(key="lsv_up").upload("lsv.idf",
                                          _idf(E, j * 1e-3, "LinearSweep", "0.005")).run()
    assert len(at.exception) == 0, at.exception
    r = at.session_state["lsv_r"]
    assert r.onset_found
    x = _lsv_chart_vline_x(at)
    assert x is not None, "E_onset line not found in the LSV chart"
    # default scale is "vs Ag/AgCl": the line must sit at the measured potential
    assert x == pytest.approx(r.e_onset - 0.210, abs=1e-6)


def test_lsv_no_onset_line_without_onset():
    E = np.linspace(-0.8, 0.5, 200)
    j = -0.015 * np.exp(-(E + 0.8) / 0.25) + 0.002 * (E + 0.8)
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    at.file_uploader(key="lsv_up").upload("lsv.idf",
                                          _idf(E, j * 1e-3, "LinearSweep", "0.005")).run()
    assert len(at.exception) == 0, at.exception
    assert not at.session_state["lsv_r"].onset_found
    assert _lsv_chart_vline_x(at) is None


# ── real data (private) ──────────────────────────────────────────────────────
@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
def test_real_metal_free_lsv_has_no_zero_cross():
    from eisforge.data.cv_data import CVData
    p = Path(LOCAL) / "koh_ipa_lsv_5mVs.idf"
    if not p.exists():
        pytest.skip("private file not found")
    S = CVData.from_ivium(p)
    r = _lsv(electrode_area=0.1256, e_ref_vs_rhe=0.210).analyze(S.potential_V, S.current_mA)
    assert not r.onset_found
    assert np.isnan(r.e_onset_zerocross)          # was 0.769 V_RHE before patch40
