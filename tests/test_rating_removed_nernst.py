"""
patch41 — B9: the literature "activity tier" ratings are removed (LSV performance_rating,
CV interpretation grade); one temperature-dependent Nernst slope (RT ln10 / F) for every
RHE conversion in app.py.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CATALYST_METAL_FREE, CVAnalyzer
from eisforge.analysis.lsv_analyzer import LSVAnalyzer

APP = Path(__file__).resolve().parents[1] / "app.py"


def _slope(t_c):
    return 8.314462618 * (273.15 + t_c) / 96485.33212 * math.log(10)


def _pt_cv():
    up = np.linspace(-0.2, 1.0, 241)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) <= 240
    i = np.where(fwd, 0.02, -0.02) + np.where(
        fwd, 0.5 * np.exp(-((e - 0.6) / 0.08) ** 2), 0.3 * np.exp(-((e - 0.4) / 0.06) ** 2))
    return e, i


# ── B9: ratings removed ──────────────────────────────────────────────────────
def test_lsv_result_has_no_rating():
    E = np.linspace(0.0, 1.4, 800)
    x = 10 ** ((E - 0.70) / 0.120)
    r = LSVAnalyzer(electrolyte="KOH", catalyst_type=CATALYST_METAL_FREE).analyze(
        E, 25.0 * x / (1.0 + x))
    assert not hasattr(r, "performance_rating")
    assert "Rating" not in r.summary()
    assert not hasattr(LSVAnalyzer, "_rate_performance")


def test_cv_interpretation_reports_onset_without_grade():
    r = CVAnalyzer(scan_rate=50.0, electrode_area=1.0).analyze(*_pt_cv())
    assert np.isfinite(r.e_onset)
    assert f"E_onset = {r.e_onset:.3f} V vs reference electrode" in r.interpretation
    for grade in ("Excellent activity", "Moderate activity", "High overpotential"):
        assert grade not in r.interpretation


# ── one Nernst slope ─────────────────────────────────────────────────────────
def test_app_has_one_nernst_slope():
    src = APP.read_text(encoding="utf-8")
    assert "def _nernst_slope(" in src
    assert "8.314 * (273.15" not in src, "use _nernst_slope()"
    assert not re.search(r"0\.059\s*[*×]", src), "use _nernst_slope(), not a fixed 0.059"


def test_nernst_slope_value():
    assert _slope(25.0) == pytest.approx(0.05916, abs=1e-5)


# ── app (AppTest) ────────────────────────────────────────────────────────────
def _apptest():
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _idf(E, i_amp, method="CyclicVoltammetry", sr="0.05"):
    rows = "\n".join(f"{a:.5E} {b:.5E} {a:.5E}" for a, b in zip(E, i_amp))
    head = f"Method={method}\nScanrate={sr}\nCurrent Range=1mA\nApply wrt OCP=false\n"
    return (head + f"primary_data\n3\n{len(E)}\n{rows}\n").encode("latin-1")


def test_cv_tab_uses_temperature_dependent_slope():
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    at.selectbox[[s.label for s in at.selectbox].index("Reference electrode")].select(
        "Ag/AgCl (3M KCl)")
    at.run()
    at.checkbox[[c.label for c in at.checkbox].index("Override pH manually")].check()
    at.run()
    at.number_input[[n.label for n in at.number_input].index("pH (manual)")].set_value(13.0)
    at.number_input[[n.label for n in at.number_input].index("Temperature (°C)")].set_value(50)
    at.run()
    e, i = _pt_cv()
    at.file_uploader(key="cv_up").upload("pt.idf", _idf(e, i * 1e-3)).run()
    assert len(at.exception) == 0, at.exception
    r = at.session_state["cv_r"]
    m = {x.label: x.value for x in at.metric}
    expected = r.e_onset + 0.210 + _slope(50.0) * 13.0
    assert float(m["E_onset (vs RHE)"].split()[0]) == pytest.approx(expected, abs=2e-4)


def test_lsv_tab_shows_no_performance_line():
    E = np.round(np.arange(0.0, 1.4, 0.005), 4)
    x = 10 ** ((E - 0.70) / 0.120)
    j = 25.0 * x / (1.0 + x)
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    at.file_uploader(key="lsv_up").upload("lsv.idf",
                                          _idf(E, j * 1e-3, "LinearSweep", "0.005")).run()
    assert len(at.exception) == 0, at.exception
    assert "lsv_r" in at.session_state
    assert not any("Performance" in s.value for s in at.success)
