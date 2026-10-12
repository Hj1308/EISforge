"""
patch44 (B5) — the backward-scan peak of alcohol oxidation is the anodic re-oxidation
peak; I_f/I_b is reported when both peaks exist and is not graded as "CO tolerance"
(Hofstead-Duffy et al. 2012; Chung et al. 2016).
"""

from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CVAnalyzer

APP = Path(__file__).resolve().parents[1] / "app.py"


def _cv(backward_peak=True, i_cap=0.02):
    up = np.linspace(-0.2, 1.0, 241)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) <= 240
    i = np.where(fwd, i_cap, -i_cap) + np.where(fwd, 0.5 * np.exp(-((e - 0.6) / 0.08) ** 2), 0.0)
    if backward_peak:
        i += np.where(fwd, 0.0, 0.3 * np.exp(-((e - 0.4) / 0.06) ** 2))
    return e, i


def test_textbook_pt_cv_gives_anodic_backward_peak_and_ratio():
    r = CVAnalyzer(scan_rate=50.0, electrode_area=1.0).analyze(*_cv())
    assert r.backward_peak_found
    assert r.i_backward_peak == pytest.approx(0.28, abs=0.01)     # 0.30 peak - 0.02 capacitive
    assert r.e_backward_peak == pytest.approx(0.40, abs=0.02)
    assert r.if_ib_ratio == pytest.approx(0.52 / 0.28, rel=0.05)  # was always NaN
    assert "not a CO-tolerance" in r.interpretation
    for old in ("CO tolerance —", "CO poisoning detected", "Severe CO poisoning"):
        assert old not in r.interpretation


def test_no_backward_peak_gives_nan():
    r = CVAnalyzer(scan_rate=50.0, electrode_area=1.0).analyze(*_cv(backward_peak=False))
    assert not r.backward_peak_found
    assert np.isnan(r.i_backward_peak) and np.isnan(r.if_ib_ratio)
    assert "No anodic backward-scan peak" in r.interpretation


def test_metal_free_still_has_no_ratio():
    r = CVAnalyzer(scan_rate=50.0, electrode_area=1.0,
                   catalyst_type="carbon_material").analyze(*_cv())
    assert np.isnan(r.if_ib_ratio)


def test_app_shows_ratio():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    e, i = _cv()
    rows = "\n".join(f"{a:.5E} {b * 1e-3:.5E} {a:.5E}" for a, b in zip(e, i))
    head = "Method=CyclicVoltammetry\nScanrate=0.05\nCurrent Range=1mA\nApply wrt OCP=false\n"
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.file_uploader(key="cv_up").upload(
        "pt.idf", (head + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1")).run()
    assert len(at.exception) == 0, at.exception
    m = {x.label: x.value for x in at.metric}
    assert float(m["I_f/I_b"]) == pytest.approx(0.52 / 0.28, rel=0.05)
    assert float(m["I_b"].split()[0]) == pytest.approx(0.28, abs=0.01)
