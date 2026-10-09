"""
patch36 (B4): CV peak search with a prominence threshold, explicit "no anodic peak",
C_dl from the CV half-width in a double-layer window, no array-edge linear background.

Design borrowed from: ixdat CyclicVoltammogram.calc_capacitance (C = half-width / nu),
PyVoltammetry FittingDoubleLayer (straight double-layer baseline), voltcycle (split the
branches and trim the edges before peak search). Real-file tests need EISFORGE_LOCAL_DATA.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CVAnalyzer
from eisforge.parsers.ivium_parser import IviumIDFParser

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")
I_CAP = 0.02            # mA, capacitive half-width of the synthetic CVs
NU = 0.05               # V/s  -> C_dl = I_CAP / NU = 0.4 mF on 1 cm2


def _cv(kind, noise=0.002):
    up = np.linspace(-0.2, 1.0, 241)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) <= 240
    i = np.where(fwd, I_CAP, -I_CAP) + np.random.default_rng(0).normal(0, noise, len(e))
    if kind == "pt":
        i += np.where(fwd, 0.5 * np.exp(-((e - 0.6) / 0.08) ** 2), 0.3 * np.exp(-((e - 0.4) / 0.06) ** 2))
    elif kind == "carbon_peak":
        i += np.where(fwd, 0.06 * np.exp(-((e - 0.8) / 0.07) ** 2), 0.0)
    elif kind == "carbon_rising":       # rises to the anodic vertex: no peak
        i += np.where(fwd, 0.08, 0.04) / (1 + np.exp(-(e - 0.9) / 0.04))
    return e, i


def _carbon(e, i, **kw):
    return CVAnalyzer(scan_rate=NU * 1e3, electrode_area=1.0,
                      catalyst_type="carbon_material", **kw).analyze(e, i)


def test_no_peak_is_reported_as_such():
    r = _carbon(*_cv("carbon_rising"))
    assert r.peak_found is False
    assert math.isnan(r.e_onset) and "undefined" in r.e_onset_method
    assert math.isnan(r.net_faradaic_current_mA)
    assert r.interpretation.startswith("No anodic peak")
    assert r.cdl_mF_cm2 == pytest.approx(I_CAP / NU, rel=0.05)


def test_real_carbon_peak_found_and_net_current():
    r = _carbon(*_cv("carbon_peak"))
    assert r.peak_found is True
    assert r.e_forward_peak == pytest.approx(0.8, abs=0.02)
    assert r.net_faradaic_current_mA == pytest.approx(0.06, rel=0.15)
    assert np.isfinite(r.e_onset) and r.e_onset < r.e_forward_peak
    assert r.cdl_mF_cm2 == pytest.approx(I_CAP / NU, rel=0.05)


def test_noble_metal_peak_unchanged():
    e, i = _cv("pt")
    r = CVAnalyzer(scan_rate=50, electrode_area=1.0, catalyst_type="noble_metal").analyze(e, i)
    assert r.peak_found is True
    assert r.e_forward_peak == pytest.approx(0.6, abs=0.02)
    assert np.isfinite(r.e_onset)


def test_explicit_dl_window_is_used_and_reported():
    r = _carbon(*_cv("carbon_peak"), dl_window=(0.0, 0.3))
    assert r.dl_window == pytest.approx((0.0, 0.3))
    assert r.cdl_mF_cm2 == pytest.approx(I_CAP / NU, rel=0.05)


def test_noise_bump_is_not_a_peak():
    """Without the noise criterion a 5%-prominence search found a fake peak here."""
    r = _carbon(*_cv("carbon_rising", noise=0.004))
    assert r.peak_found is False


# ── real files (private; skipped unless EISFORGE_LOCAL_DATA is set) ───────────
REAL = [("koh_50mVs.idf", 0.230), ("koh_ipa_50mVs.idf", 0.124)]   # C_dl mF/cm2, area 0.1256


@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
@pytest.mark.parametrize("fname, cdl", REAL)
def test_real_files_no_peak_and_cdl(fname, cdl):
    p = Path(LOCAL) / fname
    if not p.exists():
        pytest.skip(f"{p} not found")
    ds = IviumIDFParser().parse(p)
    r = CVAnalyzer(scan_rate=50, electrode_area=0.1256,
                   catalyst_type="carbon_material").analyze(np.asarray(ds.z_real), np.asarray(ds.z_imag))
    assert r.peak_found is False                     # before patch36: fake peak at 0.13 / 0.33 V_RHE
    assert math.isnan(r.e_onset)
    assert r.cdl_mF_cm2 == pytest.approx(cdl, abs=0.01)


# ── app: no crash and an explicit warning when there is no peak ───────────────
def _idf_bytes(e, i_mA):
    rows = "\n".join(f"{a:.5E} {b * 1e-3:.5E} {a:.5E}" for a, b in zip(e, i_mA))
    head = ("Method=CyclicVoltammetry\nScanrate=0.05\nN scans=1\nCurrent Range=1mA\n"
            "Apply wrt OCP=false\n")
    return (head + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1")


def test_app_cv_tab_no_peak_carbon():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    sb = [s for s in at.selectbox if s.label == "Catalyst type"][0]
    sb.select("Carbon Material (N-doped C, CNT, Graphene)").run()
    at.file_uploader(key="cv_up").upload("rising.idf", _idf_bytes(*_cv("carbon_rising"))).run()
    assert len(at.exception) == 0, f"app crashed: {at.exception}"
    assert any("No anodic peak" in w.value for w in at.warning)
    onset = [m for m in at.metric if m.label == "E_onset (vs ref)"]
    assert onset and onset[0].value == "—"
