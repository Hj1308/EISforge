"""
patch35 (B3): CVAnalyzer decides the current sign convention from the CV loop area.

A passive electrode traces the I-E loop of a CV in one fixed direction, so the sign of
the closed-path integral of I dE tells whether the file is anodic-positive (IUPAC) or
cathodic-positive (polarographic). Explicit current_convention overrides the guess.
Real-file tests run only when EISFORGE_LOCAL_DATA points to the private data folder.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.cv_analyzer import CVAnalyzer
from eisforge.parsers.ivium_parser import IviumIDFParser

LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")


def _pt_like(ib_gt_if=False, start_at_top=False):
    """Closed CV, IUPAC sign: capacitive box + forward anodic peak at 0.6 V
    and backward anodic peak at 0.4 V (larger than forward if ib_gt_if)."""
    up = np.linspace(-0.2, 1.0, 121)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) <= 120
    a_f, a_b = (0.3, 0.45) if ib_gt_if else (0.5, 0.3)
    i = (np.where(fwd, 0.02, -0.02)
         + np.where(fwd, a_f * np.exp(-((e - 0.6) / 0.08) ** 2),
                    a_b * np.exp(-((e - 0.4) / 0.06) ** 2)))
    if start_at_top:                       # same loop, cycle starting at the upper vertex
        e, i = np.roll(e, -120), np.roll(i, -120)
    return e, i


def _analyze(e, i, **kw):
    return CVAnalyzer(scan_rate=50, electrode_area=1.0, catalyst_type="noble_metal",
                      **kw).analyze(e, i)


@pytest.mark.parametrize("ib_gt_if", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_auto_orientation_closed_cv(ib_gt_if, sign):
    e, i = _pt_like(ib_gt_if)
    r = _analyze(e, sign * i)
    assert r.current_sign_flipped is (sign == -1)
    assert "loop area" in r.current_convention_used
    assert r.e_forward_peak == pytest.approx(0.6, abs=0.02)
    assert r.i_forward_peak > 0


def test_auto_orientation_independent_of_cycle_start():
    e, i = _pt_like(start_at_top=True)
    assert _analyze(e, i).current_sign_flipped is False
    assert _analyze(e, -i).current_sign_flipped is True


def test_explicit_conventions_override_auto():
    e, i = _pt_like()
    r_iupac = _analyze(e, i, current_convention="iupac")
    assert r_iupac.current_sign_flipped is False and "iupac" in r_iupac.current_convention_used
    r_polar = _analyze(e, -i, current_convention="polarographic")
    assert r_polar.current_sign_flipped is True
    assert r_polar.e_forward_peak == pytest.approx(0.6, abs=0.02)


def test_invalid_convention_rejected():
    with pytest.raises(ValueError):
        CVAnalyzer(current_convention="american")


def test_open_path_keeps_pre_patch35_rule():
    e = np.linspace(-0.2, 1.0, 121)                       # single sweep (LSV-like)
    i = 0.02 + 0.5 / (1 + np.exp(-(e - 0.6) / 0.04))
    r = _analyze(e, i)
    assert "open path" in r.current_convention_used
    assert r.current_sign_flipped is False


# ── real files (private; skipped unless EISFORGE_LOCAL_DATA is set) ───────────
@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
@pytest.mark.parametrize("fname", ["koh_50mVs.idf", "koh_ipa_50mVs.idf"])
@pytest.mark.parametrize("sign", [1, -1])
def test_real_files_orientation(fname, sign):
    p = Path(LOCAL) / fname
    if not p.exists():
        pytest.skip(f"{p} not found")
    ds = IviumIDFParser().parse(p)                 # patch34: potential vs reference
    e, i_ma = np.asarray(ds.z_real), np.asarray(ds.z_imag)
    r = _analyze(e, sign * i_ma)
    assert r.current_sign_flipped is (sign == -1)
    # before patch35 the "peak" sat at the cathodic vertex (-1.087 / -0.923 V)
    assert r.e_forward_peak > 0.0
