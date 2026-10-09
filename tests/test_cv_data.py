"""
patch37 — CVData: explicit-unit CV container, sweep split by the sign of dE (ixdat
find_signed_sections idea) and branch-by-branch blank subtraction (ixdat diff_with idea)
with NaN outside the blank's range instead of silent clamping.
Real-pair tests run only when EISFORGE_LOCAL_DATA points to the private data folder.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.data.cv_data import CVData

LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")
OFF_RHE = 0.210 + 0.0592 * 14          # Ag/AgCl (3 M KCl) -> RHE in 1 M KOH


def _cv(lo=-0.8, hi=0.5, step=0.01, offset_mA=0.0, start_at_top=False, sr=50.0, ocp=None):
    up = np.round(np.arange(lo, hi + step / 2, step), 6)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) < len(up)
    i = np.where(fwd, 0.02, -0.02) + 0.01 * e + offset_mA
    if start_at_top:
        k = len(up) - 1
        e, i = np.concatenate([e[k:], e[1:k + 1]]), np.concatenate([i[k:], i[1:k + 1]])
    meta = {} if ocp is None else {"ocp_V": ocp}
    return CVData(e, i, scan_rate_mV_s=sr, meta=meta)


def test_rejects_mismatched_arrays():
    with pytest.raises(ValueError):
        CVData(np.zeros(5), np.zeros(4))


def test_branches_standard_and_top_start_and_lsv():
    assert [b.kind for b in _cv().branches()] == ["anodic", "cathodic"]
    assert [b.kind for b in _cv(start_at_top=True).branches()] == ["cathodic", "anodic"]
    lsv = CVData(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    assert [b.kind for b in lsv.branches()] == ["anodic"]


def test_vertex_belongs_to_both_branches():
    br = _cv().branches()
    assert br[0].index[-1] == br[1].index[0]


def test_subtract_identical_gives_zero():
    r = _cv().subtract(_cv())
    assert np.all(np.isfinite(r.net_mA))
    assert np.allclose(r.net_mA, 0.0)
    assert r.overlap_fraction == pytest.approx(1.0)
    assert r.warnings == []


def test_subtract_constant_offset_on_both_branches():
    r = _cv(offset_mA=0.005).subtract(_cv())
    assert np.allclose(r.net_mA, 0.005)


def test_partial_overlap_gives_nan_not_clamped_values():
    sample = _cv(lo=-0.8, hi=0.5)
    blank = _cv(lo=-0.3, hi=0.5)              # narrower window
    r = sample.subtract(blank)
    outside = sample.potential_V < -0.3 - 1e-9
    assert np.all(np.isnan(r.net_mA[outside]))          # ixdat/np.interp would clamp here
    assert np.all(np.isfinite(r.net_mA[~outside]))
    assert r.overlap_fraction == pytest.approx(0.8 / 1.3, abs=0.01)
    assert any("shared with the blank" in w for w in r.warnings)


def test_warnings_scan_rate_and_ocp():
    r = _cv(sr=50, ocp=-0.12).subtract(_cv(sr=10, ocp=-0.29))
    text = " ".join(r.warnings)
    assert "different scan rates" in text
    assert "OCP differs by 170 mV" in text


def test_from_ivium_synthetic(tmp_path):
    up = np.round(np.arange(-0.8, 0.5001, 0.01), 5)
    e = np.concatenate([up, up[-2::-1]])
    rows = "\n".join(f"{a:.5E} {1e-6:.5E} {a - 0.2:.5E}" for a in e)
    head = ("Method=CyclicVoltammetry\nScanrate=0.05\nCurrent Range=100uA\n"
            "Apply wrt OCP=true\n")
    p = tmp_path / "cv.idf"
    p.write_bytes((head + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1"))
    cv = CVData.from_ivium(p)
    assert cv.scan_rate_mV_s == pytest.approx(50.0)
    assert cv.potential_V.min() == pytest.approx(-1.0, abs=1e-6)       # column 3 (patch34)
    assert cv.current_mA.max() == pytest.approx(1e-3)                  # A -> mA (patch32)
    assert cv.meta["ocp_V"] == pytest.approx(-0.2, abs=1e-6)


@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
def test_real_pair_net_current():
    koh, ipa = Path(LOCAL) / "koh_50mVs.idf", Path(LOCAL) / "koh_ipa_50mVs.idf"
    if not (koh.exists() and ipa.exists()):
        pytest.skip("private files not found")
    sample, blank = CVData.from_ivium(ipa), CVData.from_ivium(koh)
    r = sample.subtract(blank)
    fwd = sample.branches()[0]
    e_rhe = sample.potential_V[fwd.index] + OFF_RHE
    net_uA = r.net_mA[fwd.index] * 1e3
    for e_target, expected in ((0.30, -0.613), (0.70, -0.236)):     # hand-computed earlier
        j = int(np.argmin(np.abs(e_rhe - e_target)))
        assert net_uA[j] == pytest.approx(expected, abs=0.02)
    assert np.nanmax(net_uA) < 0                     # no oxidation current above the blank
    assert r.overlap_fraction == pytest.approx(0.873, abs=0.01)
    assert any("OCP differs by 165 mV" in w for w in r.warnings)
