"""
patch38 — blank CV uploader in the CV tab (key "cv_blank_up") and the
"blank dominates" warning of CVData.subtract.

The app tests run the real app.py through streamlit.testing.v1.AppTest (pytest
alone cannot see app.py widget bugs — see AGENTS.md). Real-pair tests run only
when EISFORGE_LOCAL_DATA points to the private data folder.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.data.cv_data import CVData

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")


def _loop(offset_mA=0.0, scale=1.0):
    up = np.round(np.arange(-0.8, 0.5001, 0.01), 5)
    e = np.concatenate([up, up[-2::-1]])
    fwd = np.arange(len(e)) < len(up)
    i = scale * (np.where(fwd, 0.002, -0.002) + 0.001 * e) + offset_mA
    return CVData(e, i, scan_rate_mV_s=50.0)


def _idf(i_offset_a=0.0, i_scale=1.0) -> bytes:
    """Minimal Ivium-style .idf: one closed CV cycle, current column in amperes."""
    up = np.round(np.arange(-0.8, 0.5001, 0.01), 5)
    e = np.concatenate([up, up[-2:0:-1]])
    fwd = np.arange(len(e)) < len(up)
    i = i_scale * (np.where(fwd, 2e-6, -2e-6) + 1e-6 * e) + i_offset_a
    rows = "\n".join(f"{a:.5E} {b:.5E} {a:.5E}" for a, b in zip(e, i))
    head = ("Method=CyclicVoltammetry\nScanrate=0.05\nN scans=1\nVertex 1=-0.8\n"
            "Vertex 2=0.5\nCurrent Range=100uA\nApply wrt OCP=false\n")
    return (head + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1")


# ── library: blank-dominates warning ─────────────────────────────────────────
def test_comparable_blank_gives_no_dominance_warning():
    r = _loop(offset_mA=0.0005).subtract(_loop())
    assert r.blank_dominant_fraction < 0.5
    assert not any("more than twice" in w for w in r.warnings)


def test_blank_ten_times_larger_is_flagged():
    r = _loop().subtract(_loop(scale=10.0))
    assert r.blank_dominant_fraction > 0.9
    assert any("more than twice the sample current" in w for w in r.warnings)


# ── app: CV tab with a blank ─────────────────────────────────────────────────
def _apptest():
    # skipped where streamlit is not installed (CI); the library tests above still run
    return pytest.importorskip("streamlit.testing.v1").AppTest


def _run(sample: bytes, blank: bytes):
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    assert len(at.exception) == 0, f"app crashed before upload: {at.exception}"
    at.file_uploader(key="cv_up").upload("alcohol.idf", sample)
    at.file_uploader(key="cv_blank_up").upload("blank.idf", blank).run()
    assert len(at.exception) == 0, f"app crashed after upload: {at.exception}"
    return at


def test_app_net_current_constant_offset():
    at = _run(_idf(i_offset_a=0.5e-6), _idf())
    net = np.asarray(at.session_state["cv_net"], dtype=float)
    assert np.all(np.isfinite(net))
    assert np.allclose(net, 0.5e-3, atol=1e-9)          # 0.5 uA = 5e-4 mA everywhere
    labels = [m.label for m in at.metric]
    assert "Max net j (anodic sweep)" in labels
    assert "Shared window" in labels


def test_app_identical_blank_warns_no_oxidation():
    at = _run(_idf(), _idf())
    assert np.allclose(np.asarray(at.session_state["cv_net"], dtype=float), 0.0)
    assert any("not positive anywhere" in w.value for w in at.warning)


def test_app_without_blank_has_no_net_section():
    at = _apptest().from_file(str(APP), default_timeout=120).run()
    at.file_uploader(key="cv_up").upload("alcohol.idf", _idf()).run()
    assert len(at.exception) == 0
    assert "cv_net" not in at.session_state
    assert "Max net j (anodic sweep)" not in [m.label for m in at.metric]


# ── real pairs (private data) ────────────────────────────────────────────────
@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
@pytest.mark.parametrize("sample, blank, expected", [
    ("koh_ipa_50mVs.idf", "koh_50mVs.idf", 0.11),
    ("koh_ipa_lsv_5mVs.idf", "koh_lsv_5mVs.idf", 0.14),
])
def test_real_valid_pairs_not_flagged(sample, blank, expected):
    s, b = Path(LOCAL) / sample, Path(LOCAL) / blank
    if not (s.exists() and b.exists()):
        pytest.skip("private files not found")
    r = CVData.from_ivium(s).subtract(CVData.from_ivium(b))
    assert r.blank_dominant_fraction == pytest.approx(expected, abs=0.02)
    assert not any("more than twice" in w for w in r.warnings)


@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
def test_real_pair_in_app():
    s, b = Path(LOCAL) / "koh_ipa_50mVs.idf", Path(LOCAL) / "koh_50mVs.idf"
    if not (s.exists() and b.exists()):
        pytest.skip("private files not found")
    at = _run(s.read_bytes(), b.read_bytes())
    texts = " ".join(w.value for w in at.warning)
    assert "OCP differs by" in texts                 # patch37 warning reaches the UI
    assert "not positive anywhere" in texts          # IPA gives no current above the blank
