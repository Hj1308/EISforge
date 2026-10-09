"""
patch32 (B1) — Ivium .idf CV current is read in amperes and converted to mA.

Runs the real app.py through streamlit.testing.v1.AppTest and uploads an .idf
file into the CV tab (key "cv_up"), so the loader actually used by the app is
exercised (pytest alone cannot see app.py widget/loader bugs — see AGENTS.md).

* test_synthetic_* always runs (no private data needed; CI-safe).
* test_real_* runs only when the environment variable EISFORGE_LOCAL_DATA points
  to a folder that contains the author's unpublished files (kept outside the
  repo until approved for publication); otherwise it is skipped.
"""

import os
from pathlib import Path

import numpy as np
import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")


def _synthetic_idf(current_range: str = "100uA", i_peak_a: float = 2.31e-6) -> bytes:
    """Minimal Ivium-style .idf: one CV cycle, current column in amperes."""
    e_up = np.round(np.arange(-0.8, 0.5001, 0.01), 5)
    e = np.concatenate([e_up, e_up[-2:0:-1]])
    i = i_peak_a * (e - e.min()) / (e.max() - e.min())          # 0 .. i_peak_a (A)
    i[len(e_up):] -= 1.0e-6                                     # cathodic return
    rows = "\n".join(f"{a:.5E} {b:.5E} {a - 0.25:.5E}" for a, b in zip(e, i))
    header = (
        "Method=CyclicVoltammetry\nScanrate=0.05\nN scans=1\n"
        f"Vertex 1=-0.8\nVertex 2=0.5\nCurrent Range={current_range}\n"
        "Apply wrt OCP=true\n"
    )
    return (header + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1")


def _run_cv_upload(name: str, content: bytes):
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert len(at.exception) == 0, f"app crashed before upload: {at.exception}"
    at.file_uploader(key="cv_up").upload(name, content).run()
    assert len(at.exception) == 0, f"app crashed after upload: {at.exception}"
    assert "cv_cur" in at.session_state, "CV tab did not load the uploaded file"
    return at, np.asarray(at.session_state["cv_cur"], dtype=float)


@pytest.mark.parametrize("current_range", ["100uA", "1mA", "10nA"])
def test_synthetic_idf_current_is_amperes_to_mA(current_range):
    _at, cur = _run_cv_upload("synthetic.idf", _synthetic_idf(current_range))
    # 2.31e-6 A must become 2.31e-3 mA whatever the hardware range says
    assert cur.max() == pytest.approx(2.31e-3, rel=1e-3)


@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
def test_real_koh_blank_50mVs():
    p = Path(LOCAL) / "koh_50mVs.idf"
    if not p.exists():
        pytest.skip(f"{p} not found")
    _at, cur = _run_cv_upload(p.name, p.read_bytes())
    # raw file: max +2.30957E-06 A, min -5.04540E-06 A (1 M KOH, 50 mV/s, Ag/AgCl 3 M KCl)
    assert cur.max() == pytest.approx(2.30957e-3, abs=1e-6)
    assert cur.min() == pytest.approx(-5.04540e-3, abs=1e-6)
