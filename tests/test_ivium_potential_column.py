"""
patch34 (B2) — Ivium .idf potential is read vs the reference electrode.

With "Apply wrt OCP=true", column 1 is the potential relative to OCP and column 3
is the absolute potential vs the reference electrode (= column 1 + OCP).

* App path: AppTest uploads an .idf into the CV tab (key "cv_up"); the loader
  actually used by app.py is exercised.
* Package path: IviumIDFParser._parse_cv (EISDataset.z_real holds the potential).
* test_real_* run only when EISFORGE_LOCAL_DATA points to the author's private
  files (kept outside the repo); otherwise skipped.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from eisforge.parsers.ivium_parser import IviumIDFParser

APP = Path(__file__).resolve().parents[1] / "app.py"
LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")
OCP = -0.25


def _idf(wrt_ocp: bool, ncols: int = 3) -> bytes:
    e_up = np.round(np.arange(-0.8, 0.5001, 0.01), 5)
    e = np.concatenate([e_up, e_up[-2:0:-1]])                 # applied, wrt OCP if wrt_ocp
    i = 2.31e-6 * (e - e.min()) / (e.max() - e.min())         # A
    i[len(e_up):] -= 1.0e-6
    e_abs = e + (OCP if wrt_ocp else 0.0)
    if ncols >= 3:
        rows = "\n".join(f"{a:.5E} {b:.5E} {c:.5E}" for a, b, c in zip(e, i, e_abs))
    else:
        rows = "\n".join(f"{a:.5E} {b:.5E}" for a, b in zip(e, i))
    header = (
        "Method=CyclicVoltammetry\nScanrate=0.05\nN scans=1\nVertex 1=-0.8\nVertex 2=0.5\n"
        f"Current Range=100uA\nApply wrt OCP={'true' if wrt_ocp else 'false'}\n"
        "Apply wrt OCP.Record real E=false\n"
    )
    return (header + f"primary_data\n{ncols}\n{len(e)}\n{rows}\n").encode("latin-1")


def _app_upload(name: str, content: bytes):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.file_uploader(key="cv_up").upload(name, content).run()
    assert len(at.exception) == 0, f"app crashed: {at.exception}"
    assert "cv_pot" in at.session_state, "CV tab did not load the uploaded file"
    pot = np.asarray(at.session_state["cv_pot"], dtype=float)
    status = " ".join(s.value for s in at.success)
    warnings = " ".join(w.value for w in at.warning)
    return pot, status, warnings


# ── app loader ────────────────────────────────────────────────────────────────
def test_app_wrt_ocp_uses_absolute_column():
    pot, status, _ = _app_upload("wrt_ocp.idf", _idf(wrt_ocp=True))
    assert pot.min() == pytest.approx(-0.8 + OCP, abs=1e-6)
    assert pot.max() == pytest.approx(0.5 + OCP, abs=1e-6)
    assert "OCP -0.250 V" in status


def test_app_not_wrt_ocp_keeps_column_1():
    pot, status, _ = _app_upload("vs_ref.idf", _idf(wrt_ocp=False))
    assert pot.min() == pytest.approx(-0.8, abs=1e-6)
    assert pot.max() == pytest.approx(0.5, abs=1e-6)
    assert "OCP" not in status


def test_app_wrt_ocp_without_column_3_warns():
    pot, _status, warnings = _app_upload("two_cols.idf", _idf(wrt_ocp=True, ncols=2))
    assert pot.min() == pytest.approx(-0.8, abs=1e-6)          # cannot be corrected
    assert "relative to OCP" in warnings


# ── package parser ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("wrt_ocp, shift", [(True, OCP), (False, 0.0)])
def test_parser_potential_column(tmp_path, wrt_ocp, shift):
    p = tmp_path / "cv.idf"
    p.write_bytes(_idf(wrt_ocp=wrt_ocp))
    ds = IviumIDFParser().parse(p)
    assert ds.metadata["data_type"] == "CV"
    assert float(np.min(ds.z_real)) == pytest.approx(-0.8 + shift, abs=1e-6)
    if wrt_ocp:
        assert ds.metadata["ocp_V"] == pytest.approx(OCP, abs=1e-6)
    else:
        assert ds.metadata["ocp_V"] is None


# ── real files (private; skipped unless EISFORGE_LOCAL_DATA is set) ───────────
REAL = [
    # file, E_min, E_max (V vs Ag/AgCl 3 M KCl), OCP (V) — from the raw files
    ("koh_50mVs.idf", -1.08720, 0.212072, -0.2874),
    ("koh_ipa_50mVs.idf", -0.922795, 0.376479, -0.1227),
]


@pytest.mark.skipif(not LOCAL, reason="EISFORGE_LOCAL_DATA not set (private files)")
@pytest.mark.parametrize("fname, e_min, e_max, ocp", REAL)
def test_real_files_absolute_potential(fname, e_min, e_max, ocp):
    p = Path(LOCAL) / fname
    if not p.exists():
        pytest.skip(f"{p} not found")
    pot, status, _ = _app_upload(p.name, p.read_bytes())
    assert pot.min() == pytest.approx(e_min, abs=1e-5)
    assert pot.max() == pytest.approx(e_max, abs=1e-5)
    assert f"OCP {ocp:+.3f} V" in status
    ds = IviumIDFParser().parse(p)
    assert ds.metadata["ocp_V"] == pytest.approx(ocp, abs=5e-4)
