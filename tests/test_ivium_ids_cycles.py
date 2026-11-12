"""patch55 (ROADMAP B16): every cycle of a multi-cycle Ivium .ids file is read.

An .ids dataset stores each cycle as its own data object (a header that starts with a
``QR=`` line, then a ``primary_data`` block); a leading object with < 10 points is a
placeholder. Before patch55 only the first real block was read, with the placeholder's
header (so a CV came out as "LSV"). Idea: ixdat IviumDatasetReader (one segment per cycle,
explicit cycle number, joined by the caller).

The synthetic file below mimics that layout; the real-file test needs the private data.
"""
from pathlib import Path

import numpy as np
import pytest

from eisforge.data.cv_data import CVData
from eisforge.parsers.ivium_parser import IviumIDFParser
from tests._private import private_file, private_refs

DATA = Path(__file__).parent / "data" / "sample_eis.idf"


def _qr(label):
    f = ["QR", "1", "000000000000000V00000", "", "1", "0", "0", "user1", "100", "", "V00000",
         "0", "1", "1", "1", "832", "1", "0", label, "5.000E-02"]
    return "QR=" + ",".join(f)


def _obj(label, method, rows, scanrate="0.05"):
    body = "\n".join(f"{a:.5E} {b:.5E} {a:.5E}" for a, b in rows)
    return (f"{_qr(label)}\nPeaks=false\n\x00\x00\nMethod={method}\nTitle=\nScanrate={scanrate}\n"
            f"N scans=3\nApply wrt OCP=false\n\x00\nData Objects\n0\nprimary_data\n3\n"
            f"{len(rows)}\n{body}\n\n")


def _cycle(offset_a):
    up = np.round(np.arange(-0.2, 1.0001, 0.02), 5)
    e = np.concatenate([up, up[-2::-1]])
    return [(x, offset_a + 1e-6 * x) for x in e]


def _ids(tmp_path, n_cycles=3, rates=None, name="multi.ids"):
    rates = rates or ["0.05"] * n_cycles
    text = _obj("scan 1", "LinearSweep", [(0.0, 0.0)] * 4)          # placeholder object
    labels = ["scan 1"] + [f"cycle {k}" for k in range(2, n_cycles + 1)]
    for k, (lab, sr) in enumerate(zip(labels, rates)):
        text += _obj(lab, "CyclicVoltammetry", _cycle(1e-7 * k), sr)
    p = tmp_path / name
    p.write_bytes(text.encode("latin-1"))
    return p, len(_cycle(0))


def test_all_cycles_read_with_their_labels(tmp_path):
    p, m = _ids(tmp_path, 3)
    cv = CVData.from_ivium(p)
    assert cv.meta["data_type"] == "CV"                    # was "LSV" (placeholder header)
    assert cv.meta["cycle_bounds"] == [(0, m), (m, 2 * m), (2 * m, 3 * m)]
    assert cv.meta["cycle_labels"] == ["scan 1", "cycle 2", "cycle 3"]
    assert len(cv.potential_V) == 3 * m
    assert cv.scan_rate_mV_s == pytest.approx(50.0)


@pytest.mark.parametrize("idx, used", [(0, 1), (1, 2), (2, 3), (-1, 3), (99, 3)])
def test_select_cycle_returns_the_recorded_cycle(tmp_path, idx, used):
    p, m = _ids(tmp_path, 3)
    c = CVData.from_ivium(p).select_cycle(idx)
    assert (c.meta["n_cycles"], c.meta["cycle_used"]) == (3, used)
    assert len(c.potential_V) == m
    # cycle k has current offset 1e-7 A * (k - 1) -> 1e-4 mA * (k - 1) at E = 0
    j0 = int(np.argmin(np.abs(c.potential_V)))
    assert c.current_mA[j0] == pytest.approx(1e-4 * (used - 1), abs=1e-9)
    assert "cycle_bounds" not in c.meta                    # a single cycle has no bounds


def test_parse_cycles_one_dataset_per_cycle(tmp_path):
    p, m = _ids(tmp_path, 4)
    parts = IviumIDFParser().parse_cycles(p)
    assert [d.metadata["cycle_number"] for d in parts] == [1, 2, 3, 4]
    assert all(len(d.z_real) == m and d.metadata["data_type"] == "CV" for d in parts)


def test_different_scan_rates_warn(tmp_path):
    p, _ = _ids(tmp_path, 2, rates=["0.05", "0.1"])
    assert "different scan rates" in CVData.from_ivium(p).meta["cycle_warning"]


def test_single_header_file_unchanged(tmp_path):
    # one QR header -> parse_cycles == [parse()], CVData identical to the pre-patch55 path
    p = tmp_path / "one.idf"
    p.write_bytes(_obj("cycle 3", "CyclicVoltammetry", _cycle(0)).encode("latin-1"))
    ds = IviumIDFParser().parse(p)
    cv = CVData.from_ivium(p)
    assert np.array_equal(cv.potential_V, ds.z_real)
    assert np.array_equal(cv.current_mA, ds.z_imag)
    assert "cycle_bounds" not in cv.meta
    assert cv.select_cycle(-1).meta["n_cycles"] == 1


@pytest.mark.skipif(not DATA.exists(), reason="sample_eis.idf not present")
def test_eis_file_path_unchanged():
    (ds,) = IviumIDFParser().parse_cycles(DATA)
    ref = IviumIDFParser().parse(DATA)
    assert ds.metadata["data_type"] == "EIS"
    assert np.array_equal(ds.frequency, ref.frequency)


# ── real file (private; names and expected counts in reference_values.json) ──
def test_real_ids_all_cycles():
    folder, case = private_refs("ids_cycles")
    cv = CVData.from_ivium(private_file(folder, case["file"]))
    assert cv.meta["data_type"] == "CV"
    assert len(cv.meta["cycle_bounds"]) == case["n_cycles"]
    for k in range(case["n_cycles"]):
        c = cv.select_cycle(k)
        assert c.meta["cycle_used"] == k + 1
        assert len(c.potential_V) == case["points_per_cycle"]
