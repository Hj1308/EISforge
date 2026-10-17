"""patch45 — CVData.select_cycle pins the cycle rule of the pre-patch45 app loader.

The legacy rule is kept unchanged so that no result moves. It is known to be off by
one for multi-cycle scans (ROADMAP B12): these tests document it and must be updated
when B12 is fixed.
"""

import numpy as np

from eisforge.data.cv_data import CVData


def _three_cycles(start_up: bool) -> CVData:
    half = np.round(np.arange(-0.8, 0.2001, 0.01), 5)
    if not start_up:
        half = half[::-1]
    cyc = np.concatenate([half, half[-2::-1]])
    e = np.concatenate([cyc] + [cyc[1:]] * 2)
    return CVData(potential_V=e, current_mA=np.arange(len(e), dtype=float))


def _pick(cv, idx):
    c = cv.select_cycle(idx)
    return c.meta["n_cycles"], c.meta["cycle_used"], len(c.potential_V)


def test_upward_start_legacy_rule():
    cv = _three_cycles(start_up=True)
    assert _pick(cv, 0) == (2, 1, 201)
    assert _pick(cv, -1) == (2, 1, 201)       # B12: 3 cycles counted as 2
    assert _pick(cv, 99) == (2, 2, 200)


def test_downward_start_legacy_rule():
    cv = _three_cycles(start_up=False)
    assert _pick(cv, 0) == (3, 1, 101)        # B12: first "cycle" is a half sweep
    assert _pick(cv, -1) == (3, 2, 200)


def test_single_sweep_returns_everything():
    lsv = CVData(potential_V=np.linspace(0, 1, 50), current_mA=np.zeros(50), label="x")
    c = lsv.select_cycle(-1)
    assert (c.meta["n_cycles"], c.meta["cycle_used"], len(c.potential_V)) == (1, 1, 50)
    assert c.label == "x" and np.array_equal(c.potential_V, lsv.potential_V)
