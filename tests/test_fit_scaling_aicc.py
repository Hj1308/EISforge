"""patch49 — CNLS fit reaches the minimum despite badly scaled parameters (x_scale="jac"),
and AICc ranks with the fit objective (|Z|-weighted RSS, k = p + 1).

Real file: tests/data/sample_eis.idf. Before patch49 the 2-RC fit stopped after 28
evaluations (relative xtol) at 25x the reachable SSE and the nested 1-RC model ranked best.
"""

from pathlib import Path

import numpy as np

from eisforge.catalogs.suggestion_engine import suggest_circuits
from eisforge.parsers.ivium_parser import IviumIDFParser

DATA = Path(__file__).parent / "data" / "sample_eis.idf"
ONE_RC, TWO_RC = "R0-p(R1,CPE1)", "R0-p(R1,CPE1)-p(R2,CPE2)"


def _suggestions():
    ds = IviumIDFParser().parse(DATA)
    out = {s.model.notation: s for s in suggest_circuits(ds.frequency, ds.z_real, ds.z_imag)}
    return ds, out


def _wrss(ds, s):
    z = ds.z_complex
    return float(np.sum(np.abs((z - s.fit_result.z_fit) / np.abs(z)) ** 2))


def test_nested_model_fits_at_least_as_well():
    ds, out = _suggestions()
    assert out[ONE_RC].converged and out[TWO_RC].converged
    # 1-RC is a special case of 2-RC, so at the minimum 2-RC cannot be worse
    assert _wrss(ds, out[TWO_RC]) <= _wrss(ds, out[ONE_RC]) * 1.0001


def test_two_rc_ranks_above_one_rc():
    _, out = _suggestions()
    assert out[TWO_RC].aicc < out[ONE_RC].aicc
    assert out[ONE_RC].delta_aicc > 10
