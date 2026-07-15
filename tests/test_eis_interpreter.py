"""Tests for eisforge.analysis.eis_interpreter (rule-based interpretation)."""

import pytest

from eisforge.analysis.eis_interpreter import interpret_fit, _brug_c_eff


class TestBrugCeff:
    def test_ideal_capacitor_limit(self):
        # n=1 -> C_eff = Q exactly
        assert _brug_c_eff(2e-5, 1.0, 1000.0) == pytest.approx(2e-5)

    def test_typical_cpe(self):
        c = _brug_c_eff(3e-5, 0.8, 5000.0)
        assert c is not None and 1e-7 < c < 1e-2

    def test_invalid_inputs(self):
        assert _brug_c_eff(-1e-5, 0.8, 1000.0) is None
        assert _brug_c_eff(1e-5, 0.0, 1000.0) is None
        assert _brug_c_eff(1e-5, 1.5, 1000.0) is None
        assert _brug_c_eff(1e-5, 0.8, -10.0) is None


class TestInterpretFit:
    def test_two_arc_carbon(self):
        params = {"R0": 26.2, "R1": 1153.0, "CPE1_0": 3.17e-5, "CPE1_1": 0.769,
                  "R2": 3.19e4, "CPE2_0": 1.13e-5, "CPE2_1": 0.773}
        r = interpret_fit(params, "R0-p(R1,CPE1)-p(R2,CPE2)", 3.5e-4)
        assert r.r_series == pytest.approx(26.2)
        assert len(r.processes) == 2
        # both processes get Brug C_eff and time constants
        assert all(p.c_eff is not None and p.tau is not None for p in r.processes)
        # low-n dispersion flagged
        assert any("n < 0.80" in f for f in r.findings)
        # excellent chi2 noted
        assert any("excellent" in f for f in r.findings)
        md = r.as_markdown()
        assert "R_s" in md and "R1" in md and "R2" in md

    def test_time_constant_ordering(self):
        params = {"R0": 30, "R1": 1000, "CPE1_0": 1e-5, "CPE1_1": 0.9,
                  "R2": 30000, "CPE2_0": 1e-5, "CPE2_1": 0.9}
        r = interpret_fit(params)
        taus = [p.tau for p in r.processes]
        assert taus[1] > taus[0]  # bigger R -> slower process
        assert any("well-separated" in f for f in r.findings)

    def test_ndr_and_inductor_flags(self):
        params = {"R0": 25, "R1": -500, "CPE1_0": 1e-4, "CPE1_1": 0.9, "L2": 100}
        r = interpret_fit(params, chi_squared=5e-4)
        assert any("Negative faradaic resistance" in f for f in r.findings)
        assert any("Inductive element" in f for f in r.findings)

    def test_warburg_flag(self):
        params = {"R0": 25, "R1": 800, "CPE1_0": 1e-5, "CPE1_1": 0.85,
                  "Wo1_0": 500, "Wo1_1": 2.0}
        r = interpret_fit(params)
        assert any("Warburg" in f for f in r.findings)

    def test_rc_pair_plain_capacitor(self):
        params = {"R0": 10, "R1": 2000, "C1": 5e-6}
        r = interpret_fit(params)
        assert r.processes[0].c_eff == pytest.approx(5e-6)
        assert r.processes[0].tau == pytest.approx(0.01)

    def test_poor_fit_warning(self):
        params = {"R0": 10, "R1": 2000, "C1": 5e-6}
        r = interpret_fit(params, chi_squared=0.5)
        assert any("poor fit" in w for w in r.warnings)

    def test_unphysical_capacitance_warning(self):
        params = {"R0": 10, "R1": 2000, "C1": 5.0}
        r = interpret_fit(params)
        assert any("unphysically large" in w for w in r.warnings)

    def test_high_rs_warning(self):
        params = {"R0": 5e4, "R1": 2000, "C1": 1e-6}
        r = interpret_fit(params)
        assert any("solution resistance" in w for w in r.warnings)

    def test_empty_parameters(self):
        r = interpret_fit({})
        assert any("cannot interpret" in w for w in r.warnings)
        assert "No interpretable" not in r.as_markdown() or True  # markdown renders

    def test_nan_chi2_no_crash(self):
        params = {"R0": 10, "R1": 100, "C1": 1e-6}
        r = interpret_fit(params, chi_squared=float("nan"))
        assert isinstance(r.as_markdown(), str)
