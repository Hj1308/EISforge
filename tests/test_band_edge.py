"""tests/test_band_edge.py — 8 unit tests for BandEdgeCalculator"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eisforge.analysis.band_edge_calculator import (
    BandEdgeCalculator, E_NHE_OFFSET, NERNST_SLOPE
)


class TestBandEdgeBasic:

    def test_tio2_ecb_vacuum(self):
        calc = BandEdgeCalculator("TiO2", pH=0.0)
        r = calc.calculate()
        expected = 5.81 - 4.5 - 0.5 * 3.20
        assert abs(r.Ecb_vacuum - expected) < 1e-6

    def test_gcn_evb_vacuum(self):
        calc = BandEdgeCalculator("g-C3N4", pH=0.0)
        r = calc.calculate()
        assert abs(r.Evb_vacuum - (r.Ecb_vacuum + r.Eg)) < 1e-6

    def test_nhe_conversion(self):
        calc = BandEdgeCalculator("TiO2", pH=0.0)
        r = calc.calculate()
        assert abs(r.Ecb_NHE - (r.Ecb_vacuum - E_NHE_OFFSET)) < 1e-6


class TestRHEConversion:

    def test_rhe_pH7(self):
        calc = BandEdgeCalculator("TiO2", pH=7.0)
        r = calc.calculate()
        expected_rhe = r.Ecb_NHE - NERNST_SLOPE * 7.0
        assert abs(r.Ecb_RHE - expected_rhe) < 1e-6

    def test_rhe_pH14(self):
        c0  = BandEdgeCalculator("ZnO", pH=0.0).calculate()
        c14 = BandEdgeCalculator("ZnO", pH=14.0).calculate()
        assert abs((c0.Ecb_RHE - c14.Ecb_RHE) - NERNST_SLOPE * 14.0) < 1e-4


class TestMottSchottky:

    def _synthetic_ms(self, Vfb_true=0.40, Nd_true=1e17, n_type=True):
        """
        Build a perfect linear 1/C^2 vs V dataset.
        Mott-Schottky:  1/C^2 = (2 / (e*eps0*eps*A^2*Nd)) * (V - Vfb)
        intercept_true = -slope * Vfb_true  =>  Vfb = -intercept/slope  ✓
        """
        epsilon_r = 10.0
        area_m2   = 1e-4          # 1 cm^2 in m^2
        V         = np.linspace(0.0, 1.0, 50)
        sign      = 1 if n_type else -1

        # slope of 1/C^2 vs V  (m^4 F^-2 V^-1)
        slope = sign * 2.0 / (
            1.602e-19 * 8.854e-12 * epsilon_r * (area_m2 ** 2) * Nd_true * 1e6
        )
        # intercept: at V=0, 1/C^2 = -slope*Vfb_true
        intercept = -slope * Vfb_true

        C_inv2 = slope * V + intercept          # perfect linear: no extra offset
        # Guard: all values must be positive before sqrt
        C_inv2 = np.abs(C_inv2) + 1e-3 * np.max(np.abs(C_inv2))
        C      = 1.0 / np.sqrt(C_inv2)
        return V, C

    def test_vfb_recovery(self):
        calc = BandEdgeCalculator("TiO2", electrode_area=1.0)
        V, C = self._synthetic_ms(Vfb_true=0.40)
        ms   = calc.mott_schottky(V, C)
        assert abs(ms.Vfb - 0.40) < 0.05

    def test_ntype_detection(self):
        calc = BandEdgeCalculator("TiO2", electrode_area=1.0)
        V, C = self._synthetic_ms(n_type=True)
        ms   = calc.mott_schottky(V, C)
        assert ms.sc_type == "n-type"


class TestCustomMaterial:

    def test_custom_bcn(self):
        calc = BandEdgeCalculator("custom", X=4.85, Eg=2.15, pH=7.0)
        r    = calc.calculate()
        expected_ecb = 4.85 - 4.5 - 0.5 * 2.15
        assert abs(r.Ecb_vacuum - expected_ecb) < 1e-6
        assert r.Eg == 2.15
