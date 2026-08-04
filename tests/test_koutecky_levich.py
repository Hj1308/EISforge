"""Tests for KLAnalyzer bulk-concentration override (C_mol_cm3)."""

import pytest

from eisforge.analysis.koutecky_levich import KLAnalyzer


class TestCMolCm3Override:
    def test_default_C_derived_from_concentration(self):
        kla = KLAnalyzer(concentration_M=1.0)
        assert kla.C == pytest.approx(1.0e-3)

    def test_custom_C_used_exactly(self):
        kla = KLAnalyzer(concentration_M=1.0, C_mol_cm3=1.2e-6)
        assert kla.C == pytest.approx(1.2e-6)

    def test_custom_C_independent_of_concentration_M(self):
        kla = KLAnalyzer(concentration_M=0.1, C_mol_cm3=1.2e-6)
        assert kla.C == pytest.approx(1.2e-6)

    def test_C_mol_cm3_is_keyword_only(self):
        with pytest.raises(TypeError):
            KLAnalyzer(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
