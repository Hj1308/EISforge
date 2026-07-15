"""Tests for eisforge.analysis.scan_rate_analyzer."""

import numpy as np
import pytest

from eisforge.analysis.scan_rate_analyzer import (
    analyze_scan_rates, _anodic_peak,
)


def _make_cv(rate_mV, ipa_target, peak_E=0.5):
    """Synthetic single-peak anodic sweep whose max current = ipa_target."""
    E = np.linspace(-0.2, 1.0, 240)
    # Gaussian-ish anodic peak centred at peak_E
    I = ipa_target * np.exp(-((E - peak_E) ** 2) / (2 * 0.05 ** 2))
    return E, I


class TestAnodicPeak:
    def test_global_max(self):
        E = np.linspace(0, 1, 100)
        I = np.sin(np.pi * E)  # max at E=0.5
        ipa, epa = _anodic_peak(E, I)
        assert ipa == pytest.approx(1.0, abs=1e-2)
        assert epa == pytest.approx(0.5, abs=0.02)

    def test_window_restricts_search(self):
        E = np.linspace(0, 1, 200)
        I = E.copy()  # monotonic, global max at E=1
        ipa, epa = _anodic_peak(E, I, window=(0.2, 0.4))
        assert epa <= 0.4 + 1e-9
        assert ipa == pytest.approx(0.4, abs=0.02)

    def test_empty_window_falls_back(self):
        E = np.linspace(0, 1, 50)
        I = E.copy()
        ipa, epa = _anodic_peak(E, I, window=(5.0, 6.0))  # outside range
        assert ipa == pytest.approx(1.0, abs=0.05)


class TestAnalyzeScanRates:
    def _diffusion_dataset(self):
        # Ipa ∝ sqrt(nu) -> b-value ≈ 0.5
        rates = [25, 50, 100, 150, 200, 300]
        data = {}
        for r in rates:
            ipa = 2.0 * np.sqrt(r / 1000.0)
            data[r] = _make_cv(r, ipa)
        return data

    def _adsorption_dataset(self):
        # Ipa ∝ nu -> b-value ≈ 1.0
        rates = [25, 50, 100, 150, 200, 300]
        data = {}
        for r in rates:
            ipa = 5.0 * (r / 1000.0)
            data[r] = _make_cv(r, ipa)
        return data

    def test_diffusion_b_value(self):
        res = analyze_scan_rates(self._diffusion_dataset())
        assert res.b_value == pytest.approx(0.5, abs=0.05)
        assert res.b_r2 > 0.999
        assert "diffusion" in res.mechanism_label()

    def test_adsorption_b_value(self):
        res = analyze_scan_rates(self._adsorption_dataset())
        assert res.b_value == pytest.approx(1.0, abs=0.05)
        assert "surface" in res.mechanism_label() or "adsorption" in res.mechanism_label()

    def test_randles_sevcik_linearity(self):
        res = analyze_scan_rates(self._diffusion_dataset())
        assert res.rs_r2 > 0.999  # pure sqrt(nu) -> perfectly linear vs sqrt(nu)

    def test_mixed_control_warning(self):
        rates = [50, 100, 150, 200, 250, 300]
        data = {r: _make_cv(r, 3.0 * (r / 1000.0) ** 0.7) for r in rates}
        res = analyze_scan_rates(data)
        assert 0.6 <= res.b_value <= 0.9
        assert any("Mixed control" in w for w in res.warnings)

    def test_too_few_points(self):
        data = {50: _make_cv(50, 1.0), 100: _make_cv(100, 1.4)}
        res = analyze_scan_rates(data)
        assert any("Fewer than 3" in w for w in res.warnings)
        assert not np.isfinite(res.b_value)

    def test_apparent_D_requires_params(self):
        res = analyze_scan_rates(self._diffusion_dataset(), compute_D=True)
        assert res.diffusion_coeff is None
        assert any("missing" in w.lower() for w in res.warnings)

    def test_apparent_D_computed(self):
        res = analyze_scan_rates(
            self._diffusion_dataset(), compute_D=True,
            n_electrons=1, area_cm2=0.07, conc_mol_cm3=1e-6,
        )
        assert res.diffusion_coeff is not None
        assert res.diffusion_coeff > 0
        assert any("apparent" in w.lower() for w in res.warnings)

    def test_peak_window_changes_result(self):
        # peak at 0.5; a wrong window catching only the tail gives different Ipa
        data = self._diffusion_dataset()
        analyze_scan_rates(data)
        res_win = analyze_scan_rates(data, peak_window=(0.4, 0.6))
        assert np.all(res_win.ipa_potential >= 0.4 - 1e-9)
        assert np.all(res_win.ipa_potential <= 0.6 + 1e-9)
