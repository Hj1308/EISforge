import logging

import numpy as np

from eisforge.core.validators import KramersKronigValidator
from eisforge.parsers.base_parser import EISDataset


def _randles_dataset(n=120, Rs=10.0, Rct=100.0, Cdl=2e-5):
    freq = np.logspace(4, -1, n)
    w = 2 * np.pi * freq
    Z = Rs + Rct / (1 + 1j * w * Rct * Cdl)
    return EISDataset(frequency=freq, z_real=Z.real, z_imag=-Z.imag, metadata={})


class TestKramersKronigValidator:
    def test_clean_randles_passes_at_default_threshold(self):
        res = KramersKronigValidator().validate(_randles_dataset())
        assert res.passed is True
        assert len(res.residuals_real) == 120
        assert len(res.residuals_imag) == 120
        assert 0 < res.max_residual < 0.005
        assert res.n_rc_elements > 0
        assert res.warning_message is None
        assert res.method in ("linKK", "voigt")

    def test_summary_reports_passed(self):
        res = KramersKronigValidator().validate(_randles_dataset())
        assert "PASSED" in res.summary()

    def test_few_points_is_non_blocking_failure(self):
        res = KramersKronigValidator().validate(_randles_dataset(n=5))
        assert res.passed is False
        assert np.isinf(res.max_residual)
        assert res.warning_message is not None
        assert "Insufficient data points" in res.warning_message
        assert res.method == "not_run"

    def test_threshold_is_honored(self):
        ds = _randles_dataset()
        default = KramersKronigValidator()
        strict = KramersKronigValidator(residual_threshold=1e-9)
        assert default.validate(ds).passed is True
        assert strict.validate(ds).passed is False

    def test_ten_points_proceeds_past_guard(self):
        res = KramersKronigValidator().validate(_randles_dataset(n=10))
        assert res.warning_message is None or "Insufficient data points" not in res.warning_message
        assert len(res.residuals_real) == 10
        assert res.method in ("linKK", "voigt")

    def test_linkk_failure_logged_and_handled(self, monkeypatch, caplog):
        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic linKK failure")

        monkeypatch.setattr("impedance.validation.linKK", _boom)
        with caplog.at_level(logging.WARNING, logger="eisforge.core.validators"):
            res = KramersKronigValidator().validate(_randles_dataset())
        assert res.passed is True
        assert res.method == "voigt"
        assert "linKK failed" in caplog.text
        assert "synthetic linKK failure" in caplog.text
        assert res.mu == 0.85
        assert "μ=" not in res.summary()

    def test_linkk_call_uses_valid_signature(self, monkeypatch):
        """linKK must be called with a signature the installed impedance
        1.7.1 actually accepts: no mu kwarg, and add_cap=False (the series
        capacitance would depress residuals and let bad spectra pass)."""
        calls = {}

        def _record(freq, Z, **kwargs):
            calls["kwargs"] = kwargs
            raise RuntimeError("stopping after recording the call")

        monkeypatch.setattr("impedance.validation.linKK", _record)
        KramersKronigValidator().validate(_randles_dataset())
        kwargs = calls["kwargs"]
        assert "mu" not in kwargs
        assert kwargs.get("add_cap") is False
        assert kwargs.get("fit_type") == "real"
