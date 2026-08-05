import numpy as np
import pytest

from eisforge.core.validators import KramersKronigValidator
from eisforge.parsers.base_parser import EISDataset


def _randles_dataset(n=120, Rs=10.0, Rct=100.0, Cdl=2e-5):
    freq = np.logspace(4, -1, n)
    w = 2 * np.pi * freq
    Z = Rs + Rct / (1 + 1j * w * Rct * Cdl)
    return EISDataset(frequency=freq, z_real=Z.real, z_imag=-Z.imag, metadata={})


@pytest.mark.filterwarnings("ignore:linKK failed")
class TestKramersKronigValidator:
    def test_clean_randles_passes_at_default_threshold(self):
        res = KramersKronigValidator().validate(_randles_dataset())
        assert res.passed is True
        assert len(res.residuals_real) == 120
        assert len(res.residuals_imag) == 120
        assert 0 < res.max_residual < 0.005
        assert res.n_rc_elements > 0
        assert res.warning_message is None

    def test_summary_reports_passed(self):
        res = KramersKronigValidator().validate(_randles_dataset())
        assert "PASSED" in res.summary()

    def test_few_points_is_non_blocking_failure(self):
        res = KramersKronigValidator().validate(_randles_dataset(n=5))
        assert res.passed is False
        assert np.isinf(res.max_residual)
        assert res.warning_message is not None
        assert "Insufficient data points" in res.warning_message

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
        assert res.n_rc_elements == 5
