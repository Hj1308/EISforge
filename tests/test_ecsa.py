import pytest
import numpy as np
from eisforge.analysis.ecsa_calculator import ECSACalculator


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def hupd_cv():
    """Synthetic full CV with H-UPD peak only on the cathodic scan."""
    v_fwd = np.linspace(0.0, 0.8, 500)
    v_bwd = np.linspace(0.8, 0.0, 500)
    i_fwd = np.zeros(500)
    i_bwd = -5e-4 * np.exp(-((v_bwd - 0.2) ** 2) / (2 * 0.015 ** 2))   # A
    return (np.concatenate([v_fwd, v_bwd]),
            np.concatenate([i_fwd, i_bwd]))


@pytest.fixture
def cdl_cv_dataset():
    """Synthetic CVs for 6 scan rates with known Cdl = 0.035 mF/cm²."""
    scan_rates_Vs = [0.010, 0.020, 0.040, 0.060, 0.080, 0.100]
    cdl_F = 0.035e-3   # F/cm²
    pots, curs = [], []
    for sr in scan_rates_Vs:
        pot = np.concatenate([np.linspace(0.10, 0.40, 200),
                              np.linspace(0.40, 0.10, 200)])
        cur = np.concatenate([np.full(200,  cdl_F * sr),
                              np.full(200, -cdl_F * sr)])
        pots.append(pot)
        curs.append(cur)
    return pots, curs, scan_rates_Vs


# ─── Method A ────────────────────────────────────────────────────────────────

class TestMethodA:

    def test_basic_results(self, hupd_cv):
        pot, cur = hupd_cv
        res = ECSACalculator.method_a_hupd(pot, cur, scan_rate=0.05,
                                           loading_mg=0.1, v_range=(0.05, 0.40))
        assert res["method"] == "H-UPD"
        assert res["ecsa_cm2"] > 0
        assert res["charge_uC"] > 0

    def test_specific_ecsa_formula(self, hupd_cv):
        pot, cur = hupd_cv
        res = ECSACalculator.method_a_hupd(pot, cur, 0.05, loading_mg=0.1)
        assert pytest.approx(res["specific_ecsa_cm2_mg"], rel=1e-9) == res["ecsa_cm2"] / 0.1

    def test_cathodic_scan_only(self, hupd_cv):
        """Adding noise to the forward scan must not change the result."""
        pot, cur = hupd_cv
        cur_noisy = cur.copy()
        cur_noisy[:500] += np.random.normal(0, 1e-6, 500)   # noise on forward scan
        res_clean = ECSACalculator.method_a_hupd(pot, cur, 0.05, 0.1)
        res_noisy = ECSACalculator.method_a_hupd(pot, cur_noisy, 0.05, 0.1)
        assert pytest.approx(res_clean["ecsa_cm2"], rel=0.01) == res_noisy["ecsa_cm2"]

    def test_q_ref_default_pt(self, hupd_cv):
        pot, cur = hupd_cv
        res = ECSACalculator.method_a_hupd(pot, cur, 0.05, 0.1)
        assert res["q_ref_used"] == ECSACalculator.Q_H_PT   # 210.0 µC/cm²

    def test_q_ref_subclass_safe(self, hupd_cv):
        """Subclass override of Q_H_PT must be respected."""
        class PdCalc(ECSACalculator):
            Q_H_PT = 212.0
        pot, cur = hupd_cv
        res_pd = PdCalc.method_a_hupd(pot, cur, 0.05, 0.1)
        res_pt = ECSACalculator.method_a_hupd(pot, cur, 0.05, 0.1)
        assert res_pd["q_ref_used"] == 212.0
        assert res_pd["ecsa_cm2"] < res_pt["ecsa_cm2"]      # higher q_ref → smaller ECSA

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="data points"):
            ECSACalculator.method_a_hupd(np.array([0.1, 0.2]),
                                         np.array([0.0, 0.0]), 0.05, 0.1)

    def test_nan_input_raises(self, hupd_cv):
        pot, cur = hupd_cv
        cur_bad = cur.copy(); cur_bad[5] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            ECSACalculator.method_a_hupd(pot, cur_bad, 0.05, 0.1)


# ─── Method C ────────────────────────────────────────────────────────────────

class TestMethodC:

    def test_cdl_accuracy(self, cdl_cv_dataset):
        pots, curs, srs = cdl_cv_dataset
        res = ECSACalculator.method_c_cdl(pots, curs, srs, v_range=(0.10, 0.40))
        assert pytest.approx(res["cdl_mF_cm2"], rel=1e-2) == 0.035   # ✅ FIXED

    def test_r_squared(self, cdl_cv_dataset):
        pots, curs, srs = cdl_cv_dataset
        res = ECSACalculator.method_c_cdl(pots, curs, srs, v_range=(0.10, 0.40))
        assert res["r_squared"] > 0.99

    def test_intercept_returned(self, cdl_cv_dataset):
        """fit_intercept must be present (needed for UI trendline)."""
        pots, curs, srs = cdl_cv_dataset
        res = ECSACalculator.method_c_cdl(pots, curs, srs, v_range=(0.10, 0.40))
        assert "fit_intercept" in res

    def test_mismatch_raises(self, cdl_cv_dataset):
        pots, curs, srs = cdl_cv_dataset
        with pytest.raises(ValueError, match="same length"):
            ECSACalculator.method_c_cdl(pots[:2], curs, srs, v_range=(0.10, 0.40))

    def test_minimum_scan_rates_warning(self, cdl_cv_dataset):
        pots, curs, srs = cdl_cv_dataset
        with pytest.warns(UserWarning, match="3 scan rates"):
            ECSACalculator.method_c_cdl(pots[:2], curs[:2], srs[:2],
                                        v_range=(0.10, 0.40))
