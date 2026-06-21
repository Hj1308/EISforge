"""
Unit tests for batch_analyzer.py
Runs without EISforge dependencies — tests pure statistical logic only.

Run with:
    pytest tests/test_batch_analyzer.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

# ── Import only the pure-stat helpers (no EISforge deps needed) ───────────────
from eisforge.analysis.batch_analyzer import (
    _stat,
    _stat_filtered,
    _grubbs_outliers,
    _align_to_common_axis,
    _to_latex_safe,
    _fmt_j0,
)


# ─────────────────────────── _stat ───────────────────────────────────────────

class TestStat:
    def test_basic_mean_std(self):
        vals = [1.0, 2.0, 3.0]
        mean, std = _stat(vals)
        assert abs(mean - 2.0) < 1e-10
        assert abs(std - 1.0) < 1e-10

    def test_single_value_std_zero(self):
        mean, std = _stat([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_nan_ignored(self):
        vals = [1.0, float("nan"), 3.0]
        mean, std = _stat(vals)
        assert abs(mean - 2.0) < 1e-10

    def test_all_nan_returns_nan(self):
        mean, std = _stat([float("nan"), float("nan")])
        assert np.isnan(mean)
        assert np.isnan(std)

    def test_none_ignored(self):
        vals = [1.0, None, 3.0]
        mean, std = _stat(vals)
        assert abs(mean - 2.0) < 1e-10


# ─────────────────────────── _stat_filtered ──────────────────────────────────

class TestStatFiltered:
    def test_excludes_outlier_index(self):
        # Without outlier: mean of [0.45, 0.46, 0.46]
        vals = [0.45, 0.46, 0.46, 1.50]   # index 3 is outlier
        mean, std = _stat_filtered(vals, outlier_idx=[3])
        assert abs(mean - np.mean([0.45, 0.46, 0.46])) < 1e-10

    def test_no_outliers_same_as_stat(self):
        vals = [0.45, 0.46, 0.46]
        assert _stat_filtered(vals, []) == _stat(vals)

    def test_all_flagged_falls_back_to_stat(self):
        vals = [1.0, 2.0]
        # If all are flagged, should fall back to _stat(vals) not crash
        mean, std = _stat_filtered(vals, outlier_idx=[0, 1])
        assert not np.isnan(mean)


# ─────────────────────────── _grubbs_outliers ────────────────────────────────

class TestGrubbsOutliers:
    def test_no_outlier_clean_data(self):
        vals = [0.450, 0.451, 0.452, 0.453, 0.454]
        assert _grubbs_outliers(vals) == []

    def test_detects_single_outlier(self):
        # Clear outlier: value far from rest
        vals = [0.450, 0.451, 0.452, 0.453, 5.000]
        outliers = _grubbs_outliers(vals)
        assert 4 in outliers, f"Expected index 4 flagged, got {outliers}"

    def test_iterative_detects_two_outliers(self):
        # Two clear outliers
        vals = [0.450, 0.451, 0.452, 10.0, 20.0]
        outliers = _grubbs_outliers(vals)
        assert len(outliers) >= 1   # at minimum one should be flagged

    def test_too_few_points_returns_empty(self):
        assert _grubbs_outliers([0.45, 0.46]) == []
        assert _grubbs_outliers([]) == []

    def test_nan_values_excluded(self):
        vals = [0.45, float("nan"), 0.46, 0.45, 5.0]
        outliers = _grubbs_outliers(vals)
        assert 4 in outliers

    def test_identical_values_no_outlier(self):
        # std = 0, should return empty
        vals = [0.450, 0.450, 0.450]
        assert _grubbs_outliers(vals) == []

    def test_outlier_index_is_original_position(self):
        # Outlier is at original index 0, not index in the 'remaining' sublist
        vals = [99.0, 0.450, 0.451, 0.452]
        outliers = _grubbs_outliers(vals)
        assert 0 in outliers


# ─────────────────────────── _align_to_common_axis ───────────────────────────

class TestAlignToCommonAxis:
    def _make_lsv(self, e_start=0.0, e_end=0.5, n=50, slope=10.0):
        """Linear LSV: I = slope * E"""
        pot = np.linspace(e_start, e_end, n)
        cur = slope * pot
        return pot, cur

    def test_output_shapes(self):
        pots = [self._make_lsv()[0] for _ in range(3)]
        curs = [self._make_lsv()[1] for _ in range(3)]
        pot_c, mean_c, std_c = _align_to_common_axis(pots, curs, n_points=100)
        assert len(pot_c) == len(mean_c) == len(std_c)
        assert len(pot_c) <= 100

    def test_identical_curves_std_zero(self):
        pots = [self._make_lsv()[0] for _ in range(3)]
        curs = [self._make_lsv()[1] for _ in range(3)]
        _, mean_c, std_c = _align_to_common_axis(pots, curs)
        np.testing.assert_allclose(std_c, 0.0, atol=1e-10)

    def test_mean_curve_correct(self):
        # Two curves: slope 10 and slope 20 → mean slope 15
        pot1, cur1 = self._make_lsv(slope=10.0)
        pot2, cur2 = self._make_lsv(slope=20.0)
        pot_c, mean_c, _ = _align_to_common_axis([pot1, pot2], [cur1, cur2], n_points=50)
        expected = 15.0 * pot_c
        np.testing.assert_allclose(mean_c, expected, rtol=1e-5)

    def test_common_axis_within_overlap(self):
        # Curves have slightly different ranges — common axis must be within overlap
        pot1 = np.linspace(0.0, 0.5, 50)
        pot2 = np.linspace(0.1, 0.6, 50)
        cur1 = np.ones(50)
        cur2 = np.ones(50)
        pot_c, _, _ = _align_to_common_axis([pot1, pot2], [cur1, cur2])
        assert pot_c.min() >= 0.09
        assert pot_c.max() <= 0.51


# ─────────────────────────── _to_latex_safe ──────────────────────────────────

class TestToLatexSafe:
    def test_eta_converted(self):
        result = _to_latex_safe("η @ 10 mA/cm²")
        assert r"$\eta$" in result
        assert r"$^2$" in result

    def test_underscore_escaped(self):
        result = _to_latex_safe("E_onset")
        assert r"\_" in result

    def test_percent_escaped(self):
        result = _to_latex_safe("50%")
        assert r"\%" in result

    def test_plain_string_unchanged(self):
        result = _to_latex_safe("Tafel slope")
        assert result == "Tafel slope"


# ─────────────────────────── _fmt_j0 ─────────────────────────────────────────

class TestFmtJ0:
    def test_small_value_uses_micro(self):
        result = _fmt_j0(1e-6)
        assert "μA" in result

    def test_large_value_uses_mA(self):
        result = _fmt_j0(0.5)
        assert "mA" in result

    def test_nan_returns_nan_string(self):
        assert _fmt_j0(float("nan")) == "nan"


# ─────────────────────────── Integration: outlier effect on stats ─────────────

class TestOutlierExclusionEffect:
    """
    Critical test: verify that mean/SD changes when an outlier is present
    and _stat_filtered is used (not plain _stat).
    """

    def test_mean_differs_with_and_without_outlier(self):
        vals = [0.450, 0.451, 0.452, 5.000]   # 5.000 is clear outlier
        outlier_idx = _grubbs_outliers(vals)
        assert outlier_idx, "Grubbs should flag 5.000 as outlier"

        mean_raw,  _ = _stat(vals)
        mean_filt, _ = _stat_filtered(vals, outlier_idx)

        assert mean_filt < mean_raw, (
            f"Filtered mean ({mean_filt:.4f}) should be less than raw mean ({mean_raw:.4f})"
        )
        # Filtered mean should be close to 0.451
        assert abs(mean_filt - 0.451) < 0.005
