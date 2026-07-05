"""Tests for eisforge.analysis.ca_analyzer."""

import numpy as np
import pytest

from eisforge.analysis.ca_analyzer import analyze_ca, CAResult


def _decay(n=1000, t_max=1000.0, i0=1e-5, i_inf=4e-6, sign=-1.0):
    """Synthetic CA: exponential-ish decay to a plateau, Ivium negative sign."""
    t = np.linspace(0.2, t_max, n)
    i = i_inf + (i0 - i_inf) * np.exp(-t / 200.0)
    return t, sign * i  # negative = anodic (Ivium)


class TestAnalyzeCA:
    def test_basic_retention(self):
        t, i = _decay(i0=1e-5, i_inf=4e-6)
        r = analyze_ca(t, i)
        # final plateau ~4e-6, initial ~1e-5 -> retention ~40%
        assert 35 <= r.retention_pct <= 55
        assert r.duration_s == pytest.approx(999.8, abs=1)

    def test_sign_handling(self):
        # negative (anodic) input must give positive currents in analysis
        t, i = _decay(sign=-1.0)
        r = analyze_ca(t, i)
        assert np.all(r.current >= 0)
        assert r.i_initial > 0

    def test_per_area_conversion(self):
        t, i = _decay()
        r_raw = analyze_ca(t, i, per_area=False)
        r_area = analyze_ca(t, i, area_cm2=0.5, per_area=True)
        assert r_area.unit_label == "A/cm²"
        # current density = raw / area -> retention (a ratio) unchanged
        assert r_area.retention_pct == pytest.approx(r_raw.retention_pct, abs=0.01)
        assert r_area.i_initial == pytest.approx(r_raw.i_initial / 0.5, rel=1e-6)

    def test_retention_at_times(self):
        t, i = _decay(t_max=2000.0, n=2000)
        r = analyze_ca(t, i, retention_times=(600.0, 1800.0))
        assert 600.0 in r.retention_at
        assert 1800.0 in r.retention_at
        # monotonic decay -> later time has lower retention
        assert r.retention_at[1800.0] < r.retention_at[600.0]

    def test_retention_time_outside_range_skipped(self):
        t, i = _decay(t_max=500.0)
        r = analyze_ca(t, i, retention_times=(600.0,))
        assert 600.0 not in r.retention_at  # 600 > 500 s duration

    def test_initial_drop_positive_for_decay(self):
        t, i = _decay()
        r = analyze_ca(t, i, drop_window_s=60.0)
        assert r.initial_drop_pct > 0  # current drops in first 60 s

    def test_high_retention_finding(self):
        t, i = _decay(i0=1e-5, i_inf=9.6e-6)  # barely decays
        r = analyze_ca(t, i)
        assert r.retention_pct >= 90
        assert any("High current retention" in f for f in r.findings)

    def test_low_retention_finding(self):
        t, i = _decay(i0=1e-5, i_inf=3e-6)
        r = analyze_ca(t, i)
        assert r.retention_pct < 70
        assert any("Substantial current decay" in f for f in r.findings)

    def test_too_few_points(self):
        t = np.array([0.0, 1.0, 2.0])
        i = np.array([-1e-5, -9e-6, -8e-6])
        r = analyze_ca(t, i)
        assert any("Fewer than 10" in w for w in r.warnings)

    def test_unsorted_input_sorted(self):
        t, i = _decay(n=500)
        perm = np.random.default_rng(0).permutation(len(t))
        r = analyze_ca(t[perm], i[perm])
        assert np.all(np.diff(r.time) >= 0)  # sorted internally

    def test_markdown_renders(self):
        t, i = _decay()
        r = analyze_ca(t, i)
        md = r.as_markdown()
        assert "Current retention" in md and "Duration" in md
