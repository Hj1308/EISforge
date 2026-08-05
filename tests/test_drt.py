"""Tests for the DRT (Distribution of Relaxation Times) analyzer.

Synthetic spectra use the module's own stored convention: z_imag is
-positive-Im(Z) (capacitive data stored positive), matching what the
Ivium parser and the EIS tab hold.
"""
import numpy as np
import pytest

from eisforge.analysis.drt_analyzer import DRTAnalyzer


def _rc_impedance(freq, r_inf, rc_pairs):
    """Z(ω) = r_inf + Σ R_k/(1 + jωτ_k) with rc_pairs = [(R, tau), ...].

    Returns (z_real, z_imag_stored) with z_imag_stored = -Im(Z) (positive).
    """
    w = 2.0 * np.pi * np.asarray(freq, dtype=float)
    z_re = np.full_like(w, float(r_inf))
    z_im_stored = np.zeros_like(w)
    for r_pol, tau in rc_pairs:
        denom = 1.0 + (w * tau) ** 2
        z_re += r_pol / denom
        z_im_stored += r_pol * w * tau / denom
    return z_re, z_im_stored


FREQ = np.logspace(-2, 4, 60)  # 0.01 Hz .. 10 kHz


def _peak_offsets(peaks_tau, expected_tau):
    """Log10 distance from each expected tau to nearest detected peak."""
    return np.min(np.abs(np.log10(peaks_tau / expected_tau)))


def test_single_rc_yields_one_peak_near_tau():
    tau = 0.01  # R=100, C=1e-4 -> tau = 1e-2 s
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, tau)])
    res = DRTAnalyzer().analyze(FREQ, zr, zi)
    assert len(res.peaks_tau) >= 1
    assert _peak_offsets(res.peaks_tau, tau) < 0.5  # within half a decade


def test_two_rc_separated_by_decades_yields_two_peaks():
    t1, t2 = 5e-4, 0.15  # ~2.5 decades apart
    zr, zi = _rc_impedance(
        FREQ, r_inf=10.0, rc_pairs=[(50.0, t1), (150.0, t2)]
    )
    res = DRTAnalyzer().analyze(FREQ, zr, zi)
    assert len(res.peaks_tau) >= 2
    for expected in (t1, t2):
        assert _peak_offsets(res.peaks_tau, expected) < 0.5


def test_r_inf_recovered_close_to_known_series_resistance():
    r_inf = 20.0
    zr, zi = _rc_impedance(FREQ, r_inf=r_inf, rc_pairs=[(100.0, 0.01)])
    res = DRTAnalyzer().analyze(FREQ, zr, zi)
    assert abs(res.r_inf - r_inf) < 5.0
    assert res.r_inf_at_bound is False


def test_validate_input_rejects_mismatched_lengths():
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    with pytest.raises(ValueError):
        DRTAnalyzer().analyze(FREQ[:-3], zr, zi)


def test_validate_input_rejects_nonpositive_frequencies():
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    bad_freq = FREQ.copy()
    bad_freq[0] = 0.0
    with pytest.raises(ValueError):
        DRTAnalyzer().analyze(bad_freq, zr, zi)


def test_validate_input_rejects_degenerate_frequency_window():
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    same_freq = np.full(len(FREQ), 100.0)
    with pytest.raises(ValueError):
        DRTAnalyzer().analyze(same_freq, zr, zi)


def test_narrow_window_triggers_span_warning():
    narrow = np.logspace(0.0, 0.5, 60)  # 0.5 decades -> span tier fires
    zr, zi = _rc_impedance(narrow, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    res = DRTAnalyzer().analyze(narrow, zr, zi)
    assert res.span_decades < 1.0
    assert res.data_ratio >= 1.0  # isolate span: count is fine
    assert res.data_warning is not None


def test_low_count_triggers_ratio_warning():
    low = np.logspace(-2, 2, 8)  # 8 points, 4 decades -> ratio tier fires
    zr, zi = _rc_impedance(low, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    res = DRTAnalyzer().analyze(low, zr, zi)
    assert res.data_ratio < 1.0
    assert res.span_decades >= 1.0  # isolate ratio: window is wide
    assert res.data_warning is not None


def test_well_conditioned_spectrum_has_no_warning():
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    res = DRTAnalyzer().analyze(FREQ, zr, zi)
    assert res.data_ratio >= 1.0
    assert res.span_decades >= 1.0
    assert res.data_warning is None


def test_regularization_order_accepted():
    zr, zi = _rc_impedance(FREQ, r_inf=20.0, rc_pairs=[(100.0, 0.01)])
    res = DRTAnalyzer(regularization_order=2).analyze(FREQ, zr, zi)
    assert len(res.tau) == res.n_tau
    assert np.isfinite(res.gamma).all()
