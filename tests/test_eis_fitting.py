"""
Unit tests for EIS impedance calculations — synthetic Randles circuit.

These tests use analytically generated data (no real instrument files needed)
to verify that the impedance math and conventions are correct.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic circuit models
# ---------------------------------------------------------------------------

def randles_impedance(
    freq: np.ndarray,
    Rs: float,
    Rct: float,
    Cdl: float,
) -> np.ndarray:
    """
    Randles circuit: Z = Rs + Rct / (1 + j*omega*Rct*Cdl)

    Parameters
    ----------
    freq : array of frequencies in Hz
    Rs   : solution resistance (Ohm)
    Rct  : charge-transfer resistance (Ohm)
    Cdl  : double-layer capacitance (F)
    """
    omega = 2 * np.pi * np.asarray(freq, dtype=float)
    Z_rc = Rct / (1.0 + 1j * omega * Rct * Cdl)
    return Rs + Z_rc


def rc_series_impedance(freq: np.ndarray, R: float, C: float) -> np.ndarray:
    """Simple series R-C: Z = R + 1/(j*omega*C)"""
    omega = 2 * np.pi * np.asarray(freq, dtype=float)
    return R + 1.0 / (1j * omega * C)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def randles_params():
    """Standard Randles circuit parameters."""
    return {"Rs": 10.0, "Rct": 150.0, "Cdl": 50e-6}  # Ohm, Ohm, F


@pytest.fixture
def freq_array():
    """Logarithmically spaced frequency array (100 kHz → 10 mHz)."""
    return np.logspace(5, -2, 60)


@pytest.fixture
def randles_data(freq_array, randles_params):
    """Pre-computed Randles impedance spectrum."""
    Z = randles_impedance(freq_array, **randles_params)
    return freq_array, Z, randles_params


# ---------------------------------------------------------------------------
# Tests: high-frequency limit
# ---------------------------------------------------------------------------

class TestHighFrequencyLimit:
    """At f→∞, Z_real → Rs and Z_imag → 0."""

    def test_zreal_approaches_rs(self, randles_data):
        freq, Z, p = randles_data
        z_hf = Z[0]  # highest frequency
        assert abs(z_hf.real - p["Rs"]) < 0.5, (
            f"Z_real at HF = {z_hf.real:.3f} Ohm, expected ≈ {p['Rs']} Ohm"
        )

    def test_zimag_approaches_zero(self, randles_data):
        freq, Z, p = randles_data
        z_hf = Z[0]
        assert abs(z_hf.imag) < 1.0, (
            f"Z_imag at HF = {z_hf.imag:.3f} Ohm, expected ≈ 0"
        )


# ---------------------------------------------------------------------------
# Tests: low-frequency limit
# ---------------------------------------------------------------------------

class TestLowFrequencyLimit:
    """At f→0, Z_real → Rs + Rct."""

    def test_zreal_approaches_rs_plus_rct(self, randles_data):
        freq, Z, p = randles_data
        z_lf = Z[-1]  # lowest frequency
        expected = p["Rs"] + p["Rct"]
        assert abs(z_lf.real - expected) < 5.0, (
            f"Z_real at LF = {z_lf.real:.3f} Ohm, expected ≈ {expected} Ohm"
        )

    def test_zimag_approaches_zero_at_dc(self, randles_data):
        freq, Z, _ = randles_data
        z_lf = Z[-1]
        assert abs(z_lf.imag) < 10.0, (
            f"Z_imag at LF = {z_lf.imag:.3f} Ohm, expected ≈ 0"
        )


# ---------------------------------------------------------------------------
# Tests: Nyquist sign convention
# ---------------------------------------------------------------------------

class TestNyquistConvention:
    """
    Standard electrochemistry Nyquist plot uses -Im(Z) on the y-axis,
    so capacitive arcs appear in the positive quadrant.
    """

    def test_negative_imaginary_part_for_capacitive_circuit(self, randles_data):
        """Im(Z) should be negative (capacitive) for Randles circuit."""
        freq, Z, _ = randles_data
        assert np.all(Z.imag <= 0), (
            "All Im(Z) values should be ≤ 0 for a purely capacitive Randles circuit."
        )

    def test_nyquist_y_axis_positive(self, randles_data):
        """The Nyquist y-axis (-Im(Z)) should be positive for capacitive arcs."""
        freq, Z, _ = randles_data
        assert np.all(-Z.imag >= 0), (
            "-Im(Z) must be ≥ 0 (positive Nyquist quadrant) for a Randles circuit."
        )


# ---------------------------------------------------------------------------
# Tests: data shape and types
# ---------------------------------------------------------------------------

class TestDataShape:

    def test_output_length_matches_input(self, freq_array, randles_params):
        Z = randles_impedance(freq_array, **randles_params)
        assert len(Z) == len(freq_array)

    def test_output_dtype_complex(self, freq_array, randles_params):
        Z = randles_impedance(freq_array, **randles_params)
        assert np.iscomplexobj(Z), "Impedance output must be complex."

    def test_magnitude_always_positive(self, randles_data):
        freq, Z, _ = randles_data
        assert np.all(np.abs(Z) > 0)


# ---------------------------------------------------------------------------
# Tests: semicircle diameter
# ---------------------------------------------------------------------------

class TestSemicircleDiameter:
    """
    For a Randles circuit, the Nyquist semicircle diameter equals Rct,
    and the semicircle is centred at (Rs + Rct/2, 0).
    """

    def test_semicircle_diameter_equals_rct(self, randles_data):
        freq, Z, p = randles_data
        z_lf = Z[-1].real   # ≈ Rs + Rct
        z_hf = Z[0].real    # ≈ Rs
        diameter = z_lf - z_hf
        assert abs(diameter - p["Rct"]) < 10.0, (
            f"Semicircle diameter = {diameter:.2f} Ohm, expected Rct = {p['Rct']} Ohm"
        )


# ---------------------------------------------------------------------------
# Tests: series RC circuit
# ---------------------------------------------------------------------------

class TestSeriesRC:

    def test_zreal_equals_R_at_all_frequencies(self):
        freq = np.logspace(4, -1, 30)
        R, C = 50.0, 1e-6
        Z = rc_series_impedance(freq, R, C)
        np.testing.assert_allclose(
            Z.real, R,
            atol=1e-10,
            err_msg="Z_real of series RC must equal R at all frequencies.",
        )

    def test_zimag_increases_with_decreasing_frequency(self):
        """1/(omega*C) increases as frequency decreases."""
        freq = np.logspace(4, -1, 30)
        R, C = 50.0, 1e-6
        Z = rc_series_impedance(freq, R, C)
        zimag_abs = np.abs(Z.imag)
        # Should be monotonically increasing (largest |Z_imag| at lowest freq)
        assert zimag_abs[-1] > zimag_abs[0], (
            "|Z_imag| must increase as frequency decreases for a series RC."
        )
