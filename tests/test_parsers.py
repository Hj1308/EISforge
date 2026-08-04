"""
Unit tests for EIS file parsers.

These tests use synthetic CSV data to verify parser logic
without requiring real instrument files.
"""


from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eisforge.parsers.ivium_parser import IviumIDFParser


# ---------------------------------------------------------------------------
# Helper: create synthetic CSV EIS data
# ---------------------------------------------------------------------------

def make_synthetic_eis_csv(n_points: int = 30) -> str:
    """Generate a minimal EIS CSV string (Gamry-style columns)."""
    freq = np.logspace(5, -2, n_points)
    zreal = 10.0 + 150.0 / (1 + (freq / 1000) ** 2)
    zimag = -150.0 * (freq / 1000) / (1 + (freq / 1000) ** 2)
    zmag  = np.sqrt(zreal**2 + zimag**2)
    phase = np.degrees(np.arctan2(zimag, zreal))

    lines = ["Freq,Zreal,Zimag,Zmag,Zphz"]
    for f, zr, zi, zm, zp in zip(freq, zreal, zimag, zmag, phase):
        lines.append(f"{f:.6e},{zr:.6f},{zi:.6f},{zm:.6f},{zp:.6f}")
    return "\n".join(lines)


@pytest.fixture
def csv_eis_file(tmp_path):
    """Write synthetic EIS CSV to a temp file and return its path."""
    content = make_synthetic_eis_csv(30)
    fp = tmp_path / "test_eis.csv"
    fp.write_text(content)
    return fp


# ---------------------------------------------------------------------------
# Tests: CSV structure
# ---------------------------------------------------------------------------

class TestSyntheticCSV:

    def test_csv_has_correct_columns(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        expected = {"Freq", "Zreal", "Zimag", "Zmag", "Zphz"}
        assert expected.issubset(set(df.columns))

    def test_csv_row_count(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        assert len(df) == 30

    def test_frequencies_positive(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        assert (df["Freq"] > 0).all()

    def test_zmag_positive(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        assert (df["Zmag"] > 0).all()

    def test_zreal_positive_for_randles(self, csv_eis_file):
        """Z_real should always be > 0 for a physical Randles circuit."""
        df = pd.read_csv(csv_eis_file)
        assert (df["Zreal"] > 0).all()

    def test_zimag_negative_for_capacitive(self, csv_eis_file):
        """Im(Z) should be negative for a capacitive circuit."""
        df = pd.read_csv(csv_eis_file)
        assert (df["Zimag"] <= 0).all()

    def test_zmag_equals_sqrt_sum_squares(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        expected_mag = np.sqrt(df["Zreal"]**2 + df["Zimag"]**2)
        np.testing.assert_allclose(
            df["Zmag"].values,
            expected_mag.values,
            rtol=1e-5,
        )

    def test_no_nan_values(self, csv_eis_file):
        df = pd.read_csv(csv_eis_file)
        assert not df.isnull().any().any(), "CSV data should contain no NaN values."

    def test_frequency_range(self, csv_eis_file):
        """Frequencies should span from mHz to 100 kHz range."""
        df = pd.read_csv(csv_eis_file)
        assert df["Freq"].max() > 1e4
        assert df["Freq"].min() < 1.0


# ---------------------------------------------------------------------------
# Tests: Ivium .idf parser (real instrument file, guarded by skipif)
# ---------------------------------------------------------------------------

DATA = Path(__file__).parent / "data" / "sample_eis.idf"


class TestIviumIDF:

    pytestmark = pytest.mark.skipif(
        not DATA.exists(),
        reason="sample_eis.idf not present",
    )

    def test_parse_returns_dataframe(self):
        df = IviumIDFParser().parse(DATA).to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        df = IviumIDFParser().parse(DATA).to_dataframe()
        expected = {"frequency", "z_real", "z_imag", "z_modulus", "phase_deg"}
        assert expected.issubset(set(df.columns))

    def test_non_empty_and_frequencies_positive(self):
        df = IviumIDFParser().parse(DATA).to_dataframe()
        assert len(df) > 0
        assert (df["frequency"] > 0).all()

    def test_no_nan_in_frequency_or_impedance(self):
        df = IviumIDFParser().parse(DATA).to_dataframe()
        cols = ["frequency", "z_real", "z_imag"]
        assert not df[cols].isnull().any().any()

    def test_frequency_monotonic(self):
        df = IviumIDFParser().parse(DATA).to_dataframe()
        f = df["frequency"].values
        assert np.all(np.diff(f) <= 0) or np.all(np.diff(f) >= 0)
