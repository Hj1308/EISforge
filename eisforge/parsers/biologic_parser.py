"""
BioLogic .mpr / .mpt file parser using galvani.

Supports BioLogic EC-Lab binary (.mpr) and text (.mpt) formats.
Requires: pip install galvani
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from galvani import BioLogic
except ImportError as e:
    raise ImportError(
        "galvani is required for BioLogic files.\n"
        "Install with: pip install galvani"
    ) from e


def parse_biologic(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a BioLogic .mpr or .mpt file and return EIS data as a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the BioLogic file (.mpr binary or .mpt text).

    Returns
    -------
    pd.DataFrame
        Columns:
            - freq_hz       : frequency in Hz (float64)
            - z_real        : real part of impedance in Ohm (float64)
            - z_imag        : imaginary part, sign-corrected so capacitive > 0 (float64)
            - z_mag         : impedance magnitude |Z| in Ohm (float64)
            - z_phase_deg   : phase angle in degrees (float64)
        Sorted by decreasing frequency (standard EIS convention).

    Raises
    ------
    ValueError
        If the file extension is not .mpr or .mpt.
    KeyError
        If expected frequency/impedance columns are not found.

    Examples
    --------
    >>> df = parse_biologic("experiment.mpr")
    >>> print(df.head())
    """
    fp = Path(filepath)
    if fp.suffix.lower() not in {".mpr", ".mpt"}:
        raise ValueError(
            f"Expected a BioLogic .mpr or .mpt file, got: '{fp.suffix}'\n"
            f"File: {fp.name}"
        )

    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    mpr = BioLogic.MPRfile(str(fp))
    df_raw = pd.DataFrame(mpr.data)

    # Locate frequency and impedance columns (vary by EC-Lab firmware version)
    freq_col  = _find_col(df_raw, ["freq/Hz", "Frequency", "frequency", "f", "freq"])
    zreal_col = _find_col(df_raw, ["Re(Z)/Ohm", "Re(Z)", "Zreal", "Z_re", "Z'"])
    zimag_col = _find_col(df_raw, ["-Im(Z)/Ohm", "-Im(Z)", "Zimag", "Z_im", "Z''"])

    freq  = df_raw[freq_col].astype(float).values
    zreal = df_raw[zreal_col].astype(float).values

    # BioLogic stores -Im(Z) as positive for capacitive arcs;
    # we keep the imaginary part positive (capacitive convention).
    zimag_raw = df_raw[zimag_col].astype(float).values
    # Detect sign: if the column name starts with '-', the stored value is already -Im(Z)
    if zimag_col.startswith("-"):
        zimag = zimag_raw  # already -Im(Z), keep as-is (capacitive positive)
    else:
        zimag = -zimag_raw  # flip sign to make capacitive arcs positive

    df = pd.DataFrame({
        "freq_hz"     : freq,
        "z_real"      : zreal,
        "z_imag"      : zimag,
        "z_mag"       : np.sqrt(zreal**2 + zimag**2),
        "z_phase_deg" : np.degrees(np.arctan2(-zimag, zreal)),
    })

    return (
        df.dropna(subset=["freq_hz", "z_real", "z_imag"])
          .sort_values("freq_hz", ascending=False)
          .reset_index(drop=True)
    )


def validate_eis_dataframe(df: pd.DataFrame) -> None:
    """
    Basic sanity checks on a parsed EIS DataFrame.

    Raises
    ------
    ValueError
        If the DataFrame fails any validation check.
    """
    required = {"freq_hz", "z_real", "z_imag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty — no EIS data found in file.")

    if (df["freq_hz"] <= 0).any():
        raise ValueError("All frequencies must be positive (> 0 Hz).")

    if df["freq_hz"].nunique() < len(df) * 0.9:
        import warnings
        warnings.warn(
            "More than 10% of frequency values are duplicated. "
            "Check if the file contains multiple EIS spectra.",
            UserWarning,
            stacklevel=2,
        )


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of the expected column names found.\n"
        f"Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )
