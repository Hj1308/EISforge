"""
Generate synthetic EIS data for testing and demonstration.

Run this script once to create example CSV files:
    python examples/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_randles(Rs=10.0, Rct=150.0, Cdl=50e-6, n_points=60):
    """
    Simulate a Randles circuit: Rs + Rct//(Cdl)

    Parameters
    ----------
    Rs      : solution resistance (Ohm)
    Rct     : charge-transfer resistance (Ohm)
    Cdl     : double-layer capacitance (F)
    n_points: number of frequency points

    Returns
    -------
    pd.DataFrame with columns: Freq, Zreal, Zimag, Zmag, Zphz
    """
    freq  = np.logspace(5, -2, n_points)          # 100 kHz → 10 mHz
    omega = 2 * np.pi * freq
    Z_rc  = Rct / (1 + 1j * omega * Rct * Cdl)
    Z     = Rs + Z_rc

    df = pd.DataFrame({
        "Freq" : freq,
        "Zreal": Z.real,
        "Zimag": Z.imag,
        "Zmag" : np.abs(Z),
        "Zphz" : np.degrees(np.angle(Z)),
    })
    return df


def generate_warburg(Rs=5.0, Rct=80.0, Cdl=30e-6, Aw=20.0, n_points=60):
    """
    Randles circuit with semi-infinite Warburg element.
    Z_W = Aw * (1 - j) / sqrt(omega)
    """
    freq  = np.logspace(5, -3, n_points)
    omega = 2 * np.pi * freq
    Z_W   = Aw * (1 - 1j) / np.sqrt(omega)
    Z_rc  = (Rct + Z_W) / (1 + 1j * omega * Cdl * (Rct + Z_W))
    Z     = Rs + Z_rc

    df = pd.DataFrame({
        "Freq" : freq,
        "Zreal": Z.real,
        "Zimag": Z.imag,
        "Zmag" : np.abs(Z),
        "Zphz" : np.degrees(np.angle(Z)),
    })
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).parent

    # Randles circuit
    df_randles = generate_randles()
    fp_randles = out_dir / "synthetic_randles.csv"
    df_randles.to_csv(fp_randles, index=False)
    print(f"Saved: {fp_randles}  ({len(df_randles)} points)")

    # Randles + Warburg
    df_warburg = generate_warburg()
    fp_warburg = out_dir / "synthetic_warburg.csv"
    df_warburg.to_csv(fp_warburg, index=False)
    print(f"Saved: {fp_warburg}  ({len(df_warburg)} points)")

    print("\nDone. Use these files to test EISforge or run Quick Start examples.")
