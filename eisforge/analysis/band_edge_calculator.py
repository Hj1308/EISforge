"""
Band Edge & Mott-Schottky Calculator — patch20
=================================================
Computes semiconductor band edge positions (Ecb, Evb) from:
  1. Empirical formula:  Ecb = χ - Ec - 0.5·Eg
  2. Mott-Schottky analysis:  1/C² vs V → Vfb, Nd, semiconductor type

All energies are in eV.  Potentials are in V.
Vacuum reference: Ec = 4.5 eV (conventional absolute scale).

Supported materials in built-in DB
------------------------------------
g-C3N4, TiO2 (anatase/rutile), ZnO, BCN, WO3, Fe2O3, BiVO4
Plus: custom χ and Eg input.

Reference
---------
Butler & Ginley (1978) J. Electrochem. Soc. 125, 228.
Mott-Schottky: Gärtner (1959), Bard & Faulkner (2001) Electrochemical Methods.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as _stats


# ── built-in materials database ───────────────────────────────────────────────

MATERIALS_DB: Dict[str, dict] = {
    "g-C3N4": {
        "chi": 4.73,
        "Eg": 2.70,
        "type": "n",
        "epsilon_r": 8.0,
        "notes": "Graphitic carbon nitride (typical bulk)",
    },
    "TiO2 (anatase)": {
        "chi": 5.81,
        "Eg": 3.20,
        "type": "n",
        "epsilon_r": 55.0,
        "notes": "TiO2 anatase phase",
    },
    "TiO2 (rutile)": {
        "chi": 5.81,
        "Eg": 3.00,
        "type": "n",
        "epsilon_r": 80.0,
        "notes": "TiO2 rutile phase",
    },
    "ZnO": {
        "chi": 5.79,
        "Eg": 3.37,
        "type": "n",
        "epsilon_r": 8.5,
        "notes": "Zinc oxide",
    },
    "BCN": {
        "chi": 4.85,
        "Eg": None,     # must be measured from Tauc plot
        "type": "n",
        "epsilon_r": 7.0,
        "notes": "Boron-carbon-nitride; Eg from Tauc plot required",
    },
    "WO3": {
        "chi": 6.59,
        "Eg": 2.70,
        "type": "n",
        "epsilon_r": 20.0,
        "notes": "Tungsten trioxide",
    },
    "Fe2O3": {
        "chi": 5.88,
        "Eg": 2.10,
        "type": "n",
        "epsilon_r": 12.0,
        "notes": "Hematite",
    },
    "BiVO4": {
        "chi": 6.04,
        "Eg": 2.40,
        "type": "n",
        "epsilon_r": 68.0,
        "notes": "Bismuth vanadate",
    },
    "Custom": {
        "chi": None,
        "Eg": None,
        "type": "n",
        "epsilon_r": 10.0,
        "notes": "User-defined material",
    },
}

# Physical constants
_E_CHARGE = 1.602176634e-19   # C
_EPS0     = 8.8541878128e-12   # F/m
_KB       = 1.380649e-23       # J/K
_EC_VAC   = 4.50               # eV  (vacuum reference to absolute NHE)
_NHE_VS_VAC = 4.44             # eV  (NHE vs absolute vacuum)


# ── result data classes ───────────────────────────────────────────────────────

@dataclasses.dataclass
class BandEdgeResult:
    """Band edge positions computed from χ and Eg."""
    material: str
    chi: float           # electronegativity (eV)
    Eg: float            # band gap (eV)
    Ecb_vac: float       # conduction band vs vacuum (eV)
    Evb_vac: float       # valence band vs vacuum (eV)
    Ecb_NHE: float       # vs NHE (V)
    Evb_NHE: float       # vs NHE (V)
    Ecb_RHE: float       # vs RHE (V)  -- pH-corrected
    Evb_RHE: float       # vs RHE (V)
    pH: float

    def summary(self) -> str:
        return (
            f"{self.material} | Eg={self.Eg:.2f} eV\n"
            f"  Ecb = {self.Ecb_NHE:+.3f} V vs NHE  ({self.Ecb_RHE:+.3f} V vs RHE, pH={self.pH:.1f})\n"
            f"  Evb = {self.Evb_NHE:+.3f} V vs NHE  ({self.Evb_RHE:+.3f} V vs RHE, pH={self.pH:.1f})"
        )


@dataclasses.dataclass
class MottSchottkyResult:
    """Result of Mott-Schottky analysis."""
    V_fb: float                  # flat-band potential (V vs ref)
    Nd: float                    # carrier density (cm⁻³)
    semiconductor_type: str      # 'n-type' or 'p-type'
    slope: float                 # slope of 1/C² vs V
    intercept: float             # intercept
    R2: float                    # R² of linear fit
    V_fit: np.ndarray            # V points used in the fit
    invC2_fit: np.ndarray        # 1/C² points used in the fit
    epsilon_r: float             # relative permittivity used
    electrode_area_cm2: float    # geometric area used

    def summary(self) -> str:
        return (
            f"{self.semiconductor_type} | "
            f"V_fb = {self.V_fb:+.4f} V | "
            f"N_D = {self.Nd:.3e} cm⁻³ | "
            f"R² = {self.R2:.4f}"
        )


# ── main calculator class ─────────────────────────────────────────────────────

class BandEdgeCalculator:
    """
    Compute band edge positions and perform Mott-Schottky analysis.

    Parameters
    ----------
    pH : float
        Solution pH (used for RHE conversion).
    T_celsius : float
        Temperature in °C (default 25).
    """

    def __init__(self, pH: float = 7.0, T_celsius: float = 25.0):
        self.pH = pH
        self.T_K = T_celsius + 273.15

    # ── 1. Band edge from χ and Eg ────────────────────────────────────────────

    def band_edges(
        self,
        chi: float,
        Eg: float,
        material: str = "Custom",
        Ec_ref: float = _EC_VAC,
    ) -> BandEdgeResult:
        """
        Butler-Ginley formula:
            Ecb (vs vacuum) = χ - Ec - 0.5·Eg
            Evb (vs vacuum) = Ecb + Eg

        χ : absolute electronegativity of the semiconductor (eV)
        Eg: optical band gap (eV)
        Ec_ref: vacuum reference level (default 4.50 eV)
        """
        Ecb_vac = chi - Ec_ref - 0.5 * Eg
        Evb_vac = Ecb_vac + Eg

        # vs NHE: subtract _NHE_VS_VAC
        Ecb_NHE = Ecb_vac - _NHE_VS_VAC
        Evb_NHE = Evb_vac - _NHE_VS_VAC

        # vs RHE: E_RHE = E_NHE + 0.0592·pH  →  E_NHE = E_RHE - 0.0592·pH
        # so  Ecb_RHE = Ecb_NHE - 0.0592·pH
        Ecb_RHE = Ecb_NHE - 0.0592 * self.pH
        Evb_RHE = Evb_NHE - 0.0592 * self.pH

        return BandEdgeResult(
            material=material,
            chi=chi,
            Eg=Eg,
            Ecb_vac=Ecb_vac,
            Evb_vac=Evb_vac,
            Ecb_NHE=Ecb_NHE,
            Evb_NHE=Evb_NHE,
            Ecb_RHE=Ecb_RHE,
            Evb_RHE=Evb_RHE,
            pH=self.pH,
        )

    def band_edges_from_db(
        self, material_key: str, Eg_override: Optional[float] = None
    ) -> BandEdgeResult:
        """Compute band edges from the built-in MATERIALS_DB."""
        if material_key not in MATERIALS_DB:
            raise KeyError(
                f"Material '{material_key}' not in DB. "
                f"Available: {list(MATERIALS_DB.keys())}"
            )
        db = MATERIALS_DB[material_key]
        chi = db["chi"]
        Eg = Eg_override if Eg_override is not None else db["Eg"]
        if Eg is None:
            raise ValueError(
                f"Eg for '{material_key}' is not in DB. "
                "Please provide Eg_override (from Tauc plot)."
            )
        return self.band_edges(chi, Eg, material=material_key)

    # ── 2. Mott-Schottky analysis ─────────────────────────────────────────────

    def mott_schottky(
        self,
        V: np.ndarray,
        C_F: np.ndarray,
        epsilon_r: float,
        electrode_area_cm2: float,
        v_range: Optional[Tuple[float, float]] = None,
    ) -> MottSchottkyResult:
        """
        Mott-Schottky analysis: linear fit of 1/C² vs V.

        Parameters
        ----------
        V       : potential array (V vs reference)
        C_F     : capacitance array (F, real part of 1/(jωZ))
        epsilon_r : relative permittivity of semiconductor
        electrode_area_cm2 : geometric area in cm²
        v_range : (V_low, V_high) — restrict fit to this window

        Returns
        -------
        MottSchottkyResult
        """
        V = np.asarray(V, dtype=float)
        C_F = np.asarray(C_F, dtype=float)

        # convert area to m²
        A_m2 = electrode_area_cm2 * 1e-4

        # 1/C² (F⁻²)
        C_safe = np.where(np.abs(C_F) > 1e-20, C_F, np.nan)
        inv_C2 = 1.0 / (C_safe ** 2)

        # optional potential window
        if v_range is not None:
            mask = (V >= v_range[0]) & (V <= v_range[1])
            V_fit, inv_C2_fit = V[mask], inv_C2[mask]
        else:
            valid = np.isfinite(inv_C2)
            V_fit, inv_C2_fit = V[valid], inv_C2[valid]

        if len(V_fit) < 3:
            raise ValueError(
                "Not enough valid points for Mott-Schottky fit "
                f"(need ≥ 3, got {len(V_fit)}). "
                "Check the potential window or data quality."
            )

        slope, intercept, r_val, _, _ = _stats.linregress(V_fit, inv_C2_fit)
        R2 = r_val ** 2

        # flat-band potential: intercept on V-axis  →  Vfb = -intercept / slope
        V_fb = -intercept / slope if abs(slope) > 1e-40 else np.nan

        # carrier density:
        # slope = ± 2 / (e · ε0 · εr · A² · Nd)
        # => Nd = 2 / (|slope| · e · ε0 · εr · A²)
        Nd_m3 = 2.0 / (
            abs(slope) * _E_CHARGE * _EPS0 * epsilon_r * (A_m2 ** 2)
        )
        Nd_cm3 = Nd_m3 * 1e-6   # convert m⁻³ → cm⁻³

        sc_type = "n-type" if slope > 0 else "p-type"

        return MottSchottkyResult(
            V_fb=V_fb,
            Nd=Nd_cm3,
            semiconductor_type=sc_type,
            slope=slope,
            intercept=intercept,
            R2=R2,
            V_fit=V_fit,
            invC2_fit=inv_C2_fit,
            epsilon_r=epsilon_r,
            electrode_area_cm2=electrode_area_cm2,
        )

    # ── 3. Capacitance from EIS data ──────────────────────────────────────────

    @staticmethod
    def capacitance_from_eis(
        freq: np.ndarray,
        z_real: np.ndarray,
        z_imag: np.ndarray,
        target_freq: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute capacitance from EIS data: C = -1 / (2π·f·Z'')

        If target_freq is given, returns a single float at the nearest freq.
        Otherwise returns a full array.
        """
        omega = 2 * np.pi * np.asarray(freq, dtype=float)
        z_imag_arr = np.asarray(z_imag, dtype=float)
        # Use negative sign: Z'' is negative for capacitive element
        # C = 1 / (ω · |Z''|)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / (omega * np.abs(z_imag_arr))
        if target_freq is not None:
            idx = np.argmin(np.abs(np.asarray(freq) - target_freq))
            return float(C[idx])
        return C
