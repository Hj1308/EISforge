"""
Band Edge & Mott-Schottky Calculator
=====================================
Computes semiconductor band edge positions (Ecb, Evb) from:
  1. Butler-Ginley empirical formula:  Ecb_NHE = χ - EC_REF_VAC - 0.5·Eg  (V vs NHE)
  2. Mott-Schottky analysis:  1/C² vs V → Vfb, Nd, semiconductor type

All energies are in eV.  Potentials are in V.
EC_REF_VAC = 4.50 eV is the NHE level on the vacuum scale used by the Butler-Ginley formula, so
χ - 4.50 - Eg/2 is already a potential vs NHE; vacuum energies are E_vac = -(E_NHE + 4.50) eV.
(E_NHE_OFFSET = 4.44 eV, Trasatti/IUPAC, is kept for reference and is NOT subtracted again.)
RHE: E_RHE = E_NHE + (RT ln10 / F)·pH  (0.05916 V/pH at 25 °C).
patch43 (B11): the old code subtracted 4.44 V a second time (TiO2 Ecb = -4.73 V vs NHE instead of
-0.29 V) and used the wrong sign for RHE.

Reference
---------
Butler & Ginley (1978) J. Electrochem. Soc. 125, 228.
Mott-Schottky: Gärtner (1959), Bard & Faulkner (2001).

API
---
    calc = BandEdgeCalculator("TiO2", pH=0.0)
    r    = calc.calculate()
    r.Ecb_vacuum, r.Ecb_NHE, r.Ecb_RHE

    calc = BandEdgeCalculator("custom", X=4.85, Eg=2.15, pH=7.0)
    ms   = BandEdgeCalculator("TiO2", electrode_area=1.0).mott_schottky(V, C)
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as _stats


# ── exported constants ────────────────────────────────────────────────────────

#: NHE vs. vacuum reference (eV) — conventional absolute scale
E_NHE_OFFSET: float = 4.44   # eV, absolute NHE vs vacuum (Trasatti/IUPAC)
EC_REF_VAC: float = 4.50     # eV, vacuum reference of the empirical band-edge formula

#: Nernst slope at 25 °C: 2.303·RT/F (V/pH)
NERNST_SLOPE: float = 0.05916

def _nernst_slope(T_celsius: float = 25.0) -> float:
    """RT ln10 / F in V per pH unit (0.05916 V at 25 °C)."""
    return 8.314462618 * (273.15 + float(T_celsius)) / 96485.33212 * np.log(10.0)


# Internal physical constants
_E_CHARGE = 1.602176634e-19   # C
_EPS0     = 8.8541878128e-12  # F/m
_DEFAULT_EPSILON_R = 10.0     # dimensionless (fallback)


# ── built-in materials database ───────────────────────────────────────────────

MATERIALS_DB: Dict[str, dict] = {
    "TiO2": {
        "chi": 5.81, "Eg": 3.20, "type": "n", "epsilon_r": 55.0,
        "notes": "TiO2 (anatase)",
    },
    "TiO2 (anatase)": {
        "chi": 5.81, "Eg": 3.20, "type": "n", "epsilon_r": 55.0,
        "notes": "TiO2 anatase phase",
    },
    "TiO2 (rutile)": {
        "chi": 5.81, "Eg": 3.00, "type": "n", "epsilon_r": 80.0,
        "notes": "TiO2 rutile phase",
    },
    "g-C3N4": {
        "chi": 4.73, "Eg": 2.70, "type": "n", "epsilon_r": 8.0,
        "notes": "Graphitic carbon nitride",
    },
    "ZnO": {
        "chi": 5.79, "Eg": 3.37, "type": "n", "epsilon_r": 8.5,
        "notes": "Zinc oxide",
    },
    "BCN": {
        "chi": 4.85, "Eg": None,  "type": "n", "epsilon_r": 7.0,
        "notes": "Boron-carbon-nitride; Eg from Tauc plot required",
    },
    "WO3": {
        "chi": 6.59, "Eg": 2.70, "type": "n", "epsilon_r": 20.0,
        "notes": "Tungsten trioxide",
    },
    "Fe2O3": {
        "chi": 5.88, "Eg": 2.10, "type": "n", "epsilon_r": 12.0,
        "notes": "Hematite",
    },
    "BiVO4": {
        "chi": 6.04, "Eg": 2.40, "type": "n", "epsilon_r": 68.0,
        "notes": "Bismuth vanadate",
    },
    "custom": {
        "chi": None, "Eg": None, "type": "n", "epsilon_r": _DEFAULT_EPSILON_R,
        "notes": "User-defined material; X and Eg must be provided",
    },
}


# ── result data classes ───────────────────────────────────────────────────────

@dataclasses.dataclass
class BandEdgeResult:
    """Band edge positions computed from \u03c7 and Eg."""
    material: str
    chi: float
    Eg: float
    Ecb_vacuum: float    # conduction-band energy vs vacuum (eV)  [= -(Ecb_NHE + EC_REF_VAC) = -chi + Eg/2]
    Evb_vacuum: float    # valence-band energy vs vacuum (eV)     [= Ecb_vacuum - Eg]
    Ecb_NHE: float       # vs NHE (V)  [= chi - EC_REF_VAC - 0.5*Eg, Butler-Ginley]
    Evb_NHE: float       # vs NHE (V)  [= Ecb_NHE + Eg]
    Ecb_RHE: float       # vs RHE (V)  [= Ecb_NHE + (RT ln10/F)*pH]
    Evb_RHE: float       # vs RHE (V)
    pH: float

    # Legacy aliases for code that uses old field names
    @property
    def Ecb_vac(self) -> float:
        return self.Ecb_vacuum

    @property
    def Evb_vac(self) -> float:
        return self.Evb_vacuum

    def summary(self) -> str:
        return (
            f"{self.material} | Eg={self.Eg:.2f} eV\n"
            f"  Ecb = {self.Ecb_NHE:+.3f} V vs NHE "
            f"({self.Ecb_RHE:+.3f} V vs RHE, pH={self.pH:.1f})\n"
            f"  Evb = {self.Evb_NHE:+.3f} V vs NHE "
            f"({self.Evb_RHE:+.3f} V vs RHE, pH={self.pH:.1f})"
        )


@dataclasses.dataclass
class MottSchottkyResult:
    """Result of Mott-Schottky analysis."""
    Vfb: float               # flat-band potential (V vs ref)  ← test expects .Vfb
    Nd: float                # carrier density (cm⁻³)
    sc_type: str             # 'n-type' or 'p-type'            ← test expects .sc_type
    slope: float
    intercept: float
    R2: float
    V_fit: np.ndarray
    invC2_fit: np.ndarray
    epsilon_r: float
    electrode_area_cm2: float
    V_intercept: float = float("nan")   # raw V-axis intercept of 1/C^2 (patch43)

    # Legacy aliases
    @property
    def V_fb(self) -> float:
        return self.Vfb

    @property
    def semiconductor_type(self) -> str:
        return self.sc_type

    def summary(self) -> str:
        return (
            f"{self.sc_type} | "
            f"Vfb = {self.Vfb:+.4f} V | "
            f"N_D = {self.Nd:.3e} cm\u207b\u00b3 | "
            f"R\u00b2 = {self.R2:.4f}"
        )


# ── main calculator class ─────────────────────────────────────────────────────

class BandEdgeCalculator:
    """
    Compute band edge positions and perform Mott-Schottky analysis.

    Parameters
    ----------
    material : str
        Material key from MATERIALS_DB, or "custom" for user-defined \u03c7/Eg.
        Examples: "TiO2", "g-C3N4", "ZnO", "custom"
    pH : float, optional
        Solution pH for RHE conversion (default 7.0).
    X : float, optional
        Absolute electronegativity in eV.  Required when material="custom".
    Eg : float, optional
        Band gap in eV.  Overrides the DB value when provided.
    electrode_area : float, optional
        Electrode geometric area in cm\u00b2. Needed for Mott-Schottky analysis.
    T_celsius : float, optional
        Temperature in \u00b0C (default 25).

    Examples
    --------
    >>> calc = BandEdgeCalculator("TiO2", pH=0.0)
    >>> r = calc.calculate()
    >>> calc_ms = BandEdgeCalculator("TiO2", electrode_area=1.0)
    >>> ms = calc_ms.mott_schottky(V_array, C_array)
    >>> calc_custom = BandEdgeCalculator("custom", X=4.85, Eg=2.15, pH=7.0)
    """

    def __init__(
        self,
        material: str = "custom",
        pH: float = 7.0,
        X: Optional[float] = None,
        Eg: Optional[float] = None,
        electrode_area: Optional[float] = None,
        T_celsius: float = 25.0,
    ) -> None:
        self.material = material
        self.pH = float(pH)
        self.T_celsius = T_celsius

        # Resolve chi and Eg from DB or user overrides
        mat_key = material if material in MATERIALS_DB else "custom"
        db = MATERIALS_DB.get(mat_key, MATERIALS_DB["custom"])

        # chi: prefer X (keyword) over DB value
        self._chi: Optional[float] = X if X is not None else db.get("chi")

        # Eg: prefer keyword override over DB value
        self._Eg: Optional[float] = Eg if Eg is not None else db.get("Eg")

        # Mott-Schottky setup
        self._epsilon_r: float = float(db.get("epsilon_r") or _DEFAULT_EPSILON_R)
        self._electrode_area: Optional[float] = electrode_area

    # ── band edge calculation ─────────────────────────────────────────────────

    def calculate(self, Eg_override: Optional[float] = None) -> BandEdgeResult:
        """
        Compute band edge positions.

        Butler-Ginley formula
        ---------------------
            Ecb_vacuum = \u03c7 - EC_REF_VAC - 0.5\u00b7Eg
            Evb_vacuum = Ecb_vacuum + Eg
            Ecb_NHE   = Ecb_vacuum - E_NHE_OFFSET   (NHE = vacuum − 4.44 eV, E_NHE_OFFSET)

        Note: EC_REF_VAC (4.50 eV) is the empirical Butler-Ginley constant;
        E_NHE_OFFSET (4.44 eV) is the absolute NHE scale (Trasatti/IUPAC).
        These are two separate subtractive steps — NOT a combined offset.

        Parameters
        ----------
        Eg_override : float, optional
            Override band gap for this call only.

        Returns
        -------
        BandEdgeResult
        """
        chi = self._chi
        Eg  = Eg_override if Eg_override is not None else self._Eg

        if chi is None:
            raise ValueError(
                f"Electronegativity (chi/X) is not set for material '{self.material}'. "
                "Pass X=... to the constructor."
            )
        if Eg is None or Eg <= 0:
            raise ValueError(
                f"Band gap Eg is not set or invalid for material '{self.material}'. "
                "Pass Eg=... to the constructor or as Eg_override."
            )

        # Butler-Ginley: chi - E^e - Eg/2 with E^e = EC_REF_VAC (4.50 eV, the NHE level vs vacuum)
        # is already a potential vs NHE (TiO2: -0.29 V). patch43 (B11): it used to be labelled
        # "vacuum" and E_NHE_OFFSET (4.44) was subtracted a second time (TiO2: -4.73 V).
        Ecb_NHE = chi - EC_REF_VAC - 0.5 * Eg
        Evb_NHE = Ecb_NHE + Eg

        # Absolute (vacuum) energy scale in eV: E_vac = -(E_NHE + EC_REF_VAC)
        Ecb_vac = -(Ecb_NHE + EC_REF_VAC)
        Evb_vac = -(Evb_NHE + EC_REF_VAC)

        # RHE: E_RHE = E_NHE + (RT ln10 / F) * pH   (patch43: the sign was negative)
        _s = _nernst_slope(self.T_celsius)
        Ecb_RHE = Ecb_NHE + _s * self.pH
        Evb_RHE = Evb_NHE + _s * self.pH

        return BandEdgeResult(
            material=self.material,
            chi=float(chi),
            Eg=float(Eg),
            Ecb_vacuum=Ecb_vac,
            Evb_vacuum=Evb_vac,
            Ecb_NHE=Ecb_NHE,
            Evb_NHE=Evb_NHE,
            Ecb_RHE=Ecb_RHE,
            Evb_RHE=Evb_RHE,
            pH=self.pH,
        )

    # ── Mott-Schottky analysis ────────────────────────────────────────────────

    def mott_schottky(
        self,
        V: np.ndarray,
        C: np.ndarray,
        epsilon_r: Optional[float] = None,
        electrode_area_cm2: Optional[float] = None,
        v_range: Optional[Tuple[float, float]] = None,
    ) -> MottSchottkyResult:
        """
        Mott-Schottky analysis: linear fit of 1/C\u00b2 vs V.

        Parameters
        ----------
        V              : potential array (V vs reference)
        C              : capacitance array (F)
        epsilon_r      : relative permittivity (overrides DB value)
        electrode_area_cm2 : geometric area in cm\u00b2 (overrides constructor value)
        v_range        : (V_low, V_high) — restrict fit to this window

        Returns
        -------
        MottSchottkyResult  with .Vfb, .sc_type, .Nd, .R2
        """
        V   = np.asarray(V, dtype=float)
        C   = np.asarray(C, dtype=float)

        eps_r = epsilon_r if epsilon_r is not None else self._epsilon_r
        area_cm2 = (
            electrode_area_cm2
            if electrode_area_cm2 is not None
            else self._electrode_area
        )
        if area_cm2 is None:
            raise ValueError(
                "electrode_area_cm2 must be provided either in the constructor "
                "(electrode_area=...) or as an argument to mott_schottky()."
            )
        A_m2 = float(area_cm2) * 1e-4

        C_safe  = np.where(np.abs(C) > 1e-20, C, np.nan)
        inv_C2  = 1.0 / (C_safe ** 2)

        if v_range is not None:
            mask = (V >= v_range[0]) & (V <= v_range[1])
            V_fit, inv_C2_fit = V[mask], inv_C2[mask]
        else:
            valid = np.isfinite(inv_C2)
            V_fit, inv_C2_fit = V[valid], inv_C2[valid]

        if len(V_fit) < 3:
            raise ValueError(
                f"Not enough valid points for Mott-Schottky fit "
                f"(need \u2265 3, got {len(V_fit)})."
            )

        slope, intercept, r_val, _, _ = _stats.linregress(V_fit, inv_C2_fit)
        R2   = r_val ** 2
        V_int = -intercept / slope if abs(slope) > 1e-40 else float("nan")
        # Full Mott-Schottky equation: 1/C^2 = 2/(e eps eps0 A^2 N) (V - Vfb - kT/e) for n-type
        # (V - Vfb + kT/e for p-type), so the V-axis intercept is Vfb + kT/e (n) or Vfb - kT/e (p).
        # patch43: Vfb used to be the raw intercept (26 mV off at 25 C).
        _kT_e = 8.617333262e-5 * (273.15 + float(self.T_celsius))
        Vfb = V_int - _kT_e if slope > 0 else V_int + _kT_e

        Nd_m3  = 2.0 / (abs(slope) * _E_CHARGE * _EPS0 * eps_r * (A_m2 ** 2))
        Nd_cm3 = Nd_m3 * 1e-6

        sc_type = "n-type" if slope > 0 else "p-type"

        return MottSchottkyResult(
            Vfb=Vfb,
            Nd=Nd_cm3,
            sc_type=sc_type,
            slope=slope,
            intercept=intercept,
            R2=R2,
            V_fit=V_fit,
            invC2_fit=inv_C2_fit,
            epsilon_r=eps_r,
            electrode_area_cm2=float(area_cm2),
            V_intercept=V_int,
        )

    # ── EIS-based capacitance ─────────────────────────────────────────────────

    @staticmethod
    def capacitance_from_eis(
        freq: np.ndarray,
        z_real: np.ndarray,
        z_imag: np.ndarray,
        target_freq: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute capacitance from EIS data: C = 1 / (2\u03c0\u00b7f\u00b7|Z''|)

        Parameters
        ----------
        freq       : frequency array (Hz)
        z_real     : real part of impedance (\u03a9)
        z_imag     : imaginary part of impedance (\u03a9)
        target_freq: if given, returns single float at nearest frequency
        """
        omega      = 2 * np.pi * np.asarray(freq, dtype=float)
        z_imag_arr = np.asarray(z_imag, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / (omega * np.abs(z_imag_arr))
        if target_freq is not None:
            idx = int(np.argmin(np.abs(np.asarray(freq) - target_freq)))
            return float(C[idx])
        return C

    # ── legacy helper kept for backward compatibility ─────────────────────────

    def band_edges(
        self,
        chi: float,
        Eg: float,
        material: str = "Custom",
        Ec_ref: float = EC_REF_VAC,
    ) -> BandEdgeResult:
        """
        Legacy method: compute band edges directly from chi and Eg.
        Prefer using calculate() on an instance constructed with material name.
        """
        tmp = BandEdgeCalculator("custom", pH=self.pH, X=chi, Eg=Eg, T_celsius=self.T_celsius)
        r   = tmp.calculate()
        return dataclasses.replace(r, material=material)

    def band_edges_from_db(
        self, material_key: str, Eg_override: Optional[float] = None
    ) -> BandEdgeResult:
        """Legacy helper: compute from built-in DB entry."""
        tmp = BandEdgeCalculator(material_key, pH=self.pH, Eg=Eg_override,
                                 T_celsius=self.T_celsius)
        return tmp.calculate()
