"""
patch20 — Band Edge & Mott-Schottky Calculator
===============================================
Computes conduction/valence band edges from Mulliken electronegativity
and band gap, plus flat-band potential and carrier density from
Mott-Schottky analysis.

Theory
------
  E_cb = X - E_c - 0.5 * Eg      (vs vacuum, eV)
  E_vb = E_cb + Eg
  E_cb (vs NHE) = E_cb - 4.44
  E_cb (vs RHE) = E_cb (vs NHE) - 0.0592 * pH

  Mott-Schottky: 1/C² = (2 / ε ε₀ e Nd A²)(E - Vfb - kT/e)

References
----------
  Butler & Ginley (1978) J. Electrochem. Soc.
  Gelderman et al. (2007) J. Chem. Educ.

Author: Hoda Jafari | July 2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Physical constants
_E_C = 4.44          # eV, energy of NHE vs absolute vacuum scale
_k_B = 8.617e-5      # eV / K
_e   = 1.602e-19     # C
_eps0 = 8.854e-12    # F/m


# ─────────────────────────────────────────────────────────────────────────────
# Materials database
# ─────────────────────────────────────────────────────────────────────────────

MATERIALS_DB: Dict[str, Dict] = {
    "g-C3N4":  {"X": 4.73,  "Eg_eV": 2.70,  "label": "g-C₃N₄",      "type": "n"},
    "TiO2":    {"X": 5.81,  "Eg_eV": 3.20,  "label": "TiO₂",        "type": "n"},
    "ZnO":     {"X": 5.79,  "Eg_eV": 3.37,  "label": "ZnO",         "type": "n"},
    "BCN":     {"X": 4.85,  "Eg_eV": None,  "label": "BCN",          "type": "n"},
    "WO3":     {"X": 6.59,  "Eg_eV": 2.60,  "label": "WO₃",         "type": "n"},
    "BiVO4":   {"X": 6.04,  "Eg_eV": 2.40,  "label": "BiVO₄",       "type": "n"},
    "SnO2":    {"X": 6.25,  "Eg_eV": 3.60,  "label": "SnO₂",        "type": "n"},
    "Fe2O3":   {"X": 5.88,  "Eg_eV": 2.20,  "label": "α-Fe₂O₃",     "type": "n"},
    "Cu2O":    {"X": 5.42,  "Eg_eV": 2.17,  "label": "Cu₂O",        "type": "p"},
    "NiO":     {"X": 5.75,  "Eg_eV": 3.50,  "label": "NiO",         "type": "p"},
    "CdS":     {"X": 5.18,  "Eg_eV": 2.42,  "label": "CdS",         "type": "n"},
    "In2S3":   {"X": 5.10,  "Eg_eV": 2.00,  "label": "In₂S₃",      "type": "n"},
    "custom":  {"X": None,  "Eg_eV": None,  "label": "Custom",       "type": "n"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BandEdgeResult:
    material: str
    X: float            # Mulliken electronegativity (eV)
    Eg_eV: float        # optical band gap (eV)
    Ecb_vacuum: float   # conduction band edge vs vacuum (eV)
    Evb_vacuum: float   # valence band edge vs vacuum (eV)
    Ecb_NHE: float
    Evb_NHE: float
    Ecb_RHE: float
    Evb_RHE: float
    pH: float

    def summary(self) -> str:
        return (
            f"{self.material}  Eg={self.Eg_eV:.2f} eV\n"
            f"  E_cb = {self.Ecb_NHE:+.3f} V vs NHE  ({self.Ecb_RHE:+.3f} V vs RHE)\n"
            f"  E_vb = {self.Evb_NHE:+.3f} V vs NHE  ({self.Evb_RHE:+.3f} V vs RHE)\n"
        )


@dataclass
class MottSchottkyResult:
    Vfb_V: float            # flat-band potential (V vs reference)
    Nd_cm3: float           # carrier density (cm⁻³)
    semiconductor_type: str # 'n-type' or 'p-type'
    slope: float            # slope of 1/C² vs V line
    intercept: float
    R2: float               # linearity of the fit
    Vfb_vs_RHE: Optional[float] = None
    warning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Band edge calculator
# ─────────────────────────────────────────────────────────────────────────────

class BandEdgeCalculator:
    """
    Compute semiconductor band edge positions from Mulliken electronegativity.

    Parameters
    ----------
    E_c : float
        Energy of the reference electrode vs absolute vacuum scale (eV).
        Default 4.44 eV (NHE / SHE).
    pH : float
        pH of the electrolyte (for RHE conversion).
    temperature_K : float
        Temperature (K).  Default 298.15 K.
    """

    def __init__(
        self,
        pH: float = 7.0,
        E_c: float = _E_C,
        temperature_K: float = 298.15,
    ):
        self.pH = pH
        self.E_c = E_c
        self.T = temperature_K

    # ── core calculation ──────────────────────────────────────────────────

    def calculate(self, X: float, Eg_eV: float, material: str = "custom") -> BandEdgeResult:
        """
        Calculate band edge positions.

        Parameters
        ----------
        X : float
            Mulliken geometric mean electronegativity of the compound (eV).
        Eg_eV : float
            Optical band gap (eV), e.g. from Tauc plot.
        material : str
            Name / label for the result.

        Returns
        -------
        BandEdgeResult
        """
        Ecb_vac = X - self.E_c - 0.5 * Eg_eV
        Evb_vac = Ecb_vac + Eg_eV

        # vs NHE: subtract absolute scale of NHE (4.44 eV)
        Ecb_nhe = Ecb_vac - 4.44
        Evb_nhe = Evb_vac - 4.44

        # vs RHE
        rhe_shift = 0.0592 * self.pH
        Ecb_rhe = Ecb_nhe - rhe_shift
        Evb_rhe = Evb_nhe - rhe_shift

        return BandEdgeResult(
            material=material,
            X=X,
            Eg_eV=Eg_eV,
            Ecb_vacuum=Ecb_vac,
            Evb_vacuum=Evb_vac,
            Ecb_NHE=Ecb_nhe,
            Evb_NHE=Evb_nhe,
            Ecb_RHE=Ecb_rhe,
            Evb_RHE=Evb_rhe,
            pH=self.pH,
        )

    def from_material_db(
        self, material_key: str, Eg_override: Optional[float] = None
    ) -> BandEdgeResult:
        """
        Look up material from the built-in database and compute band edges.

        Parameters
        ----------
        material_key : str
            Key in MATERIALS_DB (e.g. "g-C3N4", "TiO2").
        Eg_override : float, optional
            Override the database Eg (e.g. from your own Tauc plot).
        """
        if material_key not in MATERIALS_DB:
            raise ValueError(f"Unknown material '{material_key}'.  "
                             f"Available: {list(MATERIALS_DB.keys())}")
        entry = MATERIALS_DB[material_key]
        X = entry["X"]
        Eg = Eg_override if Eg_override is not None else entry["Eg_eV"]
        if X is None or Eg is None:
            raise ValueError(
                f"Material '{material_key}' requires X and Eg_eV. "
                "Set Eg_override (and X via custom entry if needed)."
            )
        return self.calculate(X, Eg, material=entry["label"])

    # ── multi-material comparison ─────────────────────────────────────────

    def compare(
        self,
        material_keys: list,
        Eg_overrides: Optional[Dict[str, float]] = None,
    ):
        """Return a list of BandEdgeResult for several materials (for band diagram)."""
        results = []
        for k in material_keys:
            eg = (Eg_overrides or {}).get(k)
            try:
                results.append(self.from_material_db(k, Eg_override=eg))
            except ValueError as exc:
                warnings.warn(str(exc))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Mott-Schottky analyser
# ─────────────────────────────────────────────────────────────────────────────

class MottSchottkyAnalyzer:
    """
    Flat-band potential and carrier density from Mott-Schottky analysis.

    The Mott-Schottky equation (for a planar electrode):

        1/C² = (2 / ε ε₀ e Nd A²) * (E - Vfb - kT/e)

    A linear region in the 1/C² vs E plot gives:
        slope  -> Nd
        x-intercept -> Vfb

    Parameters
    ----------
    epsilon_r : float
        Relative permittivity (dielectric constant) of the semiconductor.
    area_cm2 : float
        Electrode geometric area (cm²).
    e_ref_vs_nhe : float
        Potential offset of the reference electrode vs NHE (V).
        Used to convert Vfb to the NHE / RHE scale.
    pH : float
        Electrolyte pH.  Used for RHE conversion of Vfb.
    temperature_K : float
        Temperature in Kelvin.
    """

    def __init__(
        self,
        epsilon_r: float = 30.0,
        area_cm2: float = 1.0,
        e_ref_vs_nhe: float = 0.0,
        pH: float = 7.0,
        temperature_K: float = 298.15,
    ):
        self.epsilon_r = epsilon_r
        self.area_cm2 = area_cm2
        self.e_ref_vs_nhe = e_ref_vs_nhe
        self.pH = pH
        self.T = temperature_K

    # ── internal helper: select linear region ────────────────────────────

    @staticmethod
    def _best_linear_region(
        E: np.ndarray, y: np.ndarray, min_points: int = 5
    ) -> Tuple[int, int, float, float, float]:
        """
        Find the longest contiguous window with R² ≥ 0.998.
        Returns (start_idx, end_idx, slope, intercept, R2).
        """
        best = (0, len(E), 0.0, 0.0, 0.0)
        best_len = 0

        for start in range(len(E) - min_points + 1):
            for end in range(start + min_points, len(E) + 1):
                Esub = E[start:end]
                ysub = y[start:end]
                coeffs = np.polyfit(Esub, ysub, 1)
                y_pred = np.polyval(coeffs, Esub)
                ss_res = np.sum((ysub - y_pred) ** 2)
                ss_tot = np.sum((ysub - ysub.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-30)
                n = end - start
                if r2 >= 0.998 and n > best_len:
                    best_len = n
                    best = (start, end, coeffs[0], coeffs[1], r2)

        if best_len == 0:
            # fallback: global linear fit
            coeffs = np.polyfit(E, y, 1)
            y_pred = np.polyval(coeffs, E)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-30)
            best = (0, len(E), coeffs[0], coeffs[1], r2)

        return best

    # ── public API ────────────────────────────────────────────────────────

    def analyze(
        self,
        E_V: np.ndarray,
        C_F: np.ndarray,
        auto_region: bool = True,
        region_slice: Optional[slice] = None,
    ) -> MottSchottkyResult:
        """
        Perform Mott-Schottky analysis.

        Parameters
        ----------
        E_V : ndarray
            Applied potential array (V, vs the reference electrode used in the
            experiment).
        C_F : ndarray
            Capacitance array (F).  If you have C in μF, divide by 1e6 first.
        auto_region : bool
            If True, automatically select the most linear region.
        region_slice : slice, optional
            Manually specify the linear region, e.g. ``slice(10, 30)``.
            Only used when ``auto_region=False``.

        Returns
        -------
        MottSchottkyResult
        """
        E = np.asarray(E_V, dtype=float)
        C = np.asarray(C_F, dtype=float)

        # 1/C²
        C_inv2 = 1.0 / (C ** 2)

        warning = ""

        if auto_region:
            start, end, slope, intercept, R2 = self._best_linear_region(E, C_inv2)
            if R2 < 0.98:
                warning = f"Low R² = {R2:.4f} in the best linear region. Check the data range."
        else:
            sl = region_slice if region_slice is not None else slice(None)
            Esub, ysub = E[sl], C_inv2[sl]
            coeffs = np.polyfit(Esub, ysub, 1)
            y_pred = np.polyval(coeffs, Esub)
            ss_res = np.sum((ysub - y_pred) ** 2)
            ss_tot = np.sum((ysub - ysub.mean()) ** 2)
            R2 = float(1 - ss_res / (ss_tot + 1e-30))
            slope, intercept = float(coeffs[0]), float(coeffs[1])

        # Flat-band potential: Vfb = -intercept/slope + kT/e
        kT_e = _k_B * self.T  # in eV = V at unit charge
        Vfb = -intercept / slope + kT_e

        # Carrier density: Nd = 2 / (e * eps0 * eps_r * A² * slope)
        A_m2 = self.area_cm2 * 1e-4  # cm² -> m²
        Nd_m3 = abs(
            2.0 / (_e * _eps0 * self.epsilon_r * (A_m2 ** 2) * abs(slope))
        )
        Nd_cm3 = Nd_m3 * 1e-6  # m⁻³ -> cm⁻³

        # Semiconductor type
        sc_type = "n-type" if slope > 0 else "p-type"

        # Convert Vfb to NHE and RHE
        Vfb_nhe = Vfb + self.e_ref_vs_nhe
        Vfb_rhe = Vfb_nhe - 0.0592 * self.pH

        return MottSchottkyResult(
            Vfb_V=float(Vfb),
            Nd_cm3=float(Nd_cm3),
            semiconductor_type=sc_type,
            slope=float(slope),
            intercept=float(intercept),
            R2=float(R2),
            Vfb_vs_RHE=float(Vfb_rhe),
            warning=warning,
        )

    # ── convenience: compute C from EIS Cdl ──────────────────────────────

    @staticmethod
    def cdl_to_capacitance(
        Cdl_F: float, cpe_phi: Optional[float] = None, Rs: Optional[float] = None
    ) -> float:
        """
        Convert a CPE-Q value to an effective capacitance C_eff.

        If CPE parameters are provided:
            C_eff = Q^(1/phi) * Rs^((1-phi)/phi)    [Brug formula]
        Otherwise:
            C_eff = Cdl_F   (treated as a pure capacitor)
        """
        if cpe_phi is not None and Rs is not None and cpe_phi > 0:
            return (Cdl_F ** (1.0 / cpe_phi)) * (Rs ** ((1.0 - cpe_phi) / cpe_phi))
        return float(Cdl_F)
