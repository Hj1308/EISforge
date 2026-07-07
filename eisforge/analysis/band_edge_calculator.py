"""
eiforge/analysis/band_edge_calculator.py
==========================================
Semiconductor band edge position calculator for photocatalyst research.

Supported:
  - Ecb / Evb from Mulliken electronegativity (Butler & Ginley, 1978)
  - Mott-Schottky analysis  ->  flat-band potential, carrier density, n/p-type
  - Tauc plot               ->  optical band gap (direct / indirect transitions)
  - RHE / NHE / vacuum scale conversions

Built-in materials: g-C3N4, BCN, TiO2, ZnO, BiVO4, WO3, Fe2O3, CdS.

References:
  Butler & Ginley (1978) J. Electrochem. Soc. 125, 228
  Xu & Schoonen (2000) Am. Mineral. 85, 543
  Gelderman et al. (2007) J. Chem. Educ. 84, 685
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

EC_DEFAULT   = 4.5
E_NHE_OFFSET = 4.44
NERNST_SLOPE = 0.05916
EPSILON_0    = 8.854e-12
Q_E          = 1.602e-19

MATERIALS_DB: dict = {
    "g-C3N4":      {"X": 4.73, "Eg": 2.70, "type": "n", "epsilon_r":   8.0},
    "TiO2":        {"X": 5.81, "Eg": 3.20, "type": "n", "epsilon_r":  55.0},
    "TiO2_rutile": {"X": 5.81, "Eg": 3.00, "type": "n", "epsilon_r": 114.0},
    "ZnO":         {"X": 5.79, "Eg": 3.37, "type": "n", "epsilon_r":   8.5},
    "BiVO4":       {"X": 6.04, "Eg": 2.40, "type": "n", "epsilon_r":  68.0},
    "WO3":         {"X": 6.59, "Eg": 2.70, "type": "n", "epsilon_r":  20.0},
    "Fe2O3":       {"X": 5.88, "Eg": 2.20, "type": "n", "epsilon_r":  12.0},
    "CdS":         {"X": 4.88, "Eg": 2.42, "type": "n", "epsilon_r":   8.9},
    "BCN":         {"X": 4.85, "Eg": None, "type": "n", "epsilon_r":   6.0},
}


@dataclass
class BandEdgeResult:
    material: str
    X: float
    Eg: float
    Ec: float
    pH: float
    Ecb_vacuum: float = field(init=False)
    Evb_vacuum: float = field(init=False)
    Ecb_NHE:    float = field(init=False)
    Evb_NHE:    float = field(init=False)
    Ecb_RHE:    float = field(init=False)
    Evb_RHE:    float = field(init=False)

    def __post_init__(self):
        self.Ecb_vacuum = self.X - self.Ec - 0.5 * self.Eg
        self.Evb_vacuum = self.Ecb_vacuum + self.Eg
        self.Ecb_NHE    = self.Ecb_vacuum - E_NHE_OFFSET
        self.Evb_NHE    = self.Evb_vacuum - E_NHE_OFFSET
        self.Ecb_RHE    = self.Ecb_NHE - NERNST_SLOPE * self.pH
        self.Evb_RHE    = self.Evb_NHE - NERNST_SLOPE * self.pH

    def summary(self) -> str:
        return (
            f"Material : {self.material}\n"
            f"Eg       : {self.Eg:.3f} eV\n"
            f"X        : {self.X:.3f} eV\n"
            f"pH       : {self.pH}\n\n"
            f"Ecb (vacuum) = {self.Ecb_vacuum:+.3f} eV\n"
            f"Evb (vacuum) = {self.Evb_vacuum:+.3f} eV\n"
            f"Ecb (NHE)    = {self.Ecb_NHE:+.3f} V\n"
            f"Evb (NHE)    = {self.Evb_NHE:+.3f} V\n"
            f"Ecb (RHE)    = {self.Ecb_RHE:+.3f} V  (pH {self.pH})\n"
            f"Evb (RHE)    = {self.Evb_RHE:+.3f} V  (pH {self.pH})"
        )


@dataclass
class MottSchottkyResult:
    Vfb:       float
    Nd:        float
    sc_type:   str
    slope:     float
    intercept: float
    R2:        float


@dataclass
class TaucResult:
    Eg:         float
    transition: str
    R2:         float
    E_fit:      "np.ndarray" = field(repr=False, default_factory=lambda: np.array([]))
    alpha_fit:  "np.ndarray" = field(repr=False, default_factory=lambda: np.array([]))


class BandEdgeCalculator:
    """
    Semiconductor band edge position calculator.

    Parameters
    ----------
    material        : name from MATERIALS_DB or 'custom'
    X               : Mulliken electronegativity (eV); required for custom
    Eg              : band gap (eV); required for custom or BCN
    Ec              : free-electron energy reference (default 4.5 eV)
    pH              : electrolyte pH for RHE conversion (default 7.0)
    epsilon_r       : relative permittivity (for Mott-Schottky Nd)
    electrode_area  : geometric area (cm2) for Mott-Schottky Nd
    """

    def __init__(
        self,
        material: str = "custom",
        X: Optional[float] = None,
        Eg: Optional[float] = None,
        Ec: float = EC_DEFAULT,
        pH: float = 7.0,
        epsilon_r: float = 10.0,
        electrode_area: float = 1.0,
    ):
        self.material = material
        self.Ec = Ec
        self.pH = pH
        self.epsilon_r = epsilon_r
        self.electrode_area = electrode_area

        if material in MATERIALS_DB:
            db = MATERIALS_DB[material]
            self.X  = X  if X  is not None else db["X"]
            self.Eg = Eg if Eg is not None else db["Eg"]
            if epsilon_r == 10.0:
                self.epsilon_r = db["epsilon_r"]
        else:
            if X is None or Eg is None:
                raise ValueError("For custom materials provide both X and Eg.")
            self.X  = X
            self.Eg = Eg

        if self.Eg is None:
            raise ValueError(
                f"Eg is required for {material}. "
                "Provide Eg from your Tauc plot."
            )

    def calculate(self) -> BandEdgeResult:
        """Return Ecb / Evb on vacuum, NHE, and RHE scales."""
        return BandEdgeResult(
            material=self.material,
            X=self.X,
            Eg=self.Eg,
            Ec=self.Ec,
            pH=self.pH,
        )

    def mott_schottky(
        self,
        V: "np.ndarray",
        C: "np.ndarray",
        fit_range=None,
    ) -> MottSchottkyResult:
        """
        Mott-Schottky analysis (1/C^2 vs V).

        V         : potential array (V vs reference, ascending)
        C         : capacitance array (F)
        fit_range : (V_min, V_max) to restrict linear fit; None = full range
        """
        V = np.asarray(V, dtype=float)
        C = np.asarray(C, dtype=float)
        if fit_range is not None:
            mask = (V >= fit_range[0]) & (V <= fit_range[1])
            V, C = V[mask], C[mask]

        C_inv2 = 1.0 / (C ** 2)
        coeffs = np.polyfit(V, C_inv2, 1)
        slope, intercept = coeffs
        y_pred = np.polyval(coeffs, V)
        ss_res = np.sum((C_inv2 - y_pred) ** 2)
        ss_tot = np.sum((C_inv2 - np.mean(C_inv2)) ** 2)
        R2  = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        Vfb = -intercept / slope

        A_m2   = self.electrode_area * 1e-4
        Nd_m3  = 2.0 / (Q_E * EPSILON_0 * self.epsilon_r * (A_m2 ** 2) * abs(slope))
        Nd_cm3 = Nd_m3 * 1e-6

        return MottSchottkyResult(
            Vfb=Vfb,
            Nd=Nd_cm3,
            sc_type="n-type" if slope > 0 else "p-type",
            slope=slope,
            intercept=intercept,
            R2=R2,
        )

    def tauc(
        self,
        wavelength_nm: "np.ndarray",
        absorbance: "np.ndarray",
        transition: str = "direct",
        film_thickness_cm: float = 1.0,
        fit_fraction: float = 0.3,
    ) -> TaucResult:
        """
        Tauc plot analysis.

        wavelength_nm     : wavelength array (nm)
        absorbance        : UV-Vis absorbance (a.u.)
        transition        : 'direct'   -> (alpha*hv)^2
                            'indirect' -> (alpha*hv)^0.5
        film_thickness_cm : for alpha = A/d  (default 1 cm)
        fit_fraction      : fraction of rising edge used for linear fit
        """
        hv    = 1240.0 / np.asarray(wavelength_nm, dtype=float)
        alpha = np.asarray(absorbance, dtype=float) / film_thickness_cm
        n     = 0.5 if transition == "direct" else 2.0
        y     = (alpha * hv) ** n

        idx = np.argsort(hv)
        hv, y = hv[idx], y[idx]

        y_min, y_max = np.min(y), np.max(y)
        threshold = y_max - fit_fraction * (y_max - y_min)
        mask = y >= threshold
        if mask.sum() < 3:
            mask = np.ones(len(hv), dtype=bool)

        E_fit     = hv[mask]
        alpha_fit = y[mask]
        coeffs    = np.polyfit(E_fit, alpha_fit, 1)
        slope_t, intercept_t = coeffs
        y_pred = np.polyval(coeffs, E_fit)
        ss_res = np.sum((alpha_fit - y_pred) ** 2)
        ss_tot = np.sum((alpha_fit - np.mean(alpha_fit)) ** 2)
        R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        Eg     = -intercept_t / slope_t if abs(slope_t) > 1e-12 else 0.0

        return TaucResult(
            Eg=float(Eg),
            transition=transition,
            R2=float(R2),
            E_fit=E_fit,
            alpha_fit=alpha_fit,
        )

    @staticmethod
    def list_materials() -> list:
        return list(MATERIALS_DB.keys())

    @staticmethod
    def nhe_to_rhe(E_NHE: float, pH: float) -> float:
        return E_NHE - NERNST_SLOPE * pH

    @staticmethod
    def rhe_to_nhe(E_RHE: float, pH: float) -> float:
        return E_RHE + NERNST_SLOPE * pH
