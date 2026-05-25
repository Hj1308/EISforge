"""
Auto-ECSA Calculator for EISforge
Author: Hoda Jafari | May 2026

Automatically selects ECSA method based on catalyst type:
    noble_metal     -> H_upd  (Q_H / 210 uC/cm2)
    alloy           -> CO Stripping (Q_CO / 420 uC/cm2)
    metal_oxide     -> C_dl method (BET-based)
    carbon_material -> C_dl method (40 uF/cm2)
"""

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
Q_REF_HUP  = 210e-6   # C/cm2  — Pt H-upd reference charge density
Q_REF_CO   = 420e-6   # C/cm2  — CO stripping reference charge density
C_SPECIFIC = 40e-6    # F/cm2  — specific capacitance for carbon materials
C_SPECIFIC_OXIDE = 60e-6  # F/cm2  — for metal oxides


# ── Method selector ───────────────────────────────────────────────────────────
ECSA_METHOD = {
    "noble_metal":     "h_upd",
    "alloy":           "co_stripping",
    "metal_oxide":     "cdl",
    "carbon_material": "cdl",
}


# ── Result container ──────────────────────────────────────────────────────────
class ECSAResult:
    def __init__(self, ecsa_cm2, ecsa_m2g, method, catalyst_loading_mg=None, details=None):
        self.ecsa_cm2          = ecsa_cm2          # cm2
        self.ecsa_m2g          = ecsa_m2g          # m2/g  (None if no loading given)
        self.method            = method            # str
        self.catalyst_loading  = catalyst_loading_mg
        self.details           = details or {}

    def summary(self):
        lines = [
            f"ECSA Method    : {self.method}",
            f"ECSA           : {self.ecsa_cm2:.4f} cm²",
        ]
        if self.ecsa_m2g is not None:
            lines.append(f"ECSA           : {self.ecsa_m2g:.2f} m²/g")
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── Core calculator ───────────────────────────────────────────────────────────
class AutoECSA:
    """
    Automatic ECSA calculator — selects method from catalyst_type.

    Parameters
    ----------
    catalyst_type : str
        One of: noble_metal, alloy, metal_oxide, carbon_material
    catalyst_loading_mg : float, optional
        Catalyst loading in mg — enables m2/g calculation
    """

    def __init__(self, catalyst_type: str, catalyst_loading_mg: float = None):
        self.catalyst_type    = catalyst_type.lower()
        self.loading          = catalyst_loading_mg
        self.method           = ECSA_METHOD.get(self.catalyst_type, "cdl")

    # ── Public entry point ────────────────────────────────────────────────────
    def calculate(self, potential: np.ndarray, current: np.ndarray,
                  scan_rate: float = 50.0) -> ECSAResult:
        """
        Calculate ECSA from CV data.

        Parameters
        ----------
        potential : np.ndarray   — potential in V
        current   : np.ndarray   — current in A  (not mA!)
        scan_rate : float        — scan rate in mV/s
        """
        if self.method == "h_upd":
            return self._h_upd(potential, current, scan_rate)
        elif self.method == "co_stripping":
            return self._co_stripping(potential, current, scan_rate)
        elif self.method == "cdl":
            return self._cdl(potential, current, scan_rate)
        else:
            raise ValueError(f"Unknown ECSA method: {self.method}")

    # ── H_upd method (Pt, Pd, noble metals) ──────────────────────────────────
    def _h_upd(self, potential, current, scan_rate):
        """
        Integrate H-adsorption region: 0.05 – 0.40 V vs RHE
        Q_H = integral of |I| dE / scan_rate
        ECSA = Q_H / 210 uC/cm2
        """
        mask = (potential >= 0.05) & (potential <= 0.40)
        if mask.sum() < 5:
            logger.warning("H-upd: fewer than 5 points in 0.05-0.40 V range")

        E_region = potential[mask]
        I_region = np.abs(current[mask])

        # Integrate: Q = integral(I dE) / scan_rate (V/s)
        sr_vs = scan_rate * 1e-3  # mV/s → V/s
        if len(E_region) > 1:
            Q_H = np.trapezoid(I_region, E_region) / sr_vs  # Coulombs
        else:
            Q_H = 0.0

        ecsa_cm2 = Q_H / Q_REF_HUP
        ecsa_m2g = (ecsa_cm2 * 1e-4) / (self.loading * 1e-3) if self.loading else None

        return ECSAResult(
            ecsa_cm2=ecsa_cm2,
            ecsa_m2g=ecsa_m2g,
            method="H_upd (0.05-0.40 V vs RHE)",
            catalyst_loading_mg=self.loading,
            details={"Q_H (C)": f"{Q_H:.6f}", "Q_ref (C/cm2)": "210e-6"}
        )

    # ── CO Stripping method (PtRu, PtSn, Pd alloys) ──────────────────────────
    def _co_stripping(self, potential, current, scan_rate):
        """
        Integrate CO oxidation peak: 0.50 – 1.00 V
        Q_CO = integral(I dE) / scan_rate
        ECSA = Q_CO / 420 uC/cm2
        """
        mask = (potential >= 0.50) & (potential <= 1.00)
        E_region = potential[mask]
        I_region = current[mask]

        # Only positive (oxidation) current
        I_region = np.clip(I_region, 0, None)

        sr_vs = scan_rate * 1e-3
        Q_CO = np.trapezoid(I_region, E_region) / sr_vs if len(E_region) > 1 else 0.0

        ecsa_cm2 = Q_CO / Q_REF_CO
        ecsa_m2g = (ecsa_cm2 * 1e-4) / (self.loading * 1e-3) if self.loading else None

        return ECSAResult(
            ecsa_cm2=ecsa_cm2,
            ecsa_m2g=ecsa_m2g,
            method="CO Stripping (0.50-1.00 V)",
            catalyst_loading_mg=self.loading,
            details={"Q_CO (C)": f"{Q_CO:.6f}", "Q_ref (C/cm2)": "420e-6"}
        )

    # ── C_dl method (carbon materials, metal oxides) ──────────────────────────
    def _cdl(self, potential, current, scan_rate):
        """
        Double-layer capacitance method.
        Uses non-Faradaic region (mid-potential).
        C_dl = (I_anodic - I_cathodic) / (2 * scan_rate)
        ECSA = C_dl / C_specific
        """
        # Find non-Faradaic region: middle 20% of potential window
        E_min, E_max = potential.min(), potential.max()
        E_mid = (E_min + E_max) / 2
        E_window = (E_max - E_min) * 0.1

        mask = (potential >= E_mid - E_window) & (potential <= E_mid + E_window)
        if mask.sum() < 4:
            # fallback: use all data
            mask = np.ones(len(potential), dtype=bool)

        I_region = current[mask]
        I_anodic  = I_region[I_region >= 0].mean() if (I_region >= 0).any() else 0.0
        I_cathodic = np.abs(I_region[I_region < 0].mean()) if (I_region < 0).any() else 0.0

        sr_vs = scan_rate * 1e-3  # V/s
        C_dl = (I_anodic + I_cathodic) / (2 * sr_vs)  # Farads

        c_spec = C_SPECIFIC_OXIDE if self.catalyst_type == "metal_oxide" else C_SPECIFIC
        ecsa_cm2 = C_dl / c_spec
        ecsa_m2g = (ecsa_cm2 * 1e-4) / (self.loading * 1e-3) if self.loading else None

        return ECSAResult(
            ecsa_cm2=ecsa_cm2,
            ecsa_m2g=ecsa_m2g,
            method=f"C_dl (C_specific={c_spec*1e6:.0f} uF/cm2)",
            catalyst_loading_mg=self.loading,
            details={
                "C_dl (F)": f"{C_dl:.6f}",
                "I_anodic (A)": f"{I_anodic:.6f}",
                "I_cathodic (A)": f"{I_cathodic:.6f}",
            }
        )


# ── Convenience function ──────────────────────────────────────────────────────
def calculate_ecsa(potential, current, catalyst_type,
                   scan_rate=50.0, catalyst_loading_mg=None) -> ECSAResult:
    """One-line ECSA calculation."""
    calc = AutoECSA(catalyst_type=catalyst_type,
                    catalyst_loading_mg=catalyst_loading_mg)
    return calc.calculate(potential, current, scan_rate)
