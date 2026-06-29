"""
Koutecky-Levich Analysis — RDE (Rotating Disk Electrode) for AOR.
Author: Hoda Jafari | Updated: June 2026

Separates kinetic current from mass-transport limitation:

    1/j = 1/j_k + 1/(B × ω^0.5)

    B = 0.62 × n × F × D^(2/3) × ν^(-1/6) × C

Where:
    j     : measured current density (mA/cm²)
    j_k   : kinetic current density (intrinsic catalyst activity)
    B     : Levich slope
    ω     : rotation rate (rad/s)
    n     : number of electrons transferred
    F     : Faraday constant (96485 C/mol)
    D     : diffusion coefficient of alcohol (cm²/s)
    ν     : kinematic viscosity of electrolyte (cm²/s)
    C     : bulk concentration of alcohol (mol/cm³)

For alcohol electrooxidation:
    n < 2  → incomplete oxidation — product is mostly aldehyde
    n = 2  → two-electron oxidation — common for primary alcohols
    n = 4  → deeper oxidation — carboxylic acid as product
    n = 6  → complete oxidation to CO2
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ── Physical constants ──────────────────────────────────────────────────────
FARADAY = 96485.0  # C/mol
GAS_CONST = 8.31446  # J/mol/K

# ── Diffusion coefficients (cm²/s at 25°C) ────────────────────────────────
DIFFUSION_COEFF = {
    "methanol": 1.60e-5,         # 0.1M KOH, 25°C [Lamy et al., 2002]
    "ethanol": 1.08e-5,          # 0.1M KOH, 25°C [Sen Gupta et al., 2005]
    "2-propanol": 0.97e-5,       # 0.1M KOH, 25°C [Mangoufis-Giasin, 2021]
    "ethylene_glycol": 0.94e-5,  # 0.1M KOH, 25°C [Various]
    "glycerol": 0.72e-5,         # 0.1M KOH, 25°C [Verma et al., 2022]
}

# ── Kinematic viscosities (cm²/s at 25°C) ──────────────────────────────────
# NOTE: These are KINEMATIC viscosities (cm²/s), not dynamic (Poise)
KINEMATIC_VISCOSITY = {
    "KOH_01M": 0.01000,
    "KOH_1M": 0.01020,
    "NaOH_01M": 0.00920,
    "NaOH_1M": 0.00940,
    "H2SO4_05M": 0.00902,
    "HClO4_01M": 0.00890,
    "default": 0.00893,  # pure water at 25°C
}


# ── Result dataclasses ──────────────────────────────────────────────────────
@dataclass
class KLResult:
    """Results of Koutecky-Levich analysis at one potential."""

    potential_V: float
    j_kinetic: float
    n_electrons: float
    levich_slope: float
    levich_slope_theoretical: float
    r_squared: float
    intercept: float
    rotation_speeds_rpm: list
    j_measured: list
    diffusion_controlled_fraction: float
    interpretation: str
    alcohol: str = "ethanol"
    catalyst_type: str = "noble_metal"

    def summary(self) -> str:
        dc_str = (
            f"{self.diffusion_controlled_fraction:.1%}"
            if np.isfinite(self.diffusion_controlled_fraction)
            else "N/A (kinetic region)"
        )
        lines = [
            "=" * 65,
            f"  Koutecky-Levich Analysis — E = {self.potential_V:.3f} V vs RHE",
            "=" * 65,
            f"  Alcohol              : {self.alcohol}",
            f"  Catalyst type        : {self.catalyst_type}",
            "-" * 65,
            f"  j_kinetic            = {self.j_kinetic:.4f} mA/cm²",
            f"  n electrons          = {self.n_electrons:.2f}",
            f"  K-L slope (exp.)     = {self.levich_slope:.6f} mA·s^0.5/cm²",
            f"  K-L slope (theory)   = {self.levich_slope_theoretical:.6f} mA·s^0.5/cm²",
            f"  R² (K-L fit)         = {self.r_squared:.4f}",
            f"  Mass-transport frac. = {dc_str}",
            "-" * 65,
            f"  Interpretation: {self.interpretation}",
            "=" * 65,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "E (V vs RHE)": f"{self.potential_V:.3f}",
            "j_kinetic (mA/cm²)": f"{self.j_kinetic:.4f}",
            "n electrons": f"{self.n_electrons:.2f}",
            "K-L slope (exp.)": f"{self.levich_slope:.6f}",
            "K-L slope (theory)": f"{self.levich_slope_theoretical:.6f}",
            "R² (K-L fit)": f"{self.r_squared:.4f}",
            "Mass-transport fraction": (
                f"{self.diffusion_controlled_fraction:.1%}"
                if np.isfinite(self.diffusion_controlled_fraction)
                else "N/A"
            ),
        }


@dataclass
class KLFullResult:
    """Complete K-L analysis across multiple potentials."""

    results_per_potential: list
    alcohol: str
    electrolyte: str
    temperature_C: float
    electrode_area_cm2: float

    potentials_V: np.ndarray = field(default_factory=lambda: np.array([]))
    j_kinetic_arr: np.ndarray = field(default_factory=lambda: np.array([]))
    n_electrons_arr: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def mean_n_electrons(self) -> float:
        if len(self.n_electrons_arr) == 0:
            return float("nan")
        valid = self.n_electrons_arr[~np.isnan(self.n_electrons_arr)]
        return float(valid.mean()) if len(valid) > 0 else float("nan")

    @property
    def best_result(self) -> Optional[KLResult]:
        """Return the KLResult with the highest R²."""
        if not self.results_per_potential:
            return None
        return max(self.results_per_potential, key=lambda r: r.r_squared)

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "  Koutecky-Levich Full Analysis",
            f"  Alcohol: {self.alcohol} | Electrolyte: {self.electrolyte}",
            f"  Temperature: {self.temperature_C}°C | Area: {self.electrode_area_cm2} cm²",
            "=" * 65,
            f"  Mean n electrons: {self.mean_n_electrons:.2f}",
            f"  Potentials analyzed: {len(self.results_per_potential)}",
            "-" * 65,
        ]
        for r in self.results_per_potential:
            lines.append(
                f"  E={r.potential_V:.3f}V  j_k={r.j_kinetic:.3f} mA/cm²  "
                f"n={r.n_electrons:.2f}  R²={r.r_squared:.4f}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.results_per_potential])

    def to_markdown_table(self) -> str:
        df = self.to_dataframe()
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        rows = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, sep] + rows)

    def to_latex_table(self) -> str:
        df = self.to_dataframe()
        cols = " & ".join(df.columns.str.replace("_", r"\_")) + r" \\"
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Koutecky-Levich analysis results}",
            r"\begin{tabular}{" + "c" * len(df.columns) + "}",
            r"\hline",
            cols,
            r"\hline",
        ]
        for _, row in df.iterrows():
            lines.append(" & ".join(str(v) for v in row) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


class KLAnalyzer:
    """
    Koutecky-Levich analyzer for RDE measurements.

    Parameters
    ----------
    alcohol : str
        Alcohol type: 'methanol', 'ethanol', '2-propanol',
        'glycerol', 'ethylene_glycol'.
    electrolyte : str
        Electrolyte description: 'KOH', 'NaOH', 'H2SO4', etc.
    concentration_M : float
        Alcohol bulk concentration in mol/L (M). Default: 1.0.
    temperature_C : float
        Temperature in Celsius. Default: 25.
    catalyst_type : str
        Catalyst family for interpretation.
    D_cm2_s : float or None
        Custom diffusion coefficient (cm²/s). If None, uses literature value.
    nu_cm2_s : float or None
        Custom kinematic viscosity (cm²/s). If None, uses literature value.
    """

    def __init__(
        self,
        alcohol: str = "ethanol",
        electrolyte: str = "KOH",
        concentration_M: float = 1.0,
        temperature_C: float = 25.0,
        catalyst_type: str = "noble_metal",
        D_cm2_s: Optional[float] = None,
        nu_cm2_s: Optional[float] = None,
    ) -> None:
        self.alcohol = alcohol
        self.electrolyte = electrolyte
        self.concentration_M = concentration_M
        self.temperature_C = temperature_C
        self.catalyst_type = catalyst_type

        # FIX #5: use 'is not None' so D_cm2_s=0.0 doesn't silently fall through
        if D_cm2_s is not None:
            self.D = D_cm2_s
        else:
            self.D = DIFFUSION_COEFF.get(alcohol, 1.0e-5)

        if nu_cm2_s is not None:
            self.nu = nu_cm2_s
        else:
            # FIX #3: build viscosity key correctly.
            # e.g. concentration_M=1.0 → "KOH_1M", not "KOH_10M"
            # Strategy: try integer representation first, then one-decimal float.
            base = electrolyte.split()[0]
            if concentration_M == int(concentration_M):
                conc_tag = str(int(concentration_M))      # 1.0 → "1"
            else:
                conc_tag = f"{concentration_M:.1f}".replace(".", "")  # 0.1 → "01"
            visc_key = f"{base}_{conc_tag}M"
            self.nu = KINEMATIC_VISCOSITY.get(visc_key, KINEMATIC_VISCOSITY["default"])
            if visc_key not in KINEMATIC_VISCOSITY:
                logger.warning(
                    "Viscosity key '%s' not found in table. "
                    "Using default (pure water, 0.00893 cm²/s).",
                    visc_key,
                )

        self.C = concentration_M * 1e-3  # mol/L → mol/cm³

        logger.info(
            "KLAnalyzer: D=%.2e cm²/s  ν=%.5f cm²/s  C=%.4e mol/cm³",
            self.D, self.nu, self.C,
        )

    # ── Main analysis ──────────────────────────────────────────────────────

    def analyze(
        self,
        rotation_speeds_rpm: list,
        potentials: list,
        currents: list,
        electrode_area: float,
        analysis_potential: Optional[float] = None,
        analysis_potentials: Optional[list] = None,
        current_unit: str = "mA",
    ) -> KLFullResult:
        """
        Run Koutecky-Levich analysis.

        Parameters
        ----------
        rotation_speeds_rpm : list of float
            RDE rotation speeds in rpm. e.g. [400, 900, 1600, 2500].
            Minimum 3 speeds required for a reliable linear fit.
        potentials : list of np.ndarray
            Potential arrays (V vs RHE), one array per rotation speed.
        currents : list of np.ndarray
            Current arrays (in *current_unit*), one array per rotation speed.
        electrode_area : float
            Geometric electrode area in cm².
        analysis_potential : float or None
            Single potential (V) at which to perform the K-L analysis.
        analysis_potentials : list of float or None
            Multiple potentials for potential-dependent K-L analysis.
        current_unit : str
            Input current unit: 'mA', 'A', 'uA', 'nA'.

        Returns
        -------
        KLFullResult
        """
        if len(rotation_speeds_rpm) < 3:
            raise ValueError(
                f"Koutecky-Levich requires at least 3 rotation speeds. "
                f"Got {len(rotation_speeds_rpm)}."
            )
        if len(rotation_speeds_rpm) != len(potentials):
            raise ValueError(
                "rotation_speeds_rpm and potentials must have the same length."
            )

        unit_map = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "nA": 1e-6}
        factor = unit_map.get(current_unit, 1.0)
        j_arrays = [
            np.asarray(cur) * factor / electrode_area
            for cur in currents
        ]

        omega = [rpm * 2 * np.pi / 60.0 for rpm in rotation_speeds_rpm]

        if analysis_potentials is not None:
            E_list = analysis_potentials
        elif analysis_potential is not None:
            E_list = [analysis_potential]
        else:
            e_min = max(p.min() for p in potentials)
            e_max = min(p.max() for p in potentials)
            E_list = list(np.linspace(e_min + 0.05, e_max - 0.05, 8))

        results = []
        for E in E_list:
            try:
                r = self._analyze_at_potential(
                    E, potentials, j_arrays, omega, rotation_speeds_rpm,
                )
                if r is not None:
                    results.append(r)
            except Exception as exc:
                logger.warning("K-L analysis at E=%.3f V failed: %s", E, exc)

        if not results:
            raise ValueError("K-L analysis failed at all requested potentials.")

        pot_arr = np.array([r.potential_V for r in results])
        jk_arr = np.array([r.j_kinetic for r in results])
        n_arr = np.array([r.n_electrons for r in results])

        return KLFullResult(
            results_per_potential=results,
            alcohol=self.alcohol,
            electrolyte=self.electrolyte,
            temperature_C=self.temperature_C,
            electrode_area_cm2=electrode_area,
            potentials_V=pot_arr,
            j_kinetic_arr=jk_arr,
            n_electrons_arr=n_arr,
        )

    def _analyze_at_potential(
        self,
        E: float,
        potentials: list,
        j_arrays: list,
        omega: list,
        rpm_list: list,
    ) -> Optional[KLResult]:
        """Run K-L analysis at one potential *E*."""

        j_at_E = []
        for pot, j_arr in zip(potentials, j_arrays):
            order = np.argsort(pot)
            pot_s, j_s = pot[order], j_arr[order]
            j_at_E.append(float(np.interp(E, pot_s, j_s)))

        # FIX #2: warn if all currents are negative (likely sign convention issue)
        if all(j < 0 for j in j_at_E):
            warnings.warn(
                f"All currents at E={E:.3f} V are negative. "
                "K-L analysis expects positive anodic (oxidation) currents. "
                "Check your current sign convention.",
                UserWarning,
                stacklevel=3,
            )

        # Use abs(j) so a flipped sign convention doesn't invert the K-L slope
        inv_j = [1.0 / abs(j) if abs(j) > 1e-10 else np.nan for j in j_at_E]
        inv_sqw = [1.0 / np.sqrt(w) for w in omega]

        valid = [
            (x, y) for x, y in zip(inv_sqw, inv_j)
            if not np.isnan(x) and not np.isnan(y)
        ]
        if len(valid) < 3:
            logger.warning(
                "Only %d valid points at E=%.3f V — skipping.", len(valid), E
            )
            return None

        x_arr = np.array([v[0] for v in valid])
        y_arr = np.array([v[1] for v in valid])

        try:
            slope, intercept, r_val, _, _ = linregress(x_arr, y_arr)
        except Exception:
            return None

        if abs(slope) < 1e-12:
            return None

        j_kinetic = 1.0 / intercept if abs(intercept) > 1e-12 else float("nan")

        B_experimental = 1.0 / slope
        base_B = (
            0.62
            * FARADAY
            * (self.D ** (2 / 3))
            * (self.nu ** (-1 / 6))
            * self.C
        )
        base_B_mA = base_B * 1000.0

        n_electrons = B_experimental / base_B_mA if base_B_mA > 1e-12 else float("nan")
        B_theoretical_n2 = base_B_mA * 2.0

        # Diffusion-controlled fraction
        try:
            sqrt_w_arr = np.sqrt(np.array(omega))
            j_arr_chk = np.array(j_at_E)
            _, _, r_jw, _, _ = linregress(sqrt_w_arr, j_arr_chk)
            if r_jw ** 2 > 0.85:
                j_diff_max = B_experimental * np.sqrt(max(omega))
                j_total_max = max(abs(j) for j in j_at_E)
                diff_frac = (
                    abs(j_diff_max / j_total_max)
                    if j_total_max > 1e-10
                    else 0.5
                )
                diff_frac = min(1.0, max(0.0, diff_frac))
            else:
                diff_frac = float("nan")
        except Exception:
            diff_frac = float("nan")

        interpretation = self._interpret(n_electrons, j_kinetic, r_val ** 2)

        return KLResult(
            potential_V=E,
            j_kinetic=j_kinetic,
            n_electrons=n_electrons,
            levich_slope=B_experimental,
            levich_slope_theoretical=B_theoretical_n2,
            r_squared=r_val ** 2,
            intercept=intercept,
            rotation_speeds_rpm=rpm_list,
            j_measured=j_at_E,
            diffusion_controlled_fraction=diff_frac,
            interpretation=interpretation,
            alcohol=self.alcohol,
            catalyst_type=self.catalyst_type,
        )

    # ── Levich-only analysis ──────────────────────────────────────────────

    def levich_plot_data(
        self,
        rotation_speeds_rpm: list,
        j_limiting: list,
    ) -> dict:
        """
        Simple Levich plot: j_L vs √ω (mass-transport only, no intercept).

        Use this when you want the limiting current only, not j_kinetic.

        Parameters
        ----------
        rotation_speeds_rpm : list of float
        j_limiting : list of float
            Limiting current density at each speed (mA/cm²).

        Returns
        -------
        dict with slope, n_electrons, D_estimate, and fit arrays.
        """
        omega = np.array([rpm * 2 * np.pi / 60.0 for rpm in rotation_speeds_rpm])
        sqrt_w = np.sqrt(omega)
        j_arr = np.array(j_limiting)

        try:
            slope, intercept, r_val, _, _ = linregress(sqrt_w, j_arr)
        except Exception as exc:
            raise ValueError(f"Levich plot fit failed: {exc}") from exc

        base_B_mA = (
            0.62
            * FARADAY
            * (self.D ** (2 / 3))
            * (self.nu ** (-1 / 6))
            * self.C
            * 1000.0
        )
        n_from_levich = slope / base_B_mA if base_B_mA > 1e-12 else float("nan")

        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_val ** 2,
            "n_electrons": n_from_levich,
            "sqrt_omega": sqrt_w,
            "j_limiting": j_arr,
            "fit_line": slope * sqrt_w + intercept,
        }

    # ── Interpretation ──────────────────────────────────────────────────────

    def _interpret(self, n: float, j_k: float, r2: float) -> str:
        parts = []

        if r2 < 0.95:
            parts.append(
                f"Poor K-L linearity (R²={r2:.3f}) — check data quality; "
                "ensure steady-state LSV was recorded at each rotation speed."
            )

        if np.isnan(n):
            parts.append("n electrons could not be determined.")
        elif n < 1.0:
            # FIX #4: emit logger warning for physically impossible n
            logger.warning(
                "Unusually low n=%.2f — verify D, ν, and concentration.", n
            )
            parts.append(
                f"n = {n:.2f} — unusually low. Verify diffusion coefficient "
                "and alcohol concentration."
            )
        elif n < 2.5:
            parts.append(
                f"n = {n:.2f} — two-electron process. "
                "Product: aldehyde or ketone (partial oxidation)."
            )
        elif n < 3.5:
            parts.append(
                f"n = {n:.2f} — three-electron process. "
                "Mixed partial and deeper oxidation pathway."
            )
        elif n < 4.5:
            parts.append(
                f"n = {n:.2f} — four-electron process. "
                "Product: carboxylic acid."
            )
        elif n < 6.5:
            parts.append(
                f"n = {n:.2f} — near-complete oxidation. "
                "Product: CO₂ or deep oxidation pathway."
            )
        else:
            # FIX #4: emit logger warning for physically impossible n
            logger.warning(
                "Unusually high n=%.2f — check experimental conditions.", n
            )
            parts.append(
                f"n = {n:.2f} — unusually high. Check experimental conditions."
            )

        if self.catalyst_type in ("metal_free", "carbon_material"):
            parts.append(
                "Metal-free/carbon catalyst: n < 2 is common — "
                "reaction proceeds via surface defects without d-band facilitation."
            )

        if not np.isnan(j_k):
            if j_k < 0.1:
                parts.append(f"j_kinetic = {j_k:.4f} mA/cm² — low intrinsic activity.")
            elif j_k < 1.0:
                parts.append(f"j_kinetic = {j_k:.4f} mA/cm² — moderate activity.")
            else:
                parts.append(f"j_kinetic = {j_k:.4f} mA/cm² — good intrinsic activity.")

        return " | ".join(parts)

    # ── Convenience: single-potential quick analysis ────────────────────────

    def quick_analyze(
        self,
        rotation_speeds_rpm: list,
        j_at_potential: list,
    ) -> KLResult:
        """
        Quick K-L analysis when j values at one potential are already known.

        Parameters
        ----------
        rotation_speeds_rpm : list  e.g. [400, 900, 1600, 2500]
        j_at_potential : list       j (mA/cm²) at each rotation speed

        Returns
        -------
        KLResult at a single potential (potential_V = 0.0 placeholder).

        Notes
        -----
        FIX #1: requires ≥ 3 rotation speeds (previously accepted ≥ 2,
        which produced a perfect R²=1 fit with no statistical meaning).
        """
        if len(rotation_speeds_rpm) != len(j_at_potential):
            raise ValueError(
                "rotation_speeds_rpm and j_at_potential must have equal length."
            )
        # FIX #1: enforce minimum 3 speeds — same as analyze()
        if len(rotation_speeds_rpm) < 3:
            raise ValueError(
                f"Koutecky-Levich requires at least 3 rotation speeds. "
                f"Got {len(rotation_speeds_rpm)}."
            )

        omega = [rpm * 2 * np.pi / 60.0 for rpm in rotation_speeds_rpm]
        # FIX #2: use abs(j) to handle sign convention safely
        inv_j = [1.0 / abs(j) if abs(j) > 1e-10 else np.nan for j in j_at_potential]
        inv_sqw = [1.0 / np.sqrt(w) for w in omega]

        valid = [
            (x, y) for x, y in zip(inv_sqw, inv_j)
            if not np.isnan(x) and not np.isnan(y)
        ]
        if len(valid) < 3:
            raise ValueError(
                f"Not enough valid data points ({len(valid)}) for K-L fit."
            )

        x_arr = np.array([v[0] for v in valid])
        y_arr = np.array([v[1] for v in valid])

        slope, intercept, r_val, _, _ = linregress(x_arr, y_arr)

        j_kinetic = 1.0 / intercept if abs(intercept) > 1e-12 else float("nan")
        B_exp = 1.0 / slope if abs(slope) > 1e-12 else float("nan")
        base_B_mA = (
            0.62
            * FARADAY
            * (self.D ** (2 / 3))
            * (self.nu ** (-1 / 6))
            * self.C
            * 1000.0
        )
        n_electrons = B_exp / base_B_mA if base_B_mA > 1e-12 else float("nan")
        B_th_n2 = base_B_mA * 2.0

        return KLResult(
            potential_V=0.0,
            j_kinetic=j_kinetic,
            n_electrons=n_electrons,
            levich_slope=B_exp,
            levich_slope_theoretical=B_th_n2,
            r_squared=r_val ** 2,
            intercept=intercept,
            rotation_speeds_rpm=list(rotation_speeds_rpm),
            j_measured=list(j_at_potential),
            diffusion_controlled_fraction=float("nan"),
            interpretation=self._interpret(n_electrons, j_kinetic, r_val ** 2),
            alcohol=self.alcohol,
            catalyst_type=self.catalyst_type,
        )
