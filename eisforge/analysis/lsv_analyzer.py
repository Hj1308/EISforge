"""
LSV Analyzer — Automatic Linear Sweep Voltammetry Analysis for AOR.
Author: Hoda Jafari | May 2026

iR Compensation:
    E_corrected = E_measured - I(A) × R_s(Ω)
    Tafel slopes and E_onset are computed on iR-corrected potentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

FARADAY   = 96485.33    # C/mol
GAS_CONST = 8.31446     # J/mol/K


@dataclass
class LSVAnalysisResult:
    """Results of complete LSV analysis."""

    # E_onset
    e_onset: float
    e_onset_method: str

    # Tafel
    tafel_slope: float          # mV/dec
    tafel_slope_std: float
    exchange_current_density: float   # j0 (mA/cm²)
    tafel_r_squared: float
    tafel_region: tuple

    # Overpotential
    overpotential_10: float     # V
    overpotential_50: float
    overpotential_100: float

    # Activity
    j_at_onset: float
    j_limiting: float
    e_half_wave: float
    mass_activity: float        # mA/mg
    specific_activity: float    # mA/cm²_ECSA

    # iR compensation
    ir_compensated: bool = False
    r_s_used: float = 0.0

    # Measurement parameters
    scan_rate: float = 5.0
    electrode_area: float = 1.0
    ecsa: float = 0.0
    catalyst_loading: float = 0.0
    electrolyte: str = "acidic"

    mechanism_interpretation: str = ""
    performance_rating: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "  LSV Analysis Results — EISForge",
            "=" * 65,
        ]
        if self.ir_compensated:
            lines.append(f"  iR Compensation     : APPLIED (R_s = {self.r_s_used:.3f} Ω)")
        else:
            lines.append(f"  iR Compensation     : NOT applied")
        lines += [
            f"  E_onset             = {self.e_onset:.4f} V  ({self.e_onset_method})",
            "-" * 65,
            f"  Tafel slope         = {self.tafel_slope:.1f} ± {self.tafel_slope_std:.1f} mV/dec",
            f"  j0                  = {self.exchange_current_density:.4e} mA/cm²",
            f"  R² (Tafel fit)      = {self.tafel_r_squared:.4f}",
            "-" * 65,
            f"  η @ 10 mA/cm²       = {self.overpotential_10*1000:.1f} mV",
            f"  η @ 50 mA/cm²       = {self.overpotential_50*1000:.1f} mV",
            f"  η @ 100 mA/cm²      = {self.overpotential_100*1000:.1f} mV",
            "-" * 65,
            f"  Limiting current    = {self.j_limiting:.3f} mA/cm²",
            f"  Half-wave potential = {self.e_half_wave:.4f} V",
        ]
        if self.catalyst_loading > 0:
            lines.append(f"  Mass activity       = {self.mass_activity:.3f} mA/mg_cat")
        if self.ecsa > 0:
            lines.append(f"  Specific activity   = {self.specific_activity:.4f} mA/cm²_Pt")
        lines += [
            "-" * 65,
            f"  Mechanism: {self.mechanism_interpretation}",
            f"  Rating:    {self.performance_rating}",
            "=" * 65,
        ]
        return "\n".join(lines)


class LSVAnalyzer:
    """
    Automatic LSV analyzer for AOR.

    Parameters
    ----------
    scan_rate : float
        Scan rate in mV/s (LSV typically 1-10 mV/s; default: 5).
    electrode_area : float
        Geometric electrode area in cm² (default: 1.0).
    ecsa : float
        Electrochemically active surface area in cm²_metal (default: 0).
    catalyst_loading : float
        Catalyst loading in mg/cm² (default: 0).
    electrolyte : str
        'acidic' or 'alkaline'.
    e_ref_vs_rhe : float
        Potential offset to convert reference electrode to RHE (V).
    tafel_current_range : tuple
        (j_min, j_max) in mA/cm² for Tafel fit (kinetic control region).
    current_unit : str
        Input current unit: 'mA', 'A', 'uA', 'nA'.
    """

    _UNIT_TO_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "μA": 1e-3, "nA": 1e-6}

    def __init__(
        self,
        scan_rate: float = 5.0,
        electrode_area: float = 1.0,
        ecsa: float = 0.0,
        catalyst_loading: float = 0.0,
        electrolyte: str = "acidic",
        e_ref_vs_rhe: float = 0.0,
        tafel_current_range: tuple = (0.1, 2.0),
        current_unit: str = "mA",
    ) -> None:
        self.scan_rate           = scan_rate
        self.electrode_area      = max(electrode_area, 1e-10)
        self.ecsa                = ecsa
        self.catalyst_loading    = catalyst_loading
        self.electrolyte         = electrolyte
        self.e_ref_vs_rhe        = e_ref_vs_rhe
        self.tafel_current_range = tafel_current_range
        self.current_unit        = current_unit
        self._unit_factor        = self._UNIT_TO_MA.get(current_unit, 1.0)

    # ── iR Compensation ───────────────────────────────────────────────────────

    @staticmethod
    def apply_ir_compensation(
        potential: np.ndarray,
        current_ma: np.ndarray,
        r_s_ohms: float,
    ) -> np.ndarray:
        """
        Apply iR compensation to potential.

        Formula:
            E_corrected = E_measured - I(A) × R_s(Ω)

        Parameters
        ----------
        potential : np.ndarray   Measured potential (V).
        current_ma : np.ndarray  Current in mA (auto-converted to A internally).
        r_s_ohms : float         Solution resistance from EIS (Ω).

        Returns
        -------
        np.ndarray   iR-corrected potential (V).
        """
        if r_s_ohms <= 0:
            return potential.copy()
        current_a = current_ma * 1e-3          # mA → A
        ir_drop   = current_a * r_s_ohms       # V
        return potential - ir_drop

    # ── Main analyze method ───────────────────────────────────────────────────

    def analyze(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        r_s_ohms: float = 0.0,
    ) -> LSVAnalysisResult:
        """
        Perform complete LSV analysis.

        Parameters
        ----------
        potential : np.ndarray   Potential (V) — forward scan only.
        current   : np.ndarray   Current (in units of current_unit, default mA).
        r_s_ohms  : float        R_s for iR compensation (Ω). 0 = skip.

        Returns
        -------
        LSVAnalysisResult
        """
        potential = np.asarray(potential, dtype=float)
        current   = np.asarray(current,   dtype=float)

        if len(potential) < 20:
            raise ValueError(f"Insufficient data: {len(potential)} points.")

        # Convert current to mA
        current_ma = current * self._unit_factor

        # Reference electrode → RHE
        potential = potential + self.e_ref_vs_rhe

        # Apply iR compensation
        ir_compensated = r_s_ohms > 0
        if ir_compensated:
            potential = self.apply_ir_compensation(potential, current_ma, r_s_ohms)
            logger.info(f"iR compensation applied: R_s = {r_s_ohms:.3f} Ω")

        # Smooth
        current_ma = self._smooth(current_ma)

        # Ensure ascending potential
        if potential[-1] < potential[0]:
            potential  = potential[::-1]
            current_ma = current_ma[::-1]

        # Current density
        j = current_ma / self.electrode_area    # mA/cm²

        # E_onset
        e_onset, onset_method = self._detect_onset(potential, j)

        # Tafel analysis
        tafel = self._tafel_analysis(potential, j, e_onset)

        # Overpotential
        eta_10  = self._overpotential_at_j(potential, j, 10.0,  e_onset)
        eta_50  = self._overpotential_at_j(potential, j, 50.0,  e_onset)
        eta_100 = self._overpotential_at_j(potential, j, 100.0, e_onset)

        # Limiting current and half-wave
        j_lim  = float(np.max(j))
        e_half = self._half_wave_potential(potential, j, j_lim)

        # Activity
        j_at_onset = float(np.interp(e_onset, potential, j))
        mass_act   = j_at_onset / self.catalyst_loading if self.catalyst_loading > 0 else 0.0
        spec_act   = j_at_onset * self.electrode_area / self.ecsa if self.ecsa > 0 else 0.0

        return LSVAnalysisResult(
            e_onset=e_onset,
            e_onset_method=onset_method + (" (iR-corrected)" if ir_compensated else ""),
            tafel_slope=tafel["slope"],
            tafel_slope_std=tafel["slope_std"],
            exchange_current_density=tafel["j0"],
            tafel_r_squared=tafel["r2"],
            tafel_region=tafel["region"],
            overpotential_10=eta_10,
            overpotential_50=eta_50,
            overpotential_100=eta_100,
            j_at_onset=j_at_onset,
            j_limiting=j_lim,
            e_half_wave=e_half,
            mass_activity=mass_act,
            specific_activity=spec_act,
            ir_compensated=ir_compensated,
            r_s_used=r_s_ohms,
            scan_rate=self.scan_rate,
            electrode_area=self.electrode_area,
            ecsa=self.ecsa,
            catalyst_loading=self.catalyst_loading,
            electrolyte=self.electrolyte,
            mechanism_interpretation=self._interpret_tafel(tafel["slope"]),
            performance_rating=self._rate_performance(e_onset, tafel["slope"], eta_10),
        )

    # ── E_onset ───────────────────────────────────────────────────────────────

    def _detect_onset(self, potential, j) -> tuple[float, str]:
        n       = len(potential)
        bl_end  = max(int(n * 0.15), 5)
        j_max   = float(np.max(j))
        baseline = float(np.mean(j[:bl_end]))
        m_bl, b_bl = np.polyfit(potential[:bl_end], j[:bl_end], 1)

        threshold = baseline + 0.05 * j_max
        above = np.where(j > threshold)[0]
        if len(above) == 0:
            return float(potential[n // 2]), "threshold"

        region_s = int(above[0])
        region_e = min(region_s + int(n * 0.15), n - 1)
        if region_e <= region_s + 2:
            return float(potential[region_s]), "threshold"

        try:
            m_rise, b_rise = np.polyfit(potential[region_s:region_e], j[region_s:region_e], 1)
            denom = m_rise - m_bl
            if abs(denom) < 1e-10:
                return float(potential[region_s]), "threshold"
            onset = float(np.clip((b_bl - b_rise) / denom, potential.min(), potential.max()))
            return onset, "tangent (LSV)"
        except Exception:
            return float(potential[region_s]), "threshold"

    # ── Tafel analysis ────────────────────────────────────────────────────────

    def _tafel_analysis(self, potential, j, e_onset) -> dict:
        j_min, j_max_tafel = self.tafel_current_range
        mask = (j >= j_min) & (j <= j_max_tafel) & (j > 0) & (potential >= e_onset - 0.05)
        if np.sum(mask) < 5:
            mask = (j >= j_min * 0.1) & (j <= j_max_tafel * 5) & (j > 0)

        if np.sum(mask) < 3:
            return {"slope": float("nan"), "slope_std": float("nan"),
                    "j0": float("nan"), "r2": 0.0, "region": (e_onset, e_onset)}

        E_tafel = potential[mask]
        log_j   = np.log10(j[mask])

        try:
            slope, intercept, r_val, _, slope_std = linregress(log_j, E_tafel)
            tafel_mv = slope * 1000.0
            try:
                j0 = float(10 ** ((e_onset - intercept) / slope))
            except Exception:
                j0 = float("nan")
            return {"slope": tafel_mv, "slope_std": slope_std * 1000.0,
                    "j0": j0, "r2": r_val**2, "region": (float(E_tafel[0]), float(E_tafel[-1]))}
        except Exception:
            return {"slope": float("nan"), "slope_std": float("nan"),
                    "j0": float("nan"), "r2": 0.0, "region": (e_onset, e_onset)}

    # ── Overpotential ─────────────────────────────────────────────────────────

    @staticmethod
    def _overpotential_at_j(potential, j, j_target, e_onset) -> float:
        if j_target > np.max(j):
            return float("nan")
        return float(np.interp(j_target, j, potential)) - e_onset

    @staticmethod
    def _half_wave_potential(potential, j, j_lim) -> float:
        j_half = j_lim / 2.0
        if j_half > np.max(j) or j_half < np.min(j):
            return float("nan")
        return float(np.interp(j_half, j, potential))

    # ── Interpretation ────────────────────────────────────────────────────────

    @staticmethod
    def _interpret_tafel(slope_mv: float) -> str:
        if np.isnan(slope_mv): return "Tafel slope could not be determined"
        abs_s = abs(slope_mv)
        if abs_s < 40:   return f"Tafel = {slope_mv:.1f} mV/dec — Chemical step rate-limiting"
        elif abs_s < 75: return f"Tafel = {slope_mv:.1f} mV/dec — Langmuir adsorption limiting (PtRu, PdAu typical)"
        elif abs_s < 120: return f"Tafel = {slope_mv:.1f} mV/dec — First electron transfer limiting (Volmer/Butler-Volmer)"
        else: return f"Tafel = {slope_mv:.1f} mV/dec — High value: CO poisoning or diffusion limiting"

    def _rate_performance(self, e_onset, tafel_slope, eta_10) -> str:
        score = 0
        lo, hi = (0.4, 0.6) if self.electrolyte == "acidic" else (0.2, 0.4)
        if e_onset < lo:   score += 3
        elif e_onset < hi: score += 1
        if not np.isnan(tafel_slope):
            if abs(tafel_slope) < 60:   score += 3
            elif abs(tafel_slope) < 90:  score += 2
            elif abs(tafel_slope) < 120: score += 1
        if not np.isnan(eta_10):
            if eta_10 < 0.1:   score += 3
            elif eta_10 < 0.2: score += 2
            elif eta_10 < 0.3: score += 1
        if score >= 8:   return "Excellent — publishable in high-IF journals"
        elif score >= 5: return "Good — suitable catalyst"
        elif score >= 3: return "Moderate — optimization needed"
        else:            return "Poor — consider different catalyst composition"

    # ── Smoothing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(current: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter
            w = min(11, len(current) // 3)
            if w % 2 == 0: w += 1
            if w < 5: return current.copy()
            return savgol_filter(current, window_length=w, polyorder=3)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter1d
                return gaussian_filter1d(current, sigma=2)
            except Exception:
                return current.copy()

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        for enc in ["latin-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(filepath, comment="#", encoding=enc, sep=None, engine="python")
                c = df.columns.tolist()
                return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read: {filepath}")
