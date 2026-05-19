"""
LSV Analyzer — Automatic Linear Sweep Voltammetry Analysis for AOR.
Author: Hoda Jafari | May 2026

Supports all catalyst families:
    - noble_metal  : Pt, Pd, Au, Rh
    - alloy        : PtRu, PtSn, PdAu, PtCu
    - metal_oxide  : NiO, Co3O4, Co2NiO4, MnO2
    - metal_free   : B4C, N-doped Carbon, CNT, rGO

Electrolyte specifics:
    - Acid type    : H2SO4, HClO4, HCl, HNO3
    - Base type    : KOH, NaOH, Na2CO3, NH3
    - Concentration: float in mol/L (M)

iR Compensation:
    E_corrected = E_measured - I(A) × R_s(Ω)
    Tafel slopes and E_onset are computed on iR-corrected potentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from eisforge.analysis.cv_analyzer import (
    ElectrolyteInfo,
    CATALYST_NOBLE_METAL,
    CATALYST_ALLOY,
    CATALYST_METAL_OXIDE,
    CATALYST_METAL_FREE,
    ACID_H2SO4, ACID_HClO4, ACID_HCl, ACID_HNO3,
    BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3,
)

logger = logging.getLogger(__name__)

FARADAY   = 96485.33    # C/mol
GAS_CONST = 8.31446     # J/mol/K


@dataclass
class LSVAnalysisResult:
    """Results of complete LSV analysis."""

    # Catalyst and electrolyte info
    catalyst_type : str = CATALYST_NOBLE_METAL
    electrolyte   : ElectrolyteInfo = field(default_factory=ElectrolyteInfo)

    # E_onset
    e_onset        : float = 0.0
    e_onset_method : str   = ""

    # Tafel
    tafel_slope           : float = 0.0   # mV/dec
    tafel_slope_std       : float = 0.0
    exchange_current_density : float = 0.0  # j0 (mA/cm²)
    tafel_r_squared       : float = 0.0
    tafel_region          : tuple = (0.0, 0.0)

    # Overpotential
    overpotential_10  : float = 0.0   # V
    overpotential_50  : float = 0.0
    overpotential_100 : float = 0.0

    # Activity
    j_at_onset      : float = 0.0
    j_limiting      : float = 0.0
    e_half_wave     : float = 0.0
    mass_activity   : float = 0.0   # mA/mg
    specific_activity: float = 0.0  # mA/cm²_ECSA or mA/cm²_BET

    # iR compensation
    ir_compensated : bool  = False
    r_s_used       : float = 0.0

    # Measurement parameters
    scan_rate        : float = 5.0
    electrode_area   : float = 1.0
    ecsa             : float = 0.0
    catalyst_loading : float = 0.0

    mechanism_interpretation : str = ""
    performance_rating       : str = ""

    def summary(self) -> str:
        el            = self.electrolyte
        is_metal_free = self.catalyst_type == CATALYST_METAL_FREE
        is_oxide      = self.catalyst_type == CATALYST_METAL_OXIDE
        ecsa_label    = "cm²_BET" if is_metal_free else "cm²_metal"

        lines = [
            "=" * 68,
            "  LSV Analysis Results — EISForge",
            "=" * 68,
            f"  Catalyst type       : {self._catalyst_label()}",
            f"  Electrolyte         : {el.label()}  ({el.media})",
            "-" * 68,
        ]
        if self.ir_compensated:
            lines.append(f"  iR Compensation     : APPLIED (R_s = {self.r_s_used:.3f} Ω)")
        else:
            lines.append(f"  iR Compensation     : not applied")

        lines += [
            f"  E_onset             = {self.e_onset:.4f} V  ({self.e_onset_method})",
            "-" * 68,
            f"  Tafel slope         = {self.tafel_slope:.1f} ± {self.tafel_slope_std:.1f} mV/dec",
            f"  j0                  = {self.exchange_current_density:.4e} mA/cm²",
            f"  R² (Tafel fit)      = {self.tafel_r_squared:.4f}",
        ]

        if is_metal_free:
            lines.append(
                "  NOTE: Tafel > 120 mV/dec is NORMAL for metal-free catalysts"
            )

        lines += [
            "-" * 68,
            f"  η @ 10 mA/cm²       = {self.overpotential_10*1000:.1f} mV"
            if not np.isnan(self.overpotential_10) else "  η @ 10 mA/cm²       = not reached",
            f"  η @ 50 mA/cm²       = {self.overpotential_50*1000:.1f} mV"
            if not np.isnan(self.overpotential_50) else "  η @ 50 mA/cm²       = not reached",
            f"  η @ 100 mA/cm²      = {self.overpotential_100*1000:.1f} mV"
            if not np.isnan(self.overpotential_100) else "  η @ 100 mA/cm²      = not reached",
            "-" * 68,
            f"  Limiting current    = {self.j_limiting:.3f} mA/cm²",
            f"  Half-wave potential = {self.e_half_wave:.4f} V"
            if not np.isnan(self.e_half_wave) else "  Half-wave potential = not determined",
        ]

        if self.catalyst_loading > 0:
            lines.append(f"  Mass activity       = {self.mass_activity:.3f} mA/mg_cat")
        if self.ecsa > 0:
            lines.append(f"  Specific activity   = {self.specific_activity:.4f} mA/{ecsa_label}")

        lines += [
            "-" * 68,
            f"  Mechanism : {self.mechanism_interpretation}",
            f"  Rating    : {self.performance_rating}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def _catalyst_label(self) -> str:
        labels = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt / Pd / Au / Rh)",
            CATALYST_ALLOY       : "Alloy (PtRu / PtSn / PdAu / PtCu ...)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO / Co3O4 / MnO2 ...)",
            CATALYST_METAL_FREE  : "Metal-Free (B4C / N-doped C / CNT ...)",
        }
        return labels.get(self.catalyst_type, self.catalyst_type)


class LSVAnalyzer:
    """
    Automatic LSV analyzer for AOR — supports all catalyst families.

    Parameters
    ----------
    scan_rate : float
        Scan rate in mV/s (LSV typically 1–10 mV/s; default: 5).
    electrode_area : float
        Geometric electrode area in cm² (default: 1.0).
    ecsa : float
        ECSA in cm²_metal (noble/alloy) or cm²_BET (metal-free).
    catalyst_loading : float
        Catalyst loading in mg/cm².
    electrolyte : str or ElectrolyteInfo
        Electrolyte. Can be:
          - ElectrolyteInfo object (recommended)
          - simple string: 'acidic', 'alkaline', 'KOH', 'H2SO4'
    electrolyte_concentration : float
        Concentration in mol/L — used when electrolyte is a string.
    catalyst_type : str
        One of: 'noble_metal', 'alloy', 'metal_oxide', 'metal_free'
    e_ref_vs_rhe : float
        Reference electrode offset to RHE (V).
    tafel_current_range : tuple
        (j_min, j_max) mA/cm² for Tafel fit.
    current_unit : str
        Input current unit: 'mA', 'A', 'uA', 'nA'.
    """

    _UNIT_TO_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "μA": 1e-3, "nA": 1e-6}

    # Tafel current ranges per catalyst type (mA/cm²)
    _TAFEL_RANGES = {
        CATALYST_NOBLE_METAL : (0.1,  2.0),
        CATALYST_ALLOY       : (0.1,  2.0),
        CATALYST_METAL_OXIDE : (0.05, 1.0),   # lower currents, higher onset
        CATALYST_METAL_FREE  : (0.02, 1.0),   # gentle slope, lower currents
    }

    def __init__(
        self,
        scan_rate                : float = 5.0,
        electrode_area           : float = 1.0,
        ecsa                     : float = 0.0,
        catalyst_loading         : float = 0.0,
        electrolyte                      = "acidic",
        electrolyte_concentration: float = 0.5,
        catalyst_type            : str   = CATALYST_NOBLE_METAL,
        e_ref_vs_rhe             : float = 0.0,
        tafel_current_range      : tuple = None,
        current_unit             : str   = "mA",
    ) -> None:
        self.scan_rate        = scan_rate
        self.electrode_area   = max(electrode_area, 1e-10)
        self.ecsa             = ecsa
        self.catalyst_loading = catalyst_loading
        self.catalyst_type    = catalyst_type
        self.e_ref_vs_rhe     = e_ref_vs_rhe
        self.current_unit     = current_unit
        self._unit_factor     = self._UNIT_TO_MA.get(current_unit, 1.0)

        # Electrolyte
        if isinstance(electrolyte, ElectrolyteInfo):
            self.electrolyte_info = electrolyte
        else:
            self.electrolyte_info = ElectrolyteInfo.from_string(
                str(electrolyte), concentration=electrolyte_concentration
            )

        # Tafel range: use provided or default per catalyst type
        if tafel_current_range is not None:
            self.tafel_current_range = tafel_current_range
        else:
            self.tafel_current_range = self._TAFEL_RANGES.get(
                catalyst_type, (0.1, 2.0)
            )

    # ── iR Compensation ───────────────────────────────────────────────────────

    @staticmethod
    def apply_ir_compensation(
        potential  : np.ndarray,
        current_ma : np.ndarray,
        r_s_ohms   : float,
    ) -> np.ndarray:
        """E_corrected = E_measured - I(A) × R_s(Ω)"""
        if r_s_ohms <= 0:
            return potential.copy()
        current_a = current_ma * 1e-3
        return potential - current_a * r_s_ohms

    # ── Main analyze ──────────────────────────────────────────────────────────

    def analyze(
        self,
        potential : np.ndarray,
        current   : np.ndarray,
        r_s_ohms  : float = 0.0,
    ) -> LSVAnalysisResult:
        """
        Full LSV analysis — adapts to catalyst_type and electrolyte.

        Parameters
        ----------
        potential : np.ndarray   Potential (V) — forward scan only.
        current   : np.ndarray   Current (in current_unit, default mA).
        r_s_ohms  : float        R_s for iR compensation (Ω). 0 = skip.
        """
        potential = np.asarray(potential, dtype=float)
        current   = np.asarray(current,   dtype=float)

        if len(potential) < 20:
            raise ValueError(f"Insufficient data: {len(potential)} points.")

        current_ma = current * self._unit_factor
        potential  = potential + self.e_ref_vs_rhe

        ir_compensated = r_s_ohms > 0
        if ir_compensated:
            potential = self.apply_ir_compensation(potential, current_ma, r_s_ohms)
            logger.info(f"iR compensation applied: R_s = {r_s_ohms:.3f} Ω")

        current_ma = self._smooth(current_ma)

        if potential[-1] < potential[0]:
            potential  = potential[::-1]
            current_ma = current_ma[::-1]

        j = current_ma / self.electrode_area    # mA/cm²

        e_onset, onset_method = self._detect_onset(potential, j)
        tafel  = self._tafel_analysis(potential, j, e_onset)
        eta_10  = self._overpotential_at_j(potential, j, 10.0,  e_onset)
        eta_50  = self._overpotential_at_j(potential, j, 50.0,  e_onset)
        eta_100 = self._overpotential_at_j(potential, j, 100.0, e_onset)
        j_lim   = float(np.max(j))
        e_half  = self._half_wave_potential(potential, j, j_lim)
        j_at_onset = float(np.interp(e_onset, potential, j))

        mass_act = (
            j_at_onset / self.catalyst_loading
            if self.catalyst_loading > 0 else 0.0
        )
        spec_act = (
            j_at_onset * self.electrode_area / self.ecsa
            if self.ecsa > 0 else 0.0
        )

        return LSVAnalysisResult(
            catalyst_type            = self.catalyst_type,
            electrolyte              = self.electrolyte_info,
            e_onset                  = e_onset,
            e_onset_method           = onset_method + (" (iR-corrected)" if ir_compensated else ""),
            tafel_slope              = tafel["slope"],
            tafel_slope_std          = tafel["slope_std"],
            exchange_current_density = tafel["j0"],
            tafel_r_squared          = tafel["r2"],
            tafel_region             = tafel["region"],
            overpotential_10         = eta_10,
            overpotential_50         = eta_50,
            overpotential_100        = eta_100,
            j_at_onset               = j_at_onset,
            j_limiting               = j_lim,
            e_half_wave              = e_half,
            mass_activity            = mass_act,
            specific_activity        = spec_act,
            ir_compensated           = ir_compensated,
            r_s_used                 = r_s_ohms,
            scan_rate                = self.scan_rate,
            electrode_area           = self.electrode_area,
            ecsa                     = self.ecsa,
            catalyst_loading         = self.catalyst_loading,
            mechanism_interpretation = self._interpret_tafel(tafel["slope"]),
            performance_rating       = self._rate_performance(e_onset, tafel["slope"], eta_10),
        )

    # ── E_onset ───────────────────────────────────────────────────────────────

    def _detect_onset(self, potential, j) -> tuple[float, str]:
        n       = len(potential)
        bl_end  = max(int(n * 0.15), 5)
        j_max   = float(np.max(j))
        baseline = float(np.mean(j[:bl_end]))

        # Metal-free: gentler threshold (3% instead of 5%)
        thresh_pct = 0.03 if self.catalyst_type == CATALYST_METAL_FREE else 0.05
        threshold  = baseline + thresh_pct * j_max

        try:
            m_bl, b_bl = np.polyfit(potential[:bl_end], j[:bl_end], 1)
        except Exception:
            m_bl, b_bl = 0.0, baseline

        above = np.where(j > threshold)[0]
        if len(above) == 0:
            return float(potential[n // 2]), "threshold"

        region_s = int(above[0])
        region_e = min(region_s + int(n * 0.15), n - 1)
        if region_e <= region_s + 2:
            return float(potential[region_s]), "threshold"

        try:
            m_rise, b_rise = np.polyfit(
                potential[region_s:region_e], j[region_s:region_e], 1
            )
            denom = m_rise - m_bl
            if abs(denom) < 1e-10:
                return float(potential[region_s]), "threshold"
            onset = float(np.clip(
                (b_bl - b_rise) / denom, potential.min(), potential.max()
            ))
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
            return {
                "slope"    : tafel_mv,
                "slope_std": slope_std * 1000.0,
                "j0"       : j0,
                "r2"       : r_val ** 2,
                "region"   : (float(E_tafel[0]), float(E_tafel[-1])),
            }
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

    def _interpret_tafel(self, slope_mv: float) -> str:
        if np.isnan(slope_mv):
            return "Tafel slope could not be determined"

        abs_s     = abs(slope_mv)
        ctype     = self.catalyst_type
        el        = self.electrolyte_info
        conc_note = f" [{el.concentration} M {el.compound}]"

        # Metal-free: completely different thresholds
        if ctype == CATALYST_METAL_FREE:
            if abs_s < 120:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Good kinetics for metal-free catalyst. "
                    f"Rate-limiting: surface defect activation or C-H bond breaking."
                )
            elif abs_s < 250:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Normal range for metal-free / ceramic catalyst (B4C, CNT, N-doped C). "
                    f"Multi-step mechanism without d-band facilitation."
                )
            else:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Very high slope: strong mass-transport or surface passivation. "
                    f"Check electrode preparation and electrolyte diffusion."
                )

        # Metal oxide
        if ctype == CATALYST_METAL_OXIDE:
            if abs_s < 60:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Excellent for metal oxide. M(OH)x/MOOx redox mediation active."
                )
            elif abs_s < 120:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Typical for oxide catalyst. First electron transfer limiting."
                )
            else:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"High slope for oxide: check OH- supply "
                    f"({'low' if el.concentration < 0.5 else 'normal'} concentration)."
                )

        # Noble metal and alloy
        conc_warning = ""
        if el.is_acidic() and el.compound == ACID_HCl:
            conc_warning = " [Cl- adsorption may inflate Tafel slope]"
        elif el.is_alkaline() and el.concentration < 0.1:
            conc_warning = " [Low OH- — insufficient for bifunctional mechanism]"

        if abs_s < 40:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"Chemical step rate-limiting{conc_warning}"
            )
        elif abs_s < 75:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"Langmuir adsorption limiting — PtRu / PdAu bifunctional mechanism{conc_warning}"
            )
        elif abs_s < 120:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"First electron transfer limiting (Volmer / Butler-Volmer){conc_warning}"
            )
        else:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"High value: CO poisoning or diffusion limiting. "
                f"Consider alloy catalyst or higher {el.compound} concentration.{conc_warning}"
            )

    def _rate_performance(self, e_onset, tafel_slope, eta_10) -> str:
        """
        Performance rating adapted per catalyst type and electrolyte.
        Metal-free catalysts are judged on a different scale.
        """
        el    = self.electrolyte_info
        ctype = self.catalyst_type
        score = 0

        # E_onset thresholds per catalyst type and electrolyte
        onset_thresholds = {
            CATALYST_NOBLE_METAL : (0.20, 0.40) if el.is_alkaline() else (0.40, 0.60),
            CATALYST_ALLOY       : (0.10, 0.30) if el.is_alkaline() else (0.30, 0.50),
            CATALYST_METAL_OXIDE : (1.30, 1.50) if el.is_alkaline() else (1.50, 1.70),
            CATALYST_METAL_FREE  : (0.40, 0.70) if el.is_alkaline() else (0.55, 0.85),
        }
        lo, hi = onset_thresholds.get(ctype, (0.40, 0.60))

        if e_onset < lo:   score += 3
        elif e_onset < hi: score += 1

        # Tafel slope thresholds — metal-free has different scale
        if not np.isnan(tafel_slope):
            if ctype == CATALYST_METAL_FREE:
                if abs(tafel_slope) < 120:   score += 3
                elif abs(tafel_slope) < 200: score += 2
                elif abs(tafel_slope) < 300: score += 1
            elif ctype == CATALYST_METAL_OXIDE:
                if abs(tafel_slope) < 60:   score += 3
                elif abs(tafel_slope) < 100: score += 2
                elif abs(tafel_slope) < 130: score += 1
            else:
                if abs(tafel_slope) < 60:   score += 3
                elif abs(tafel_slope) < 90:  score += 2
                elif abs(tafel_slope) < 120: score += 1

        # Overpotential at 10 mA/cm²
        if not np.isnan(eta_10):
            if ctype == CATALYST_METAL_FREE:
                if eta_10 < 0.3:   score += 3
                elif eta_10 < 0.5: score += 2
                elif eta_10 < 0.7: score += 1
            else:
                if eta_10 < 0.1:   score += 3
                elif eta_10 < 0.2: score += 2
                elif eta_10 < 0.3: score += 1

        # Electrolyte-specific bonus/penalty
        if el.compound == BASE_Na2CO3:
            score = max(0, score - 1)  # Na2CO3 inherently slower
        if el.is_alkaline() and el.concentration >= 1.0 and ctype == CATALYST_METAL_FREE:
            score += 1  # high OH- helps metal-free

        catalyst_label = {
            CATALYST_NOBLE_METAL : "noble metal",
            CATALYST_ALLOY       : "alloy",
            CATALYST_METAL_OXIDE : "metal oxide",
            CATALYST_METAL_FREE  : "metal-free",
        }.get(ctype, "catalyst")

        el_label = f"{el.concentration} M {el.compound}"

        if score >= 8:
            return f"Excellent {catalyst_label} in {el_label} — publishable in high-IF journals"
        elif score >= 5:
            return f"Good {catalyst_label} in {el_label} — suitable catalyst"
        elif score >= 3:
            return f"Moderate {catalyst_label} in {el_label} — optimization needed"
        else:
            return f"Poor performance in {el_label} — consider different composition or electrolyte"

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

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def for_noble_metal(cls, electrolyte_compound="H2SO4", concentration=0.5, **kw):
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_NOBLE_METAL, **kw)

    @classmethod
    def for_alloy(cls, electrolyte_compound="H2SO4", concentration=0.5, **kw):
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_ALLOY, **kw)

    @classmethod
    def for_metal_free(cls, electrolyte_compound="KOH", concentration=1.0, **kw):
        """For B4C, N-doped Carbon, CNT, rGO — uses wider Tafel thresholds."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_FREE, **kw)

    @classmethod
    def for_metal_oxide(cls, electrolyte_compound="KOH", concentration=1.0, **kw):
        """For NiO, Co3O4, Co2NiO4, MnO2."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_OXIDE, **kw)

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        for enc in ["latin-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(filepath, comment="#", encoding=enc,
                                 sep=None, engine="python")
                c  = df.columns.tolist()
                return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read: {filepath}")
