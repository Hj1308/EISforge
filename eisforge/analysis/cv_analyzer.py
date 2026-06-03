"""
CV Analyzer — Automatic Cyclic Voltammetry Analysis for AOR.
Author: Hoda Jafari | May 2026

Supports three catalyst families:
    - noble_metal  : Pt, Pd, Au, Rh — uses I_f/I_b, H-UPD ECSA
    - alloy        : PtRu, PtSn, PdAu, PtCu — bifunctional mechanism
    - metal_oxide  : NiO, Co3O4, NiCoO, MnO2 — oxide peaks, high onset
    - carbon_material   : N-doped Carbon, CNT, rGO, graphene-based — no I_f/I_b, Cdl ECSA

Electrolyte specifics:
    - Acid type    : H2SO4, HClO4, HCl, HNO3
    - Base type    : KOH, NaOH, Na2CO3, NH3
    - Concentration: float in mol/L (M)

iR Compensation:
    E_corrected = E_measured - I(A) × R_s(Ohm)
    Current must be in Amperes. Set current_unit="mA" for auto-conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


# ── Catalyst type constants ──────────────────────────────────────────────────

CATALYST_NOBLE_METAL = "noble_metal"   # Pt, Pd, Au, Rh
CATALYST_ALLOY       = "alloy"         # PtRu, PtSn, PdAu, PtCu, PdNi
CATALYST_METAL_OXIDE = "metal_oxide"   # NiO, Co3O4, NiCoO, MnO2, Co2NiO4
CATALYST_METAL_FREE  = "carbon_material"    # N-doped C, CNT, graphene-based, rGO, g-C3N4

# ── Electrolyte constants ───────────────────────────────────────────────────

ACID_H2SO4  = "H2SO4"
ACID_HClO4  = "HClO4"
ACID_HCl    = "HCl"
ACID_HNO3   = "HNO3"

BASE_KOH    = "KOH"
BASE_NaOH   = "NaOH"
BASE_Na2CO3 = "Na2CO3"
BASE_NH3    = "NH3"


@dataclass
class ElectrolyteInfo:
    """
    Detailed electrolyte description.

    Parameters
    ----------
    media : str
        'acidic' or 'alkaline'
    compound : str
        Specific acid or base: 'H2SO4', 'HClO4', 'KOH', 'NaOH', etc.
    concentration : float
        Concentration in mol/L (M).
    """
    media         : str   = "acidic"
    compound      : str   = "H2SO4"
    concentration : float = 0.5

    def label(self) -> str:
        return f"{self.concentration} M {self.compound}"

    def is_acidic(self) -> bool:
        return self.media == "acidic"

    def is_alkaline(self) -> bool:
        return self.media == "alkaline"

    @classmethod
    def from_string(cls, s: str, concentration: float = 0.5) -> "ElectrolyteInfo":
        """
        Create ElectrolyteInfo from a simple string.
        e.g. 'acidic', 'alkaline', 'KOH', 'H2SO4', '1M KOH'
        """
        s = s.strip()
        acids   = [ACID_H2SO4, ACID_HClO4, ACID_HCl, ACID_HNO3]
        bases   = [BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3]
        compound = "H2SO4"
        media    = "acidic"
        for a in acids:
            if a.lower() in s.lower():
                compound, media = a, "acidic"
                break
        for b in bases:
            if b.lower() in s.lower():
                compound, media = b, "alkaline"
                break
        if "alkaline" in s.lower() or "base" in s.lower():
            media = "alkaline"
            if compound == "H2SO4":
                compound = "KOH"
        return cls(media=media, compound=compound, concentration=concentration)


@dataclass
class CVAnalysisResult:
    """Results of complete CV analysis."""

    # Catalyst and electrolyte info
    catalyst_type    : str = CATALYST_NOBLE_METAL
    electrolyte      : ElectrolyteInfo = field(default_factory=ElectrolyteInfo)

    # Potentials
    e_onset          : float = 0.0
    e_onset_method   : str   = ""
    e_forward_peak   : float = 0.0
    e_backward_peak  : float = 0.0

    # Currents (mA)
    i_forward_peak   : float = 0.0
    i_backward_peak  : float = 0.0
    if_ib_ratio      : float = float("nan")
    baseline_current : float = 0.0

    # Current density — geometric (mA/cm²)
    j_forward_peak   : float = 0.0
    j_backward_peak  : float = 0.0

    # Current density — ECSA-normalized (mA/cm²_metal or mA/cm²_BET)
    j_specific_forward  : float = 0.0
    j_specific_backward : float = 0.0

    # Metal-free specific
    capacitive_background_mA  : float = 0.0   # background subtracted
    net_faradaic_current_mA   : float = 0.0   # i_f - background
    cdl_mF_cm2                : float = 0.0   # double-layer capacitance

    # iR compensation
    ir_compensated : bool  = False
    r_s_used       : float = 0.0

    # Measurement parameters
    scan_rate        : float = 50.0
    electrode_area   : float = 1.0
    ecsa             : float = 0.0
    catalyst_loading : float = 0.0
    interpretation   : str   = ""

    def summary(self) -> str:
        el = self.electrolyte
        is_carbon_material = self.catalyst_type == CATALYST_METAL_FREE

        lines = [
            "=" * 68,
            "  CV Analysis Results — EISForge",
            "=" * 68,
            f"  Catalyst type     : {self._catalyst_label()}",
            f"  Electrolyte       : {el.label()}  ({el.media})",
            "-" * 68,
        ]

        if self.ir_compensated:
            lines.append(f"  iR Compensation   : APPLIED  (R_s = {self.r_s_used:.3f} Ω)")
        else:
            lines.append(f"  iR Compensation   : not applied")

        lines += [
            f"  E_onset           = {self.e_onset:.4f} V  ({self.e_onset_method})",
            f"  E_forward_peak    = {self.e_forward_peak:.4f} V",
        ]

        if not is_carbon_material:
            lines.append(
                f"  E_backward_peak   = {self.e_backward_peak:.4f} V"
            )

        lines += [
            "-" * 68,
            f"  I_forward_peak    = {self.i_forward_peak:.4f} mA",
        ]

        if is_carbon_material:
            lines += [
                f"  Capacitive bg     = {self.capacitive_background_mA:.4f} mA",
                f"  Net faradaic I    = {self.net_faradaic_current_mA:.4f} mA",
                f"  C_dl              = {self.cdl_mF_cm2:.4f} mF/cm²",
                "  I_f/I_b           : NOT applicable (metal-free catalyst)",
            ]
        else:
            lines += [
                f"  I_backward_peak   = {self.i_backward_peak:.4f} mA",
                f"  I_f/I_b           = {self.if_ib_ratio:.3f}"
                if not np.isnan(self.if_ib_ratio)
                else "  I_f/I_b           : not calculable",
            ]

        lines += [
            "-" * 68,
            f"  j_f (geometric)   = {self.j_forward_peak:.4f} mA/cm²",
        ]
        if self.ecsa > 0:
            label = "cm²_BET" if is_carbon_material else "cm²_metal"
            lines.append(
                f"  j_f (ECSA)        = {self.j_specific_forward:.4f} mA/{label}"
            )

        lines += [
            f"  Scan rate         = {self.scan_rate} mV/s",
            f"  Electrode area    = {self.electrode_area} cm²",
            "-" * 68,
            f"  Interpretation    : {self.interpretation}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def _catalyst_label(self) -> str:
        labels = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt / Pd / Au / Rh)",
            CATALYST_ALLOY       : "Alloy (PtRu / PtSn / PdAu / PtCu ...)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO / Co3O4 / MnO2 ...)",
            CATALYST_METAL_FREE  : "Metal-Free (carbon_material / N-doped C / CNT ...)",
        }
        return labels.get(self.catalyst_type, self.catalyst_type)


class CVAnalyzer:
    """
    Automatic CV analyzer for AOR — supports all catalyst families.

    Parameters
    ----------
    scan_rate : float
        Scan rate in mV/s (default: 50).
    electrode_area : float
        Geometric electrode area in cm² (default: 1.0).
    ecsa : float
        Electrochemically active surface area in cm²_metal or cm²_BET.
        For metal-free catalysts, set to BET-derived surface area.
    onset_method : str
        E_onset detection: 'tangent', 'threshold', or 'derivative'.
    electrolyte : str or ElectrolyteInfo
        Electrolyte description. Can be:
          - ElectrolyteInfo object (recommended)
          - simple string: 'acidic', 'alkaline', 'KOH', 'H2SO4'
    electrolyte_concentration : float
        Concentration in mol/L — used when electrolyte is a string.
    catalyst_type : str
        One of: 'noble_metal', 'alloy', 'metal_oxide', 'carbon_material'
        Controls which metrics are computed and how results are interpreted.
    current_unit : str
        Unit of input current: 'mA', 'A', 'uA', 'nA'.
    catalyst_loading : float
        Catalyst loading in mg/cm².
    """

    _UNIT_TO_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "μA": 1e-3, "nA": 1e-6}

    # E_onset reference ranges per electrolyte compound (V vs RHE)
    # Format: {compound: (excellent_max, moderate_max)}
    _ONSET_RANGES = {
        # Acids
        ACID_H2SO4 : {"noble_metal": (0.45, 0.65), "alloy": (0.35, 0.55),
                       "metal_oxide": (1.30, 1.55), "carbon_material": (0.60, 0.90)},
        ACID_HClO4 : {"noble_metal": (0.40, 0.60), "alloy": (0.30, 0.50),
                       "metal_oxide": (1.25, 1.50), "carbon_material": (0.55, 0.85)},
        ACID_HCl   : {"noble_metal": (0.50, 0.70), "alloy": (0.40, 0.60),
                       "metal_oxide": (1.35, 1.60), "carbon_material": (0.65, 0.95)},
        ACID_HNO3  : {"noble_metal": (0.50, 0.70), "alloy": (0.40, 0.60),
                       "metal_oxide": (1.35, 1.60), "carbon_material": (0.65, 0.95)},
        # Bases
        BASE_KOH   : {"noble_metal": (0.20, 0.40), "alloy": (0.10, 0.30),
                       "metal_oxide": (1.30, 1.50), "carbon_material": (0.40, 0.70)},
        BASE_NaOH  : {"noble_metal": (0.22, 0.42), "alloy": (0.12, 0.32),
                       "metal_oxide": (1.32, 1.52), "carbon_material": (0.42, 0.72)},
        BASE_Na2CO3: {"noble_metal": (0.30, 0.55), "alloy": (0.20, 0.45),
                       "metal_oxide": (1.40, 1.65), "carbon_material": (0.50, 0.80)},
        BASE_NH3   : {"noble_metal": (0.35, 0.60), "alloy": (0.25, 0.50),
                       "metal_oxide": (1.45, 1.70), "carbon_material": (0.55, 0.85)},
    }

    def __init__(
        self,
        scan_rate              : float = 50.0,
        electrode_area         : float = 1.0,
        ecsa                   : float = 0.0,
        onset_method           : str   = "tangent",
        onset_threshold        : float = 0.05,
        smoothing              : bool  = True,
        electrolyte            = "acidic",
        electrolyte_concentration: float = 0.5,
        catalyst_type          : str   = CATALYST_NOBLE_METAL,
        current_unit           : str   = "mA",
        catalyst_loading       : float = 0.0,
    e_ref_vs_rhe     : float = 0.0,
    ) -> None:
        self.scan_rate               = scan_rate
        self.electrode_area          = max(electrode_area, 1e-10)
        self.ecsa                    = ecsa
        self.onset_method            = onset_method
        self.onset_threshold         = onset_threshold
        self.smoothing               = smoothing
        self.catalyst_type           = catalyst_type
        self.catalyst_loading        = catalyst_loading
        self.e_ref_vs_rhe = e_ref_vs_rhe
        self.current_unit            = current_unit
        self._unit_factor            = self._UNIT_TO_MA.get(current_unit, 1.0)

        # Build ElectrolyteInfo
        if isinstance(electrolyte, ElectrolyteInfo):
            self.electrolyte_info = electrolyte
        else:
            self.electrolyte_info = ElectrolyteInfo.from_string(
                str(electrolyte), concentration=electrolyte_concentration
            )

        # Adjust onset sensitivity for metal-free (higher Tafel slope)
        if catalyst_type == CATALYST_METAL_FREE:
            self.onset_threshold = max(onset_threshold, 0.03)
            logger.info("Metal-Free mode: onset threshold adjusted for higher Tafel slope.")

    # ── iR Compensation ───────────────────────────────────────────────────────

    @staticmethod
    def apply_ir_compensation(
        potential  : np.ndarray,
        current_ma : np.ndarray,
        r_s_ohms   : float,
    ) -> np.ndarray:
        """
        Apply iR compensation.
        E_corrected = E_measured - I(A) × R_s(Ω)
        """
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
    ) -> CVAnalysisResult:
        """
        Full CV analysis — adapts automatically to catalyst_type.

        Parameters
        ----------
        potential : np.ndarray   Potential (V)
        current   : np.ndarray   Current (in current_unit, default mA)
        r_s_ohms  : float        Solution resistance for iR compensation (Ω)

        Returns
        -------
        CVAnalysisResult
        """
        potential = np.asarray(potential, dtype=float)
        current   = np.asarray(current,   dtype=float)

        if len(potential) < 6:
            raise ValueError(
                f"Insufficient data: {len(potential)} points. Minimum 6 required."
            )

        current_ma     = current * self._unit_factor
        ir_compensated = r_s_ohms > 0

        if ir_compensated:
            potential = self.apply_ir_compensation(potential, current_ma, r_s_ohms)
            logger.info(f"iR compensation applied: R_s = {r_s_ohms:.3f} Ω")

        if self.smoothing:
            current_ma = self._smooth(current_ma)

        # Metal-free: background subtraction before peak detection
        background = np.zeros_like(current_ma)
        if self.catalyst_type == CATALYST_METAL_FREE:
            current_ma, background = self._subtract_capacitive_background(
                potential, current_ma
            )

        # Split scans
        fwd_mask, bwd_mask = self._split_scans(potential, current_ma)
        e_fwd, i_fwd = potential[fwd_mask], current_ma[fwd_mask]
        e_bwd, i_bwd = potential[bwd_mask], current_ma[bwd_mask]

        if len(e_fwd) < 3 or len(e_bwd) < 3:
            mid  = len(potential) // 2
            e_fwd, i_fwd = potential[:mid], current_ma[:mid]
            e_bwd, i_bwd = potential[mid:], current_ma[mid:]
            logger.warning("Scan split: midpoint fallback.")

        # Peaks
        i_f = float(i_fwd[np.argmax(i_fwd)])
        e_f = float(e_fwd[np.argmax(i_fwd)])
        i_b = float(i_bwd[np.argmax(i_bwd)])
        e_b = float(e_bwd[np.argmax(i_bwd)])

        # E_onset (higher derivative sensitivity for metal-free)
        onset_method = (
            "derivative"
            if self.catalyst_type == CATALYST_METAL_FREE
            and self.onset_method == "tangent"
            else self.onset_method
        )
        e_onset, baseline = self._detect_onset_method(
            onset_method, e_fwd, i_fwd, i_f
        )

        # I_f/I_b — only for metal/alloy catalysts
        if self.catalyst_type in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
            ratio = i_f / i_b if i_b > 1e-10 else float("nan")
        else:
            ratio = float("nan")   # not meaningful for oxide/metal-free

        # Current densities
        j_f      = i_f / self.electrode_area
        j_b      = i_b / self.electrode_area
        j_spec_f = i_f / self.ecsa if self.ecsa > 0 else 0.0
        j_spec_b = i_b / self.ecsa if self.ecsa > 0 else 0.0

        # Capacitive background (metal-free)
        bg_mean   = float(np.mean(np.abs(background))) if self.catalyst_type == CATALYST_METAL_FREE else 0.0
        net_farad = i_f - bg_mean
        cdl       = self._estimate_cdl(background) if self.catalyst_type == CATALYST_METAL_FREE else 0.0

        # ── Edge guard: onset must not land on the very edge (detection failure) ──
        _p10 = float(np.percentile(e_fwd, 10))
        _p90 = float(np.percentile(e_fwd, 90))
        if e_onset <= _p10 or e_onset >= _p90:
            e_onset, baseline = self._onset_threshold(e_fwd, i_fwd, i_f)
            onset_method = "threshold (auto-fallback)"


        return CVAnalysisResult(
            catalyst_type           = self.catalyst_type,
            electrolyte             = self.electrolyte_info,
            e_onset                 = e_onset,
            e_onset_method          = onset_method + (" (iR-corrected)" if ir_compensated else ""),
            e_forward_peak          = e_f,
            e_backward_peak         = e_b,
            i_forward_peak          = i_f,
            i_backward_peak         = i_b,
            if_ib_ratio             = ratio,
            baseline_current        = baseline,
            j_forward_peak          = j_f,
            j_backward_peak         = j_b,
            j_specific_forward      = j_spec_f,
            j_specific_backward     = j_spec_b,
            capacitive_background_mA= bg_mean,
            net_faradaic_current_mA = net_farad,
            cdl_mF_cm2              = cdl,
            ir_compensated          = ir_compensated,
            r_s_used                = r_s_ohms,
            scan_rate               = self.scan_rate,
            electrode_area          = self.electrode_area,
            ecsa                    = self.ecsa,
            catalyst_loading        = self.catalyst_loading,
            interpretation          = self._interpret(e_onset, i_f, i_b, ratio),
        )

    # ── Capacitive background subtraction (metal-free) ────────────────────────

    def _subtract_capacitive_background(
        self,
        potential  : np.ndarray,
        current_ma : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate and subtract capacitive background current.

        For metal-free catalysts (N-doped C, CNT, graphene-based), the CV shows a
        roughly rectangular capacitive envelope. We fit a low-order polynomial
        to regions far from the faradaic peak and subtract it.

        Returns
        -------
        corrected_current, background
        """
        n         = len(current_ma)
        bg_pts    = int(n * 0.15)
        bg_pts    = max(bg_pts, 3)

        # Use first and last 15% as baseline regions
        x_bg = np.concatenate([potential[:bg_pts], potential[-bg_pts:]])
        y_bg = np.concatenate([current_ma[:bg_pts], current_ma[-bg_pts:]])

        try:
            coeffs     = np.polyfit(x_bg, y_bg, 1)          # linear background
            background = np.polyval(coeffs, potential)
            corrected  = current_ma - background
            logger.info("Capacitive background subtracted (linear fit).")
        except Exception:
            background = np.zeros_like(current_ma)
            corrected  = current_ma.copy()

        return corrected, background

    def _estimate_cdl(self, background: np.ndarray) -> float:
        """
        Estimate double-layer capacitance from background current.
        C_dl = I_background / (scan_rate)
        scan_rate in V/s for result in F, then convert to mF/cm²
        """
        if self.scan_rate <= 0 or self.electrode_area <= 0:
            return 0.0
        scan_rate_V_s = self.scan_rate * 1e-3   # mV/s → V/s
        i_bg_A        = np.mean(np.abs(background)) * 1e-3   # mA → A
        cdl_F         = i_bg_A / scan_rate_V_s
        cdl_mF_cm2    = (cdl_F * 1e3) / self.electrode_area
        return float(cdl_mF_cm2)

    # ── Scan splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split_scans(potential, current):
        n    = len(potential)
        peak = int(np.argmax(potential))
        if peak < 2:     peak = n // 2
        if peak > n - 3: peak = n // 2
        fwd = np.zeros(n, dtype=bool)
        bwd = np.zeros(n, dtype=bool)
        fwd[:peak + 1] = True
        bwd[peak + 1:] = True
        return fwd, bwd

    # ── E_onset detection ─────────────────────────────────────────────────────

    def _detect_onset_method(self, method, e_fwd, i_fwd, i_peak):
        if method == "tangent":
            return self._onset_tangent(e_fwd, i_fwd, i_peak)
        elif method == "threshold":
            return self._onset_threshold(e_fwd, i_fwd, i_peak)
        else:
            return self._onset_derivative(e_fwd, i_fwd)

    def _detect_onset(self, potential, current, i_peak):
        return self._detect_onset_method(self.onset_method, potential, current, i_peak)

    def _onset_tangent(self, potential, current, i_peak):
        n      = len(potential)
        bl_end = max(int(n * 0.2), 3)
        baseline = float(np.mean(current[:bl_end]))
        try:
            m_bl, b_bl = np.polyfit(potential[:bl_end], current[:bl_end], 1)
        except Exception:
            return self._onset_threshold(potential, current, i_peak)
        peak_idx = int(np.argmax(current))
        s = int(peak_idx * 0.55)
        e = int(peak_idx * 0.85)
        if e <= s + 2:
            s, e = max(0, peak_idx - max(int(n * 0.1), 3)), peak_idx
        try:
            m_rise, b_rise = np.polyfit(potential[s:e], current[s:e], 1)
            denom = m_rise - m_bl
            if abs(denom) < 1e-12:
                return self._onset_threshold(potential, current, i_peak)
            onset = float(np.clip((b_bl - b_rise) / denom, potential.min(), potential.max()))
            return onset, baseline
        except Exception:
            return self._onset_threshold(potential, current, i_peak)

    def _onset_threshold(self, potential, current, i_peak):
        n_bl     = max(int(len(current) * 0.15), 3)
        baseline = float(np.mean(current[:n_bl]))
        threshold = baseline + self.onset_threshold * i_peak
        above = np.where(current > threshold)[0]
        if len(above) == 0:
            return float(potential[len(potential) // 2]), baseline
        return float(potential[int(above[0])]), baseline

    def _onset_derivative(self, potential, current):
        if len(current) < 5:
            return self._onset_threshold(potential, current, float(np.max(current)))
        try:
            d2   = np.gradient(np.gradient(current, potential), potential)
            peak = int(np.argmax(current))
            pre  = d2[:peak]
            if len(pre) == 0:
                return float(potential[0]), float(current[0])
            idx  = int(np.argmax(pre))
            return float(potential[idx]), float(np.mean(current[:max(idx, 1)]))
        except Exception:
            return self._onset_threshold(potential, current, float(np.max(current)))

    # ── Smoothing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(current: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter
            w = min(11, len(current) // 4)
            if w < 5: return current.copy()
            if w % 2 == 0: w += 1
            return savgol_filter(current, window_length=w, polyorder=3)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter1d
                return gaussian_filter1d(current, sigma=2)
            except Exception:
                return current.copy()

    # ── Interpretation ────────────────────────────────────────────────────────

    def _interpret(self, e_onset, i_f, i_b, ratio, e_ref_vs_rhe: float = 0.0) -> str:
        parts  = []
        el     = self.electrolyte_info
        ctype  = self.catalyst_type
        conc   = el.concentration

        # ── Onset potential assessment ──────────────────────────────────────
        ranges = self._ONSET_RANGES.get(
            el.compound,
            self._ONSET_RANGES[ACID_H2SO4 if el.is_acidic() else BASE_KOH]
        ).get(ctype, (0.45, 0.65))

        lo, hi = ranges

        # Concentration correction: higher concentration → slightly lower onset
        conc_note = ""
        if conc != 0.5 and el.is_acidic():
            conc_note = f" [{conc} M {el.compound}]"
        elif conc != 1.0 and el.is_alkaline():
            conc_note = f" [{conc} M {el.compound}]"

        # ── RHE conversion for interpretation ──────────────────────────────
        # If pH is known from alkaline electrolyte, include Nernst term
        if e_ref_vs_rhe != 0.0:
            if el.is_alkaline():
                _conc_koh = max(el.concentration, 1e-6)
                _ph = 14 + np.log10(_conc_koh)  # approx for KOH/NaOH
                _ph = min(_ph, 14.0)
            else:
                _ph = 0.0
            e_onset_rhe = e_onset + e_ref_vs_rhe + 0.059 * _ph
            rhe_note = f" (= {e_onset_rhe:.3f} V vs RHE)"
        else:
            e_onset_rhe = e_onset
            rhe_note = ""

        if e_onset_rhe < lo:
            parts.append(f"E_onset={e_onset:.3f} V{rhe_note} — Excellent activity{conc_note}")
        elif e_onset_rhe < hi:
            parts.append(f"E_onset={e_onset:.3f} V{rhe_note} — Moderate activity{conc_note}")
        else:
            parts.append(f"E_onset={e_onset:.3f} V{rhe_note} — High overpotential{conc_note}")

        # ── Catalyst-specific assessment ────────────────────────────────────
        if ctype == CATALYST_METAL_FREE:
            parts.append(
                "Metal-free carbon catalyst: I_f/I_b NOT applicable. "
                "ECSA estimated via C_dl method. "
                "No CO poisoning pathway — IPA/alcohol oxidation via different mechanism."
            )
            if i_f > 0:
                parts.append(f"Net peak current = {i_f:.4f} mA (after capacitive background subtraction)")
            if not np.isnan(ratio):
                pass  # deliberately skip I_f/I_b for metal-free

        elif ctype == CATALYST_METAL_OXIDE:
            parts.append(
                "Metal oxide catalyst: oxidation proceeds via M(OH)x/MOOx redox mediation. "
                "High onset expected (>1.3 V vs RHE in alkaline)."
            )
            if not np.isnan(ratio):
                parts.append(f"I_f/I_b = {ratio:.2f}")

        elif ctype == CATALYST_ALLOY:
            if not np.isnan(ratio):
                if ratio > 2.0:
                    parts.append(f"I_f/I_b={ratio:.2f} — Excellent CO tolerance (bifunctional mechanism active)")
                elif ratio > 1.0:
                    parts.append(f"I_f/I_b={ratio:.2f} — Good CO tolerance")
                else:
                    parts.append(f"I_f/I_b={ratio:.2f} — CO poisoning detected — check alloy composition")

        else:   # noble_metal
            if not np.isnan(ratio):
                if ratio > 2.0:
                    parts.append(f"I_f/I_b={ratio:.2f} — Excellent CO tolerance")
                elif ratio > 1.0:
                    parts.append(f"I_f/I_b={ratio:.2f} — Good CO tolerance")
                elif ratio > 0.5:
                    parts.append(f"I_f/I_b={ratio:.2f} — Moderate CO poisoning")
                else:
                    parts.append(f"I_f/I_b={ratio:.2f} — Severe CO poisoning")

        # ── Electrolyte-specific note ───────────────────────────────────────
        if el.compound == BASE_Na2CO3:
            parts.append("Na2CO3 electrolyte: mildly alkaline (pH ~11.6) — slower OH- supply than KOH/NaOH")
        elif el.compound == ACID_HCl:
            parts.append("HCl electrolyte: Cl- may adsorb on Pt/Pd — check for chloride poisoning")
        elif el.compound == ACID_HNO3:
            parts.append("HNO3 electrolyte: oxidising acid — may affect catalyst surface")

        # ── Concentration note ──────────────────────────────────────────────
        if el.is_acidic() and conc > 1.0:
            parts.append(f"High acid concentration ({conc} M) may suppress OH_ads formation")
        elif el.is_alkaline() and conc < 0.1:
            parts.append(f"Low base concentration ({conc} M) — insufficient OH- supply; higher Tafel slope expected")

        return " | ".join(parts)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @classmethod
    def for_noble_metal(
        cls,
        electrolyte_compound   : str   = "H2SO4",
        concentration          : float = 0.5,
        **kwargs,
    ) -> "CVAnalyzer":
        """Convenience constructor for Pt, Pd, Au, Rh catalysts."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_NOBLE_METAL, **kwargs)

    @classmethod
    def for_alloy(
        cls,
        electrolyte_compound   : str   = "H2SO4",
        concentration          : float = 0.5,
        **kwargs,
    ) -> "CVAnalyzer":
        """Convenience constructor for PtRu, PtSn, PdAu, PtCu catalysts."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_ALLOY, **kwargs)

    @classmethod
    def for_carbon_material(
        cls,
        electrolyte_compound   : str   = "KOH",
        concentration          : float = 1.0,
        **kwargs,
    ) -> "CVAnalyzer":
        """
        Convenience constructor for N-doped Carbon, CNT, rGO, graphene-based catalysts.

        Automatically:
          - Skips I_f/I_b calculation
          - Applies capacitive background subtraction
          - Uses derivative onset detection (sensitive to gentle slope)
          - Reports C_dl for ECSA estimation
        """
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_FREE, **kwargs)

    @classmethod
    def for_metal_oxide(
        cls,
        electrolyte_compound   : str   = "KOH",
        concentration          : float = 1.0,
        **kwargs,
    ) -> "CVAnalyzer":
        """Convenience constructor for NiO, Co3O4, Co2NiO4, MnO2 catalysts."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_OXIDE, **kwargs)

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
        raise ValueError(f"Cannot read file: {filepath}")


# ── Quick usage examples ──────────────────────────────────────────────────────

if __name__ == "__main__":

    E = np.linspace(0.0, 1.2, 200)
    I = np.sin(np.pi * E) * 5 + np.random.normal(0, 0.1, 200)

    # ── Pt in H2SO4 (noble metal, acidic) ──────────────────────────────────
    ana_pt = CVAnalyzer.for_noble_metal(
        electrolyte_compound="H2SO4",
        concentration=0.5,
        scan_rate=50,
        electrode_area=0.196,
    )
    result_pt = ana_pt.analyze(E, I)
    print(result_pt.summary())

    # ── carbon_material in KOH (metal-free, alkaline) ──────────────────────────────────
    ana_b4c = CVAnalyzer.for_carbon_material(
        electrolyte_compound="KOH",
        concentration=1.0,
        scan_rate=50,
        electrode_area=0.196,
    )
    result_b4c = ana_b4c.analyze(E, I)
    print(result_b4c.summary())

    # ── PtRu in KOH (alloy, alkaline) ──────────────────────────────────────
    ana_ptru = CVAnalyzer.for_alloy(
        electrolyte_compound="KOH",
        concentration=1.0,
        scan_rate=50,
        electrode_area=0.196,
    )
    result_ptru = ana_ptru.analyze(E, I)
    print(result_ptru.summary())
