"""
CV Analyzer — Automatic Cyclic Voltammetry Analysis for AOR.
Author: Hoda Jafari | May 2026

iR Compensation:
    E_corrected = E_measured - I(A) × R_s(Ohm)

    IMPORTANT: Current must be in Amperes for the formula to give Volts.
    If current is in mA, set current_unit="mA" and the class converts automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CVAnalysisResult:
    """Results of complete CV analysis."""

    # Potentials
    e_onset: float
    e_onset_method: str
    e_forward_peak: float
    e_backward_peak: float

    # Currents (mA)
    i_forward_peak: float
    i_backward_peak: float
    if_ib_ratio: float
    baseline_current: float

    # Current density — geometric (mA/cm²)
    j_forward_peak: float = 0.0
    j_backward_peak: float = 0.0

    # Current density — ECSA-normalized (mA/cm²_metal)
    j_specific_forward: float = 0.0
    j_specific_backward: float = 0.0

    # iR compensation
    ir_compensated: bool = False
    r_s_used: float = 0.0

    # Measurement parameters
    scan_rate: float = 50.0
    electrode_area: float = 1.0
    ecsa: float = 0.0
    catalyst_loading: float = 0.0
    interpretation: str = ""
    electrolyte: str = "acidic"

    def summary(self) -> str:
        lines = [
            "=" * 64,
            "  CV Analysis Results — EISForge",
            "=" * 64,
        ]
        if self.ir_compensated:
            lines.append(f"  iR Compensation   : APPLIED (R_s = {self.r_s_used:.3f} Ω)")
        else:
            lines.append(f"  iR Compensation   : NOT applied")
        lines += [
            f"  E_onset           = {self.e_onset:.4f} V  ({self.e_onset_method})",
            f"  E_forward_peak    = {self.e_forward_peak:.4f} V",
            f"  E_backward_peak   = {self.e_backward_peak:.4f} V",
            "-" * 64,
            f"  I_f               = {self.i_forward_peak:.4f} mA",
            f"  I_b               = {self.i_backward_peak:.4f} mA",
            f"  I_f/I_b           = {self.if_ib_ratio:.3f}",
            "-" * 64,
            f"  j_f (geometric)   = {self.j_forward_peak:.4f} mA/cm²",
            f"  j_b (geometric)   = {self.j_backward_peak:.4f} mA/cm²",
        ]
        if self.ecsa > 0:
            lines += [
                f"  j_f (ECSA)        = {self.j_specific_forward:.4f} mA/cm²_Pt",
                f"  j_b (ECSA)        = {self.j_specific_backward:.4f} mA/cm²_Pt",
            ]
        lines += [
            f"  Scan rate         = {self.scan_rate} mV/s",
            f"  Electrode area    = {self.electrode_area} cm²",
            "-" * 64,
            f"  Interpretation    : {self.interpretation}",
            "=" * 64,
        ]
        return "\n".join(lines)


class CVAnalyzer:
    """
    Automatic CV analyzer for AOR.

    Parameters
    ----------
    scan_rate : float
        Scan rate in mV/s (default: 50).
    electrode_area : float
        Geometric electrode area in cm² (default: 1.0).
    ecsa : float
        Electrochemically active surface area in cm²_metal (default: 0).
    onset_method : str
        E_onset detection: 'tangent', 'threshold', or 'derivative'.
    electrolyte : str
        'acidic' or 'alkaline'.
    current_unit : str
        Unit of input current: 'mA', 'A', 'uA', 'nA'.
        All currents are internally stored as mA.
    catalyst_loading : float
        Catalyst loading in mg/cm².
    """

    # Conversion factors to mA
    _UNIT_TO_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "μA": 1e-3, "nA": 1e-6}

    def __init__(
        self,
        scan_rate: float = 50.0,
        electrode_area: float = 1.0,
        ecsa: float = 0.0,
        onset_method: str = "tangent",
        onset_threshold: float = 0.05,
        smoothing: bool = True,
        electrolyte: str = "acidic",
        catalyst_loading: float = 0.0,
        current_unit: str = "mA",
    ) -> None:
        self.scan_rate        = scan_rate
        self.electrode_area   = max(electrode_area, 1e-10)
        self.ecsa             = ecsa
        self.onset_method     = onset_method
        self.onset_threshold  = onset_threshold
        self.smoothing        = smoothing
        self.electrolyte      = electrolyte
        self.catalyst_loading = catalyst_loading
        self.current_unit     = current_unit
        self._unit_factor     = self._UNIT_TO_MA.get(current_unit, 1.0)

    # ── iR Compensation ───────────────────────────────────────────────────────

    @staticmethod
    def apply_ir_compensation(
        potential: np.ndarray,
        current_ma: np.ndarray,
        r_s_ohms: float,
    ) -> np.ndarray:
        """
        Apply iR compensation to potential array.

        Formula:
            E_corrected = E_measured - I(A) × R_s(Ω)

        Parameters
        ----------
        potential : np.ndarray
            Measured potential (V).
        current_ma : np.ndarray
            Current in mA — converted internally to A.
        r_s_ohms : float
            Solution resistance in Ohms (R0 from EIS fit).

        Returns
        -------
        np.ndarray
            iR-corrected potential (V).

        Notes
        -----
        R_s should come from the high-frequency intercept of the EIS Nyquist plot
        (R0 parameter in CNLS fit). Always verify that R0 from EIS was measured
        in the same electrolyte and at a similar potential.
        """
        if r_s_ohms <= 0:
            return potential.copy()

        # Current: mA → A for correct V = IR calculation
        current_a  = current_ma * 1e-3
        ir_drop    = current_a * r_s_ohms          # V
        return potential - ir_drop

    # ── Main analyze method ───────────────────────────────────────────────────

    def analyze(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        r_s_ohms: float = 0.0,
    ) -> CVAnalysisResult:
        """
        Perform complete CV analysis.

        Parameters
        ----------
        potential : np.ndarray
            Potential array (V).
        current : np.ndarray
            Current array (in units specified by current_unit, default mA).
        r_s_ohms : float
            Solution resistance for iR compensation (Ω).
            Set to 0 to skip iR compensation (default).
            Obtain R_s from EIS fit (R0 parameter).

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

        # Convert current to mA
        current_ma = current * self._unit_factor

        # Apply iR compensation if R_s provided
        ir_compensated = r_s_ohms > 0
        if ir_compensated:
            potential = self.apply_ir_compensation(potential, current_ma, r_s_ohms)
            logger.info(f"iR compensation applied: R_s = {r_s_ohms:.3f} Ω")

        # Smooth current
        if self.smoothing:
            current_ma = self._smooth(current_ma)

        # Split forward / backward scan
        fwd_mask, bwd_mask = self._split_scans(potential, current_ma)
        e_fwd, i_fwd = potential[fwd_mask], current_ma[fwd_mask]
        e_bwd, i_bwd = potential[bwd_mask], current_ma[bwd_mask]

        if len(e_fwd) < 3 or len(e_bwd) < 3:
            mid = len(potential) // 2
            e_fwd, i_fwd = potential[:mid], current_ma[:mid]
            e_bwd, i_bwd = potential[mid:], current_ma[mid:]
            logger.warning("Scan direction auto-detected: using midpoint split.")

        # Peaks
        i_f = float(i_fwd[np.argmax(i_fwd)])
        e_f = float(e_fwd[np.argmax(i_fwd)])
        i_b = float(i_bwd[np.argmax(i_bwd)])
        e_b = float(e_bwd[np.argmax(i_bwd)])

        # E_onset
        e_onset, baseline = self._detect_onset(e_fwd, i_fwd, i_f)

        # I_f/I_b ratio
        ratio = i_f / i_b if i_b > 1e-10 else float("nan")

        # Current density
        j_f      = i_f / self.electrode_area
        j_b      = i_b / self.electrode_area
        j_spec_f = i_f / self.ecsa if self.ecsa > 0 else 0.0
        j_spec_b = i_b / self.ecsa if self.ecsa > 0 else 0.0

        return CVAnalysisResult(
            e_onset=e_onset,
            e_onset_method=self.onset_method + (" (iR-corrected)" if ir_compensated else ""),
            e_forward_peak=e_f,
            e_backward_peak=e_b,
            i_forward_peak=i_f,
            i_backward_peak=i_b,
            if_ib_ratio=ratio,
            baseline_current=baseline,
            j_forward_peak=j_f,
            j_backward_peak=j_b,
            j_specific_forward=j_spec_f,
            j_specific_backward=j_spec_b,
            ir_compensated=ir_compensated,
            r_s_used=r_s_ohms,
            scan_rate=self.scan_rate,
            electrode_area=self.electrode_area,
            ecsa=self.ecsa,
            catalyst_loading=self.catalyst_loading,
            interpretation=self._interpret(e_onset, i_f, i_b, ratio),
            electrolyte=self.electrolyte,
        )

    # ── Scan splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split_scans(
        potential: np.ndarray,
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n       = len(potential)
        peak    = int(np.argmax(potential))
        if peak < 2:      peak = n // 2
        if peak > n - 3:  peak = n // 2
        fwd = np.zeros(n, dtype=bool)
        bwd = np.zeros(n, dtype=bool)
        fwd[:peak + 1] = True
        bwd[peak + 1:] = True
        return fwd, bwd

    # ── E_onset detection ─────────────────────────────────────────────────────

    def _detect_onset(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        i_peak: float,
    ) -> tuple[float, float]:
        if self.onset_method == "tangent":
            return self._onset_tangent(potential, current, i_peak)
        elif self.onset_method == "threshold":
            return self._onset_threshold(potential, current, i_peak)
        else:
            return self._onset_derivative(potential, current)

    def _onset_tangent(self, potential, current, i_peak):
        n       = len(potential)
        bl_end  = max(int(n * 0.2), 3)
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
            d2 = np.gradient(np.gradient(current, potential), potential)
            peak = int(np.argmax(current))
            pre  = d2[:peak]
            if len(pre) == 0:
                return float(potential[0]), float(current[0])
            idx = int(np.argmax(pre))
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

    def _interpret(self, e_onset, i_f, i_b, ratio) -> str:
        parts = []
        if np.isnan(ratio):
            parts.append("I_f/I_b: not calculable")
        elif ratio > 2.0:
            parts.append(f"I_f/I_b={ratio:.2f} — Excellent CO tolerance")
        elif ratio > 1.0:
            parts.append(f"I_f/I_b={ratio:.2f} — Good CO tolerance")
        elif ratio > 0.5:
            parts.append(f"I_f/I_b={ratio:.2f} — Moderate CO poisoning")
        else:
            parts.append(f"I_f/I_b={ratio:.2f} — Severe CO poisoning")

        lo, hi = (0.4, 0.6) if self.electrolyte == "acidic" else (0.2, 0.4)
        if e_onset < lo:   parts.append(f"E_onset={e_onset:.3f}V — Excellent activity")
        elif e_onset < hi: parts.append(f"E_onset={e_onset:.3f}V — Moderate activity")
        else:              parts.append(f"E_onset={e_onset:.3f}V — High overpotential")
        return " | ".join(parts)

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        for enc in ["latin-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(filepath, comment="#", encoding=enc,
                                 sep=None, engine="python")
                c = df.columns.tolist()
                return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read file: {filepath}")
