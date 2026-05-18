"""
CV Analyzer — تحلیل خودکار ولتامتری چرخه‌ای برای AOR.
نویسنده: Hoda Jafari | May 2026
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


@dataclass
class CVAnalysisResult:
    e_onset: float
    e_onset_method: str
    e_forward_peak: float
    e_backward_peak: float
    i_forward_peak: float
    i_backward_peak: float
    if_ib_ratio: float
    baseline_current: float
    j_forward_peak: float = 0.0       # mA/cm² هندسی
    j_backward_peak: float = 0.0
    j_specific_forward: float = 0.0   # mA/cm²_Pt (ECSA)
    j_specific_backward: float = 0.0
    scan_rate: float = 50.0
    electrode_area: float = 1.0
    ecsa: float = 0.0
    catalyst_loading: float = 0.0
    interpretation: str = ""
    electrolyte: str = "acidic"

    def summary(self) -> str:
        lines = [
            "═" * 62,
            "  📊 نتایج تحلیل CV — EISForge",
            "═" * 62,
            f"  E_onset          = {self.e_onset:.4f} V  ({self.e_onset_method})",
            f"  E_forward_peak   = {self.e_forward_peak:.4f} V",
            f"  E_backward_peak  = {self.e_backward_peak:.4f} V",
            "─" * 62,
            f"  I_f              = {self.i_forward_peak:.4f} mA",
            f"  I_b              = {self.i_backward_peak:.4f} mA",
            f"  I_f/I_b          = {self.if_ib_ratio:.3f}",
            "─" * 62,
            f"  j_f (هندسی)      = {self.j_forward_peak:.4f} mA/cm²",
            f"  j_b (هندسی)      = {self.j_backward_peak:.4f} mA/cm²",
        ]
        if self.ecsa > 0:
            lines += [
                f"  j_f (ECSA)       = {self.j_specific_forward:.4f} mA/cm²_Pt",
                f"  j_b (ECSA)       = {self.j_specific_backward:.4f} mA/cm²_Pt",
                f"  ECSA             = {self.ecsa:.4f} cm²_Pt",
            ]
        lines += [
            f"  سرعت scan        = {self.scan_rate} mV/s",
            f"  مساحت الکترود    = {self.electrode_area} cm²",
            "─" * 62,
            f"  تفسیر: {self.interpretation}",
            "═" * 62,
        ]
        return "\n".join(lines)


class CVAnalyzer:
    """
    تحلیلگر خودکار CV برای AOR.

    Parameters
    ----------
    scan_rate : float         سرعت scan (mV/s)
    electrode_area : float    مساحت هندسی الکترود (cm²)
    ecsa : float              ECSA (cm²_Pt) — اختیاری
    onset_method : str        روش E_onset: tangent/threshold/derivative
    electrolyte : str         acidic یا alkaline
    catalyst_loading : float  بارگذاری کاتالیست (mg/cm²)
    """

    def __init__(
        self,
        scan_rate: float = 50.0,
        electrode_area: float = 1.0,
        ecsa: float = 0.0,
        onset_method: str = "tangent",
        onset_threshold: float = 0.05,
        smoothing_window: int = 11,
        electrolyte: str = "acidic",
        catalyst_loading: float = 0.0,
    ) -> None:
        self.scan_rate        = scan_rate
        self.electrode_area   = max(electrode_area, 1e-10)
        self.ecsa             = ecsa
        self.onset_method     = onset_method
        self.onset_threshold  = onset_threshold
        self.smoothing_window = smoothing_window
        self.electrolyte      = electrolyte
        self.catalyst_loading = catalyst_loading

    def analyze(self, potential: np.ndarray, current: np.ndarray) -> CVAnalysisResult:
        if len(potential) < 10:
            raise ValueError(f"داده کم: {len(potential)} نقطه.")

        current_smooth = self._smooth(current)
        fwd_mask, bwd_mask = self._split_scans(potential)

        e_fwd = potential[fwd_mask];  i_fwd = current_smooth[fwd_mask]
        e_bwd = potential[bwd_mask];  i_bwd = current_smooth[bwd_mask]

        if len(e_fwd) < 3 or len(e_bwd) < 3:
            raise ValueError("نمی‌توان scan رفت/برگشت را تفکیک کرد.")

        i_f_idx = int(np.argmax(i_fwd))
        i_b_idx = int(np.argmax(i_bwd))

        i_f = float(i_fwd[i_f_idx]);  e_f = float(e_fwd[i_f_idx])
        i_b = float(i_bwd[i_b_idx]);  e_b = float(e_bwd[i_b_idx])

        e_onset, baseline = self._detect_onset(e_fwd, i_fwd, i_f)
        ratio = i_f / i_b if i_b > 1e-10 else float("nan")

        # نرمال‌سازی
        j_f      = i_f / self.electrode_area
        j_b      = i_b / self.electrode_area
        j_spec_f = i_f / self.ecsa if self.ecsa > 0 else 0.0
        j_spec_b = i_b / self.ecsa if self.ecsa > 0 else 0.0

        return CVAnalysisResult(
            e_onset=e_onset, e_onset_method=self.onset_method,
            e_forward_peak=e_f, e_backward_peak=e_b,
            i_forward_peak=i_f, i_backward_peak=i_b,
            if_ib_ratio=ratio, baseline_current=baseline,
            j_forward_peak=j_f, j_backward_peak=j_b,
            j_specific_forward=j_spec_f, j_specific_backward=j_spec_b,
            scan_rate=self.scan_rate, electrode_area=self.electrode_area,
            ecsa=self.ecsa, catalyst_loading=self.catalyst_loading,
            interpretation=self._interpret(e_onset, i_f, i_b, ratio),
            electrolyte=self.electrolyte,
        )

    def _smooth(self, current):
        w = min(self.smoothing_window, len(current) // 3)
        if w % 2 == 0: w += 1
        if w < 5: return current.copy()
        try:    return savgol_filter(current, window_length=w, polyorder=3)
        except: return gaussian_filter1d(current, sigma=2)

    @staticmethod
    def _split_scans(potential):
        peak = int(np.argmax(potential))
        fwd = np.zeros(len(potential), dtype=bool)
        bwd = np.zeros(len(potential), dtype=bool)
        fwd[:peak + 1] = True
        bwd[peak + 1:] = True
        return fwd, bwd

    def _detect_onset(self, potential, current, i_peak):
        if self.onset_method == "tangent":    return self._tangent(potential, current, i_peak)
        elif self.onset_method == "threshold": return self._threshold(potential, current, i_peak)
        else:                                  return self._derivative(potential, current)

    def _tangent(self, potential, current, i_peak):
        n = len(potential)
        bl_end = max(int(n * 0.2), 3)
        baseline = float(np.mean(current[:bl_end]))
        m1, b1 = np.polyfit(potential[:bl_end], current[:bl_end], 1)
        peak_idx = int(np.argmax(current))
        s, e = int(peak_idx * 0.6), int(peak_idx * 0.9)
        if e <= s + 2: s, e = max(0, peak_idx - 5), peak_idx
        try:
            m2, b2 = np.polyfit(potential[s:e], current[s:e], 1)
            denom = m2 - m1
            if abs(denom) < 1e-10: return self._threshold(potential, current, i_peak)
            onset = float(np.clip((b1 - b2) / denom, potential.min(), potential.max()))
            return onset, baseline
        except:
            return self._threshold(potential, current, i_peak)

    def _threshold(self, potential, current, i_peak):
        baseline = float(np.mean(current[:max(int(len(current)*0.15), 3)]))
        thresh = baseline + self.onset_threshold * i_peak
        above = np.where(current > thresh)[0]
        if len(above) == 0: return float(potential[len(potential)//2]), baseline
        return float(potential[int(above[0])]), baseline

    def _derivative(self, potential, current):
        if len(current) < 5: return self._threshold(potential, current, float(np.max(current)))
        d2 = np.gradient(np.gradient(current, potential), potential)
        peak = int(np.argmax(current))
        pre  = d2[:peak]
        if len(pre) == 0: return float(potential[0]), float(current[0])
        idx = int(np.argmax(pre))
        return float(potential[idx]), float(np.mean(current[:max(idx, 1)]))

    def _interpret(self, e_onset, i_f, i_b, ratio) -> str:
        parts = []
        if np.isnan(ratio):       parts.append("⚠️ I_f/I_b محاسبه نشد")
        elif ratio > 2.0:         parts.append(f"✅ I_f/I_b={ratio:.2f} مقاومت عالی به CO")
        elif ratio > 1.0:         parts.append(f"✅ I_f/I_b={ratio:.2f} مقاومت خوب به CO")
        elif ratio > 0.5:         parts.append(f"⚠️ I_f/I_b={ratio:.2f} مسمومیت CO متوسط")
        else:                     parts.append(f"❌ I_f/I_b={ratio:.2f} مسمومیت شدید CO")

        thr_lo, thr_hi = (0.4, 0.6) if self.electrolyte == "acidic" else (0.2, 0.4)
        if e_onset < thr_lo:      parts.append(f"✅ E_onset={e_onset:.3f}V عالی")
        elif e_onset < thr_hi:    parts.append(f"⚠️ E_onset={e_onset:.3f}V متوسط")
        else:                     parts.append(f"❌ E_onset={e_onset:.3f}V بالا")
        return " | ".join(parts)

    @staticmethod
    def load_csv(filepath: str):
        df = pd.read_csv(filepath, comment="#")
        cols = df.columns.tolist()
        return df[cols[0]].to_numpy(), df[cols[1]].to_numpy()
