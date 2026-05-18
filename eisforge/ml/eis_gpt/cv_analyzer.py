"""
CV Analyzer — تحلیل خودکار ولتامتری چرخه‌ای (Cyclic Voltammetry).

نویسنده: Hoda Jafari
تاریخ: May 2026

زمینه علمی:
-----------
در AOR (Alcohol Oxidation Reaction)، CV سه پارامتر کلیدی دارد:

۱. E_onset (پتانسیل شروع واکنش):
   اولین پتانسیلی که جریان اکسیداسیون از baseline جدا می‌شود.
   تعریف‌های مختلف در ادبیات:
     - روش مماس (Tangent method): رایج‌ترین روش
     - روش threshold: جریان از baseline بیشتر از X% شود
     - روش دوم مشتق: نقطه inflection جریان

۲. I_f (Forward peak current):
   جریان پیک در scan رفت (اکسیداسیون مستقیم الکل)
   مثلاً برای اتانول روی Pt: CH3CH2OH → CO2 + H+ + e-

۳. I_b (Backward peak current):  
   جریان پیک در scan برگشت (اکسیداسیون CO_ads)
   CO_ads + OH → CO2 + H+ + e-

۴. I_f/I_b ratio:
   معیار مقاومت کاتالیست به مسمومیت CO
   I_f/I_b > 1 → کاتالیست مقاوم به CO ✅
   I_f/I_b < 1 → کاتالیست مسموم می‌شود ❌

مرجع:
    Zhao et al., J. Power Sources (2019)
    Antolini, J. Power Sources (2007)
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
    """
    نتایج تحلیل کامل CV.

    Attributes
    ----------
    e_onset : float
        پتانسیل شروع واکنش (V vs RHE یا vs Ref).
    e_onset_method : str
        روش تشخیص E_onset.
    i_forward_peak : float
        جریان پیک رفت I_f (mA یا mA/cm²).
    e_forward_peak : float
        پتانسیل پیک رفت (V).
    i_backward_peak : float
        جریان پیک برگشت I_b (mA یا mA/cm²).
    e_backward_peak : float
        پتانسیل پیک برگشت (V).
    if_ib_ratio : float
        نسبت I_f/I_b — معیار مقاومت به CO.
    baseline_current : float
        جریان baseline در E_onset (mA).
    scan_rate : float
        سرعت scan (mV/s) اگر موجود باشد.
    interpretation : str
        تفسیر خودکار نتایج.
    """

    e_onset: float
    e_onset_method: str
    i_forward_peak: float
    e_forward_peak: float
    i_backward_peak: float
    e_backward_peak: float
    if_ib_ratio: float
    baseline_current: float
    scan_rate: float = 0.0
    interpretation: str = ""

    def summary(self) -> str:
        """خلاصه کامل نتایج."""
        lines = [
            "═" * 55,
            "  📊 نتایج تحلیل CV — EISForge",
            "═" * 55,
            f"  E_onset    = {self.e_onset:.4f} V  ({self.e_onset_method})",
            f"  I_f (پیک رفت)    = {self.i_forward_peak:.4f} mA  @ {self.e_forward_peak:.4f} V",
            f"  I_b (پیک برگشت)  = {self.i_backward_peak:.4f} mA  @ {self.e_backward_peak:.4f} V",
            f"  I_f/I_b ratio    = {self.if_ib_ratio:.3f}",
            "─" * 55,
            f"  تفسیر: {self.interpretation}",
            "═" * 55,
        ]
        return "\n".join(lines)


class CVAnalyzer:
    """
    تحلیلگر خودکار ولتامتری چرخه‌ای برای AOR.

    Parameters
    ----------
    onset_method : str
        روش تشخیص E_onset:
        - 'tangent'    : روش مماس (پیش‌فرض، رایج‌ترین در ادبیات)
        - 'threshold'  : آستانه جریان
        - 'derivative' : دوم مشتق
    onset_threshold : float
        آستانه جریان برای روش threshold
        (کسری از I_f_peak، پیش‌فرض: 0.05 = 5%).
    smoothing_window : int
        پنجره smoothing برای حذف نویز (پیش‌فرض: 11 نقطه).
    electrolyte : str
        نوع الکترولیت: 'acidic' یا 'alkaline'.
        تأثیر بر تفسیر I_f/I_b ratio دارد.
    """

    def __init__(
        self,
        onset_method: str = "tangent",
        onset_threshold: float = 0.05,
        smoothing_window: int = 11,
        electrolyte: str = "acidic",
    ) -> None:
        if onset_method not in ("tangent", "threshold", "derivative"):
            raise ValueError(
                f"onset_method نامعتبر: {onset_method}. "
                "گزینه‌ها: 'tangent', 'threshold', 'derivative'"
            )
        self.onset_method = onset_method
        self.onset_threshold = onset_threshold
        self.smoothing_window = smoothing_window
        self.electrolyte = electrolyte

    def analyze(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        scan_rate: float = 50.0,
    ) -> CVAnalysisResult:
        """
        تحلیل کامل CV.

        Parameters
        ----------
        potential : np.ndarray
            آرایه پتانسیل (V). باید شامل هر دو scan رفت و برگشت باشد.
        current : np.ndarray
            آرایه جریان (mA یا mA/cm²).
        scan_rate : float
            سرعت scan در mV/s (برای اطلاعات، تأثیر محاسباتی ندارد).

        Returns
        -------
        CVAnalysisResult
            نتایج کامل تحلیل.

        Raises
        ------
        ValueError
            اگر داده‌ها کافی نباشند.
        """
        if len(potential) < 20:
            raise ValueError(
                f"داده‌های CV خیلی کم هستند: {len(potential)} نقطه. "
                "حداقل ۲۰ نقطه لازم است."
            )

        # ── Smoothing برای حذف نویز ──────────────────────────────────────────
        current_smooth = self._smooth(current)

        # ── تفکیک scan رفت و برگشت ───────────────────────────────────────────
        forward_mask, backward_mask = self._split_forward_backward(potential)

        e_fwd = potential[forward_mask]
        i_fwd = current_smooth[forward_mask]
        e_bwd = potential[backward_mask]
        i_bwd = current_smooth[backward_mask]

        if len(e_fwd) < 5 or len(e_bwd) < 5:
            raise ValueError(
                "نمی‌توان scan رفت و برگشت را تفکیک کرد. "
                "مطمئن شوید CV شامل یک سیکل کامل است."
            )

        # ── پیک رفت (I_f) ────────────────────────────────────────────────────
        i_f_peak_idx = np.argmax(i_fwd)
        i_forward_peak = float(i_fwd[i_f_peak_idx])
        e_forward_peak = float(e_fwd[i_f_peak_idx])

        # ── پیک برگشت (I_b) ──────────────────────────────────────────────────
        i_b_peak_idx = np.argmax(i_bwd)
        i_backward_peak = float(i_bwd[i_b_peak_idx])
        e_backward_peak = float(e_bwd[i_b_peak_idx])

        # ── E_onset ───────────────────────────────────────────────────────────
        e_onset, baseline_current = self._detect_onset(
            e_fwd, i_fwd, i_forward_peak
        )

        # ── I_f/I_b ratio ─────────────────────────────────────────────────────
        if_ib_ratio = (
            i_forward_peak / i_backward_peak
            if i_backward_peak > 0
            else float("nan")
        )

        # ── تفسیر خودکار ─────────────────────────────────────────────────────
        interpretation = self._interpret(
            e_onset, i_forward_peak, i_backward_peak, if_ib_ratio
        )

        result = CVAnalysisResult(
            e_onset=e_onset,
            e_onset_method=self.onset_method,
            i_forward_peak=i_forward_peak,
            e_forward_peak=e_forward_peak,
            i_backward_peak=i_backward_peak,
            e_backward_peak=e_backward_peak,
            if_ib_ratio=if_ib_ratio,
            baseline_current=baseline_current,
            scan_rate=scan_rate,
            interpretation=interpretation,
        )

        logger.info("CV تحلیل شد: E_onset=%.4f V, I_f/I_b=%.3f",
                    e_onset, if_ib_ratio)
        return result

    def _smooth(self, current: np.ndarray) -> np.ndarray:
        """
        Smoothing جریان با Savitzky-Golay filter.

        Savitzky-Golay بهتر از moving average است چون
        شکل پیک‌ها را حفظ می‌کند.
        """
        window = min(self.smoothing_window, len(current) // 3)
        if window % 2 == 0:
            window += 1
        if window < 5:
            return current.copy()
        try:
            return savgol_filter(current, window_length=window, polyorder=3)
        except Exception:
            return gaussian_filter1d(current, sigma=2)

    @staticmethod
    def _split_forward_backward(
        potential: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        تفکیک scan رفت (افزایش پتانسیل) از برگشت (کاهش پتانسیل).

        Returns
        -------
        forward_mask, backward_mask : np.ndarray of bool
        """
        dp = np.diff(potential)
        peak_idx = np.argmax(potential)

        forward_mask  = np.zeros(len(potential), dtype=bool)
        backward_mask = np.zeros(len(potential), dtype=bool)
        forward_mask[:peak_idx + 1]  = True
        backward_mask[peak_idx + 1:] = True

        return forward_mask, backward_mask

    def _detect_onset(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        i_peak: float,
    ) -> tuple[float, float]:
        """
        تشخیص E_onset با روش انتخاب‌شده.

        Returns
        -------
        e_onset : float
        baseline_current : float
        """
        if self.onset_method == "tangent":
            return self._onset_tangent(potential, current, i_peak)
        elif self.onset_method == "threshold":
            return self._onset_threshold(potential, current, i_peak)
        else:
            return self._onset_derivative(potential, current)

    def _onset_tangent(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        i_peak: float,
    ) -> tuple[float, float]:
        """
        روش مماس برای تشخیص E_onset.

        الگوریتم:
        1. baseline را از ابتدای scan (قبل از واکنش) تخمین بزن
        2. مماس را روی شیب تند قبل از پیک رسم کن
        3. تقاطع این دو خط = E_onset
        """
        n = len(potential)

        # ── تخمین baseline (20% اول scan) ─────────────────────────────────
        baseline_end = max(int(n * 0.2), 3)
        baseline_slope, baseline_intercept = np.polyfit(
            potential[:baseline_end], current[:baseline_end], 1
        )
        baseline_current_at_onset = float(
            np.mean(current[:baseline_end])
        )

        # ── پیدا کردن ناحیه شیب تند (60-90% از مسیر به پیک) ──────────────
        peak_idx = np.argmax(current)
        slope_start = int(peak_idx * 0.6)
        slope_end   = int(peak_idx * 0.9)

        if slope_end <= slope_start + 2:
            slope_start = max(0, peak_idx - 5)
            slope_end   = peak_idx

        rise_slope, rise_intercept = np.polyfit(
            potential[slope_start:slope_end],
            current[slope_start:slope_end],
            1,
        )

        # ── تقاطع دو خط ───────────────────────────────────────────────────
        # baseline: i = m1·E + b1
        # rise:     i = m2·E + b2
        # E_onset = (b1 - b2) / (m2 - m1)
        denom = rise_slope - baseline_slope
        if abs(denom) < 1e-10:
            # اگر موازی هستند، از روش threshold استفاده کن
            return self._onset_threshold(potential, current, i_peak)

        e_onset = (baseline_intercept - rise_intercept) / denom
        e_onset = float(np.clip(e_onset, potential.min(), potential.max()))

        return e_onset, baseline_current_at_onset

    def _onset_threshold(
        self,
        potential: np.ndarray,
        current: np.ndarray,
        i_peak: float,
    ) -> tuple[float, float]:
        """
        روش آستانه: اولین نقطه‌ای که جریان از baseline بیشتر از X% پیک شود.
        """
        baseline = float(np.mean(current[:max(int(len(current) * 0.15), 3)]))
        threshold_current = baseline + self.onset_threshold * i_peak

        above_threshold = np.where(current > threshold_current)[0]
        if len(above_threshold) == 0:
            return float(potential[len(potential) // 2]), baseline

        onset_idx = int(above_threshold[0])
        return float(potential[onset_idx]), baseline

    def _onset_derivative(
        self,
        potential: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """
        روش دوم مشتق: نقطه inflection جریان = E_onset.
        """
        if len(current) < 5:
            return self._onset_threshold(
                potential, current, float(np.max(current))
            )

        d2i = np.gradient(np.gradient(current, potential), potential)
        peak_idx = np.argmax(current)
        d2i_before_peak = d2i[:peak_idx]

        if len(d2i_before_peak) == 0:
            return float(potential[0]), float(current[0])

        onset_idx = int(np.argmax(d2i_before_peak))
        baseline = float(np.mean(current[:max(onset_idx, 1)]))
        return float(potential[onset_idx]), baseline

    def _interpret(
        self,
        e_onset: float,
        i_f: float,
        i_b: float,
        ratio: float,
    ) -> str:
        """تفسیر خودکار نتایج بر اساس ادبیات AOR."""
        lines = []

        # تفسیر I_f/I_b
        if np.isnan(ratio):
            lines.append("⚠️  نسبت I_f/I_b قابل محاسبه نیست.")
        elif ratio > 2.0:
            lines.append(
                f"✅ I_f/I_b = {ratio:.2f} — کاتالیست عالی: "
                "مقاومت بسیار بالا به مسمومیت CO."
            )
        elif ratio > 1.0:
            lines.append(
                f"✅ I_f/I_b = {ratio:.2f} — کاتالیست خوب: "
                "مقاومت مناسب به CO."
            )
        elif ratio > 0.5:
            lines.append(
                f"⚠️  I_f/I_b = {ratio:.2f} — کاتالیست متوسط: "
                "مسمومیت CO وجود دارد."
            )
        else:
            lines.append(
                f"❌ I_f/I_b = {ratio:.2f} — کاتالیست ضعیف: "
                "مسمومیت شدید CO — Pt خالص در محیط اسیدی."
            )

        # تفسیر E_onset
        if self.electrolyte == "acidic":
            if e_onset < 0.4:
                lines.append(
                    f"✅ E_onset = {e_onset:.3f} V — عالی برای محیط اسیدی."
                )
            elif e_onset < 0.6:
                lines.append(
                    f"⚠️  E_onset = {e_onset:.3f} V — متوسط برای محیط اسیدی."
                )
            else:
                lines.append(
                    f"❌ E_onset = {e_onset:.3f} V — بالا — "
                    "overpotential زیاد در محیط اسیدی."
                )
        else:  # alkaline
            if e_onset < 0.3:
                lines.append(
                    f"✅ E_onset = {e_onset:.3f} V — عالی برای محیط بازی."
                )
            elif e_onset < 0.5:
                lines.append(
                    f"⚠️  E_onset = {e_onset:.3f} V — متوسط برای محیط بازی."
                )
            else:
                lines.append(
                    f"❌ E_onset = {e_onset:.3f} V — بالا برای محیط بازی."
                )

        return " | ".join(lines)

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        """
        بارگذاری داده CV از CSV.

        فرمت مورد انتظار:
            ستون اول: potential (V)
            ستون دوم: current (mA)
        """
        df = pd.read_csv(filepath, comment="#")
        cols = df.columns.tolist()
        return df[cols[0]].to_numpy(), df[cols[1]].to_numpy()
