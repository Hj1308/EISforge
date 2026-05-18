"""
LSV Analyzer — تحلیل خودکار Linear Sweep Voltammetry برای AOR.

نویسنده: Hoda Jafari | May 2026

پارامترهای استخراج‌شده:
    ۱. E_onset     — پتانسیل شروع واکنش (دقیق‌تر از CV)
    ۲. Tafel slope — مکانیزم واکنش (b = 2.303RT/αnF)
    ۳. Exchange current density (j₀) — از Tafel extrapolation
    ۴. Overpotential در جریان‌های خاص (η@10، η@50 mA/cm²)
    ۵. Limiting current density (j_lim)
    ۶. Half-wave potential (E₁/₂)
    ۷. Mass activity و Specific activity

زمینه علمی Tafel slope برای AOR:
    b ≈ 30 mV/dec  → مرحله شیمیایی محدودکننده (Heyrovsky)
    b ≈ 60 mV/dec  → جذب الکل محدودکننده (Langmuir)
    b ≈ 120 mV/dec → انتقال الکترون اول محدودکننده (Volmer/Butler-Volmer)
    b > 120 mV/dec → مسمومیت CO یا انتشار محدودکننده

مرجع:
    Bard & Faulkner, Electrochemical Methods (2001) Ch. 3
    Lamy et al., Electrochim. Acta 47 (2002) 3701
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ── ثابت‌های فیزیکی ──────────────────────────────────────────────────────────
FARADAY    = 96485.33   # C/mol
GAS_CONST  = 8.31446    # J/mol/K
TEMP_298K  = 298.15     # K
THERMAL_V  = GAS_CONST * TEMP_298K / FARADAY  # ≈ 0.02569 V


@dataclass
class LSVAnalysisResult:
    """نتایج تحلیل کامل LSV برای AOR."""

    # ── E_onset ────────────────────────────────────────────────────
    e_onset: float                    # V
    e_onset_method: str

    # ── Tafel Analysis ─────────────────────────────────────────────
    tafel_slope: float                # mV/dec
    tafel_slope_std: float            # خطای standard
    exchange_current_density: float   # j₀ (mA/cm²)
    tafel_r_squared: float            # R² برازش Tafel
    tafel_region: tuple               # (E_start, E_end) ناحیه Tafel

    # ── Overpotential ──────────────────────────────────────────────
    overpotential_10:  float          # η @ j=10 mA/cm²  (V)
    overpotential_50:  float          # η @ j=50 mA/cm²  (V)
    overpotential_100: float          # η @ j=100 mA/cm² (V)

    # ── جریان‌های مرجع ─────────────────────────────────────────────
    j_at_onset:    float              # mA/cm²
    j_limiting:    float              # mA/cm² — جریان حدی
    e_half_wave:   float              # E₁/₂ (V)

    # ── فعالیت کاتالیست ────────────────────────────────────────────
    mass_activity:     float          # mA/mg_catalyst
    specific_activity: float          # mA/cm²_ECSA

    # ── پارامترهای اندازه‌گیری ─────────────────────────────────────
    scan_rate:        float = 5.0     # mV/s (LSV معمولاً کندتر از CV)
    electrode_area:   float = 1.0     # cm²
    ecsa:             float = 0.0     # cm²_Pt
    catalyst_loading: float = 0.0     # mg/cm²
    electrolyte:      str   = "acidic"

    # ── تفسیر ──────────────────────────────────────────────────────
    mechanism_interpretation: str = ""
    performance_rating: str = ""

    def summary(self) -> str:
        lines = [
            "═" * 65,
            "  📊 نتایج تحلیل LSV — EISForge",
            "═" * 65,
            "",
            "  ── E_onset ─────────────────────────────────────────",
            f"  E_onset          = {self.e_onset:.4f} V  ({self.e_onset_method})",
            "",
            "  ── Tafel Analysis ───────────────────────────────────",
            f"  Tafel slope      = {self.tafel_slope:.1f} ± {self.tafel_slope_std:.1f} mV/dec",
            f"  j₀               = {self.exchange_current_density:.4e} mA/cm²",
            f"  R² (Tafel fit)   = {self.tafel_r_squared:.4f}",
            f"  Tafel region     = [{self.tafel_region[0]:.3f}, {self.tafel_region[1]:.3f}] V",
            "",
            "  ── Overpotential ────────────────────────────────────",
            f"  η @ 10  mA/cm²   = {self.overpotential_10*1000:.1f} mV",
            f"  η @ 50  mA/cm²   = {self.overpotential_50*1000:.1f} mV",
            f"  η @ 100 mA/cm²   = {self.overpotential_100*1000:.1f} mV",
            "",
            "  ── فعالیت ───────────────────────────────────────────",
            f"  j_limiting       = {self.j_limiting:.3f} mA/cm²",
            f"  E₁/₂             = {self.e_half_wave:.4f} V",
            f"  Mass activity    = {self.mass_activity:.3f} mA/mg",
        ]
        if self.ecsa > 0:
            lines.append(
                f"  Specific activity= {self.specific_activity:.4f} mA/cm²_Pt"
            )
        lines += [
            "",
            "  ── پارامترهای اندازه‌گیری ───────────────────────────",
            f"  Scan rate        = {self.scan_rate} mV/s",
            f"  Electrode area   = {self.electrode_area} cm²",
        ]
        if self.ecsa > 0:
            lines.append(f"  ECSA             = {self.ecsa:.4f} cm²_Pt")
        if self.catalyst_loading > 0:
            lines.append(f"  Loading          = {self.catalyst_loading} mg/cm²")
        lines += [
            "",
            "  ── تفسیر مکانیزم ────────────────────────────────────",
            f"  {self.mechanism_interpretation}",
            "",
            "  ── ارزیابی عملکرد ───────────────────────────────────",
            f"  {self.performance_rating}",
            "═" * 65,
        ]
        return "\n".join(lines)


class LSVAnalyzer:
    """
    تحلیلگر خودکار LSV برای AOR.

    Parameters
    ----------
    scan_rate : float
        سرعت scan در mV/s (پیش‌فرض: 5 mV/s برای LSV).
        LSV معمولاً از 1-10 mV/s است (کندتر از CV).
    electrode_area : float
        مساحت هندسی الکترود (cm²).
    ecsa : float
        ECSA در cm²_Pt — برای Specific activity.
    catalyst_loading : float
        بارگذاری کاتالیست در mg/cm² — برای Mass activity.
    electrolyte : str
        'acidic' یا 'alkaline'.
    e_ref_vs_rhe : float
        تبدیل پتانسیل مرجع به RHE (V).
        مثلاً برای Ag/AgCl در H2SO4 0.5M: +0.197V
        اگر داده‌ها از قبل vs RHE هستند: 0.0
    tafel_current_range : tuple
        بازه جریان برای Tafel fit به صورت (j_min, j_max) mA/cm².
        پیش‌فرض: (0.1, 2.0) — ناحیه kinetic control.
    """

    def __init__(
        self,
        scan_rate: float = 5.0,
        electrode_area: float = 1.0,
        ecsa: float = 0.0,
        catalyst_loading: float = 0.0,
        electrolyte: str = "acidic",
        e_ref_vs_rhe: float = 0.0,
        tafel_current_range: tuple = (0.1, 2.0),
        smoothing_window: int = 11,
    ) -> None:
        self.scan_rate          = scan_rate
        self.electrode_area     = max(electrode_area, 1e-10)
        self.ecsa               = ecsa
        self.catalyst_loading   = catalyst_loading
        self.electrolyte        = electrolyte
        self.e_ref_vs_rhe       = e_ref_vs_rhe
        self.tafel_current_range = tafel_current_range
        self.smoothing_window   = smoothing_window

    def analyze(
        self,
        potential: np.ndarray,
        current: np.ndarray,
    ) -> LSVAnalysisResult:
        """
        تحلیل کامل LSV.

        Parameters
        ----------
        potential : np.ndarray
            پتانسیل (V) — فقط scan رفت (افزایشی).
        current : np.ndarray
            جریان (mA) — جریان مطلق.

        Returns
        -------
        LSVAnalysisResult
        """
        if len(potential) < 20:
            raise ValueError(f"داده کم: {len(potential)} نقطه.")

        # ── تبدیل به RHE ──────────────────────────────────────────────────────
        potential = potential + self.e_ref_vs_rhe

        # ── Smooth و نرمال‌سازی ───────────────────────────────────────────────
        current_smooth = self._smooth(current)
        j = current_smooth / self.electrode_area   # mA/cm²

        # ── مطمئن شویم scan صعودی است ────────────────────────────────────────
        if potential[-1] < potential[0]:
            potential = potential[::-1]
            j = j[::-1]

        # ── E_onset ───────────────────────────────────────────────────────────
        e_onset, onset_method = self._detect_onset_lsv(potential, j)

        # ── Tafel Analysis ────────────────────────────────────────────────────
        tafel = self._tafel_analysis(potential, j, e_onset)

        # ── Overpotential ─────────────────────────────────────────────────────
        eta_10  = self._overpotential_at_j(potential, j, 10.0,  e_onset)
        eta_50  = self._overpotential_at_j(potential, j, 50.0,  e_onset)
        eta_100 = self._overpotential_at_j(potential, j, 100.0, e_onset)

        # ── جریان حدی و E₁/₂ ─────────────────────────────────────────────────
        j_lim   = float(np.max(j))
        e_half  = self._half_wave_potential(potential, j, j_lim)

        # ── فعالیت ────────────────────────────────────────────────────────────
        j_at_onset = float(np.interp(e_onset, potential, j))

        # Mass activity: j_f در E_onset / loading
        mass_act = (
            j_at_onset / self.catalyst_loading
            if self.catalyst_loading > 0 else 0.0
        )

        # Specific activity: j / ECSA (نرمال‌شده با ECSA)
        spec_act = (
            j_at_onset * self.electrode_area / self.ecsa
            if self.ecsa > 0 else 0.0
        )

        # ── تفسیر ─────────────────────────────────────────────────────────────
        mechanism = self._interpret_tafel(tafel["slope"])
        performance = self._rate_performance(
            e_onset, tafel["slope"], eta_10, self.electrolyte
        )

        return LSVAnalysisResult(
            e_onset=e_onset,
            e_onset_method=onset_method,
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
            scan_rate=self.scan_rate,
            electrode_area=self.electrode_area,
            ecsa=self.ecsa,
            catalyst_loading=self.catalyst_loading,
            electrolyte=self.electrolyte,
            mechanism_interpretation=mechanism,
            performance_rating=performance,
        )

    # ── E_onset ───────────────────────────────────────────────────────────────

    def _detect_onset_lsv(
        self, potential: np.ndarray, j: np.ndarray
    ) -> tuple[float, str]:
        """
        تشخیص E_onset از LSV با روش مماس.

        در LSV چون فقط scan رفت داریم، روش مماس دقیق‌تر است.
        الگوریتم:
            1. baseline از ابتدای scan
            2. مماس روی شیب تند اولیه
            3. تقاطع = E_onset
        """
        n = len(potential)
        bl_end = max(int(n * 0.15), 5)

        # baseline
        m_bl, b_bl = np.polyfit(potential[:bl_end], j[:bl_end], 1)
        baseline_j = float(np.mean(j[:bl_end]))

        # پیدا کردن ناحیه شیب تند
        # جایی که j از baseline بیشتر از 5% j_max شده
        j_max = float(np.max(j))
        threshold = baseline_j + 0.05 * j_max
        above = np.where(j > threshold)[0]

        if len(above) == 0:
            return float(potential[n // 2]), "threshold"

        onset_region_start = int(above[0])
        onset_region_end   = min(onset_region_start + int(n * 0.15), n - 1)

        if onset_region_end <= onset_region_start + 2:
            return float(potential[onset_region_start]), "threshold"

        try:
            m_rise, b_rise = np.polyfit(
                potential[onset_region_start:onset_region_end],
                j[onset_region_start:onset_region_end],
                1,
            )
            denom = m_rise - m_bl
            if abs(denom) < 1e-10:
                return float(potential[onset_region_start]), "threshold"

            e_onset = float(np.clip(
                (b_bl - b_rise) / denom,
                potential.min(), potential.max()
            ))
            return e_onset, "tangent (LSV)"

        except Exception:
            return float(potential[onset_region_start]), "threshold"

    # ── Tafel Analysis ────────────────────────────────────────────────────────

    def _tafel_analysis(
        self,
        potential: np.ndarray,
        j: np.ndarray,
        e_onset: float,
    ) -> dict:
        """
        تحلیل Tafel: log(j) vs E در ناحیه kinetic control.

        معادله Tafel:
            η = a + b·log(j)
            b = Tafel slope (mV/dec)
            a = -b·log(j₀)  →  j₀ = 10^(-a/b)

        ناحیه Tafel: بین E_onset و جایی که انتشار شروع می‌شود.
        معمولاً جریان‌های کم (0.1 تا 2 mA/cm²).
        """
        j_min, j_max_tafel = self.tafel_current_range

        # فیلتر: فقط نقاطی که j در بازه Tafel است
        mask = (j >= j_min) & (j <= j_max_tafel) & (j > 0)

        # اطمینان از اینکه بعد از E_onset هستیم
        mask = mask & (potential >= e_onset - 0.05)

        if np.sum(mask) < 5:
            # اگر نقاط کافی نبود، بازه را گسترش بدیم
            mask = (j >= j_min * 0.1) & (j <= j_max_tafel * 5) & (j > 0)
            mask = mask & (potential >= e_onset - 0.1)

        if np.sum(mask) < 3:
            return {
                "slope": float("nan"), "slope_std": float("nan"),
                "j0": float("nan"), "r2": 0.0,
                "region": (float(e_onset), float(e_onset)),
            }

        E_tafel = potential[mask]
        j_tafel = j[mask]

        # Tafel plot: E vs log10(j)
        log_j = np.log10(j_tafel)

        try:
            slope_reg, intercept, r_val, _, slope_std = linregress(log_j, E_tafel)
        except Exception:
            return {
                "slope": float("nan"), "slope_std": float("nan"),
                "j0": float("nan"), "r2": 0.0,
                "region": (float(E_tafel[0]), float(E_tafel[-1])),
            }

        # Tafel slope در mV/dec
        # E = a + b·log(j)  →  b = dE/d(log j) در V/dec → ×1000 برای mV/dec
        tafel_slope_mv = slope_reg * 1000.0

        # j₀: وقتی η=0، یعنی E = E_eq (پتانسیل تعادل)
        # برای AOR در محیط اسیدی E_eq ≈ E_onset
        # j₀ = 10^((E_onset - intercept) / slope_reg)
        try:
            j0 = float(10 ** ((e_onset - intercept) / slope_reg))
        except Exception:
            j0 = float("nan")

        return {
            "slope":     tafel_slope_mv,
            "slope_std": slope_std * 1000.0,
            "j0":        j0,
            "r2":        r_val ** 2,
            "region":    (float(E_tafel[0]), float(E_tafel[-1])),
        }

    # ── Overpotential ─────────────────────────────────────────────────────────

    def _overpotential_at_j(
        self,
        potential: np.ndarray,
        j: np.ndarray,
        j_target: float,
        e_onset: float,
    ) -> float:
        """
        Overpotential در یک چگالی جریان خاص.

        η = E(j_target) - E_onset
        """
        if j_target > np.max(j):
            return float("nan")
        e_at_j = float(np.interp(j_target, j, potential))
        return e_at_j - e_onset

    # ── Half-wave potential ────────────────────────────────────────────────────

    @staticmethod
    def _half_wave_potential(
        potential: np.ndarray,
        j: np.ndarray,
        j_lim: float,
    ) -> float:
        """E₁/₂ پتانسیلی که j = j_lim/2."""
        j_half = j_lim / 2.0
        if j_half > np.max(j) or j_half < np.min(j):
            return float("nan")
        return float(np.interp(j_half, j, potential))

    # ── تفسیر ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _interpret_tafel(slope_mv: float) -> str:
        """تفسیر مکانیزم از Tafel slope."""
        if np.isnan(slope_mv):
            return "⚠️ Tafel slope محاسبه نشد"

        abs_slope = abs(slope_mv)

        if abs_slope < 40:
            return (
                f"✅ Tafel slope = {slope_mv:.1f} mV/dec\n"
                "   مرحله شیمیایی محدودکننده (chemical step)\n"
                "   واکنش جذب/دفع واسطه سریع است\n"
                "   n_electrons > 1 در مرحله محدودکننده"
            )
        elif abs_slope < 75:
            return (
                f"✅ Tafel slope = {slope_mv:.1f} mV/dec\n"
                "   جذب الکل روی سطح کاتالیست محدودکننده (Langmuir)\n"
                "   α ≈ 0.5 — انتقال الکترون متقارن\n"
                "   رایج برای PtRu، PdAu در AOR"
            )
        elif abs_slope < 90:
            return (
                f"⚠️ Tafel slope = {slope_mv:.1f} mV/dec\n"
                "   انتقال الکترون اول محدودکننده\n"
                "   α ≈ 0.5 — Butler-Volmer\n"
                "   رایج برای Pt در اتانول"
            )
        elif abs_slope < 150:
            return (
                f"⚠️ Tafel slope = {slope_mv:.1f} mV/dec\n"
                "   انتقال الکترون اول محدودکننده (Volmer)\n"
                "   α ≈ 0.5، سطح کم پوشیده\n"
                "   مرحله اول کند است"
            )
        else:
            return (
                f"❌ Tafel slope = {slope_mv:.1f} mV/dec — بالا!\n"
                "   احتمال: مسمومیت CO یا انتشار محدودکننده\n"
                "   یا سطح کاتالیست غیرفعال شده\n"
                "   EIS بگیرید تا مکانیزم را تأیید کنید"
            )

    @staticmethod
    def _rate_performance(
        e_onset: float,
        tafel_slope: float,
        eta_10: float,
        electrolyte: str,
    ) -> str:
        """ارزیابی کلی عملکرد کاتالیست."""
        score = 0

        # E_onset
        thresholds = {"acidic": (0.4, 0.6), "alkaline": (0.2, 0.4)}
        lo, hi = thresholds.get(electrolyte, (0.4, 0.6))
        if e_onset < lo:
            score += 3
        elif e_onset < hi:
            score += 1

        # Tafel slope
        if not np.isnan(tafel_slope):
            if abs(tafel_slope) < 60:
                score += 3
            elif abs(tafel_slope) < 90:
                score += 2
            elif abs(tafel_slope) < 120:
                score += 1

        # Overpotential @ 10 mA/cm²
        if not np.isnan(eta_10):
            if eta_10 < 0.1:
                score += 3
            elif eta_10 < 0.2:
                score += 2
            elif eta_10 < 0.3:
                score += 1

        if score >= 8:
            return "🏆 عملکرد عالی — قابل انتشار در ژورنال‌های IF بالا"
        elif score >= 5:
            return "✅ عملکرد خوب — کاتالیست مناسب"
        elif score >= 3:
            return "⚠️ عملکرد متوسط — نیاز به بهینه‌سازی"
        else:
            return "❌ عملکرد ضعیف — تغییر ترکیب کاتالیست پیشنهاد می‌شود"

    # ── Smoothing ─────────────────────────────────────────────────────────────

    def _smooth(self, current: np.ndarray) -> np.ndarray:
        w = min(self.smoothing_window, len(current) // 3)
        if w % 2 == 0:
            w += 1
        if w < 5:
            return current.copy()
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(current, window_length=w, polyorder=3)
        except Exception:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(current, sigma=2)

    # ── CSV loader ────────────────────────────────────────────────────────────

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        """بارگذاری داده LSV از CSV."""
        df = pd.read_csv(filepath, comment="#")
        cols = df.columns.tolist()
        return df[cols[0]].to_numpy(dtype=float), df[cols[1]].to_numpy(dtype=float)
