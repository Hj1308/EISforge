"""
EIS-CV Correlator — ارتباط‌دهنده EIS با CV برای AOR.

نویسنده: Hoda Jafari
تاریخ: May 2026

هدف:
----
یکی از بزرگ‌ترین چالش‌های AOR اینه که:
  - CV اطلاعات کینتیکی کلی می‌دهد (E_onset، I_f، I_b)
  - EIS اطلاعات مکانیستی می‌دهد (R_ct، CPE، Warburg)

اما این دو معمولاً جداگانه تحلیل می‌شوند!

این ماژول آن‌ها را به هم وصل می‌کند:

سوال ۱: EIS من در چه ناحیه‌ای از CV گرفته شده؟
  → قبل از E_onset (بدون واکنش)
  → روی E_onset (شروع واکنش)  
  → بعد از E_onset (واکنش کامل)

سوال ۲: R_ct از EIS با I_f از CV چه رابطه‌ای دارد؟
  → R_ct بزرگ = I_f کوچک (درست!)
  → اگر این رابطه برقرار نباشد → مشکل در اندازه‌گیری

سوال ۳: آیا Warburg در EIS با انتشار محدود در CV سازگار است؟
  → peak separation بزرگ در CV = Warburg در EIS
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from eisforge.analysis.cv_analyzer import CVAnalysisResult
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)


@dataclass
class EISCVCorrelationResult:
    """
    نتایج همبستگی EIS و CV.

    Attributes
    ----------
    eis_potential : float
        پتانسیلی که EIS در آن گرفته شده (V).
    e_onset : float
        پتانسیل شروع واکنش از CV (V).
    eis_region : str
        ناحیه EIS نسبت به E_onset:
        'pre-onset'، 'onset'، یا 'post-onset'.
    r_ct : float
        مقاومت انتقال بار از EIS (Ω).
    i_forward_peak : float
        جریان پیک رفت از CV (mA).
    consistency_score : float
        امتیاز سازگاری EIS و CV (0 تا 1).
        1.0 = کاملاً سازگار، 0.0 = ناسازگار.
    warnings : list[str]
        هشدارهای سازگاری.
    recommendations : list[str]
        پیشنهادهای بهبود اندازه‌گیری.
    """

    eis_potential: float
    e_onset: float
    eis_region: str
    r_ct: float
    i_forward_peak: float
    consistency_score: float
    warnings: list
    recommendations: list

    def report(self) -> str:
        """گزارش کامل همبستگی."""
        lines = [
            "═" * 60,
            "  🔗 گزارش همبستگی EIS-CV — EISForge",
            "═" * 60,
            "",
            "  📍 موقعیت EIS در CV:",
            f"     پتانسیل EIS  = {self.eis_potential:.4f} V",
            f"     E_onset      = {self.e_onset:.4f} V",
            f"     ناحیه        = {self._region_persian()}",
            "",
            "  ⚡ پارامترهای کلیدی:",
            f"     R_ct (EIS)   = {self.r_ct:.2f} Ω",
            f"     I_f (CV)     = {self.i_forward_peak:.4f} mA",
            f"     امتیاز سازگاری = {self.consistency_score:.0%}",
            "",
        ]

        if self.warnings:
            lines.append("  ⚠️  هشدارها:")
            for w in self.warnings:
                lines.append(f"     • {w}")
            lines.append("")

        if self.recommendations:
            lines.append("  💡 پیشنهادها:")
            for r in self.recommendations:
                lines.append(f"     • {r}")
            lines.append("")

        lines.append("═" * 60)
        return "\n".join(lines)

    def _region_persian(self) -> str:
        mapping = {
            "pre-onset":  "قبل از واکنش (E < E_onset) — بدون اکسیداسیون الکل",
            "onset":      "روی E_onset — شروع واکنش",
            "post-onset": "بعد از E_onset — واکنش فعال",
        }
        return mapping.get(self.eis_region, self.eis_region)


class EISCVCorrelator:
    """
    ارتباط‌دهنده EIS و CV برای AOR.

    Parameters
    ----------
    onset_tolerance : float
        بازه پتانسیل برای تشخیص "روی E_onset" (V، پیش‌فرض: 0.05).
    electrolyte : str
        'acidic' یا 'alkaline' — برای تفسیر صحیح.
    """

    def __init__(
        self,
        onset_tolerance: float = 0.05,
        electrolyte: str = "acidic",
    ) -> None:
        self.onset_tolerance = onset_tolerance
        self.electrolyte = electrolyte

    def correlate(
        self,
        cv_result: CVAnalysisResult,
        eis_fit_result: FitResult,
        eis_potential: float,
    ) -> EISCVCorrelationResult:
        """
        همبستگی کامل EIS و CV.

        Parameters
        ----------
        cv_result : CVAnalysisResult
            نتایج تحلیل CV.
        eis_fit_result : FitResult
            نتایج فیت EIS.
        eis_potential : float
            پتانسیلی که EIS در آن اندازه‌گیری شده (V).

        Returns
        -------
        EISCVCorrelationResult
        """
        warnings = []
        recommendations = []

        # ── تشخیص ناحیه EIS ──────────────────────────────────────────────────
        eis_region = self._classify_region(
            eis_potential, cv_result.e_onset
        )

        # ── استخراج R_ct از EIS fit ───────────────────────────────────────────
        r_ct = self._extract_r_ct(eis_fit_result)

        # ── بررسی سازگاری ─────────────────────────────────────────────────────
        consistency_score, w, r = self._check_consistency(
            eis_region=eis_region,
            r_ct=r_ct,
            cv_result=cv_result,
            eis_potential=eis_potential,
        )
        warnings.extend(w)
        recommendations.extend(r)

        # ── پیشنهادهای اندازه‌گیری ────────────────────────────────────────────
        recommendations.extend(
            self._measurement_recommendations(eis_region, cv_result)
        )

        result = EISCVCorrelationResult(
            eis_potential=eis_potential,
            e_onset=cv_result.e_onset,
            eis_region=eis_region,
            r_ct=r_ct,
            i_forward_peak=cv_result.i_forward_peak,
            consistency_score=consistency_score,
            warnings=warnings,
            recommendations=recommendations,
        )

        logger.info(
            "همبستگی EIS-CV: ناحیه=%s، سازگاری=%.0f%%",
            eis_region, consistency_score * 100,
        )
        return result

    def _classify_region(
        self, eis_potential: float, e_onset: float
    ) -> str:
        """تشخیص ناحیه EIS نسبت به E_onset."""
        delta = eis_potential - e_onset
        if delta < -self.onset_tolerance:
            return "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            return "onset"
        else:
            return "post-onset"

    @staticmethod
    def _extract_r_ct(fit_result: FitResult) -> float:
        """
        استخراج R_ct از نتایج فیت EIS.

        در مدارهای AOR، R_ct معمولاً R1 است
        (دومین مقاومت بعد از R_solution).
        """
        params = fit_result.parameters
        for key in ["R1", "R_1", "R_ct"]:
            if key in params:
                return float(params[key])
        # اگر نام استاندارد نبود، دومین مقاومت را برگردان
        r_values = [v for k, v in params.items() if k.startswith("R")]
        if len(r_values) >= 2:
            return float(sorted(r_values)[1])
        return float("nan")

    def _check_consistency(
        self,
        eis_region: str,
        r_ct: float,
        cv_result: CVAnalysisResult,
        eis_potential: float,
    ) -> tuple[float, list, list]:
        """
        بررسی سازگاری فیزیکی بین EIS و CV.

        قوانین سازگاری:
        1. pre-onset: R_ct باید خیلی بزرگ باشد (واکنشی نیست)
        2. onset: R_ct باید متوسط باشد
        3. post-onset: R_ct باید کوچک باشد
        """
        warnings = []
        recommendations = []
        score = 1.0

        if np.isnan(r_ct):
            warnings.append("R_ct از فیت EIS استخراج نشد.")
            return 0.5, warnings, recommendations

        if eis_region == "pre-onset":
            # انتظار: R_ct بزرگ (> 1000 Ω برای محیط اسیدی)
            if r_ct < 100:
                score -= 0.4
                warnings.append(
                    f"R_ct = {r_ct:.1f} Ω خیلی کوچک است برای E < E_onset. "
                    "ممکن است واکنش دیگری در این پتانسیل رخ دهد."
                )
            else:
                recommendations.append(
                    "✅ R_ct بزرگ در pre-onset — سازگار با غیرفعال بودن کاتالیست."
                )

        elif eis_region == "onset":
            # انتظار: R_ct در حال کاهش
            recommendations.append(
                "💡 EIS روی E_onset — بهترین نقطه برای مطالعه مکانیزم شروع واکنش."
            )

        elif eis_region == "post-onset":
            # انتظار: R_ct کوچک، با I_f رابطه معکوس دارد
            if r_ct > 5000:
                score -= 0.3
                warnings.append(
                    f"R_ct = {r_ct:.1f} Ω خیلی بزرگ است برای E > E_onset. "
                    "بررسی کنید: آیا پتانسیل EIS با CV یکسان بوده؟"
                )

            # بررسی رابطه R_ct با I_f/I_b
            if cv_result.if_ib_ratio < 1.0 and r_ct < 50:
                score -= 0.2
                warnings.append(
                    "I_f/I_b < 1 (مسمومیت CO) ولی R_ct کوچک است — "
                    "ممکن است دو قوس EIS وجود داشته باشد که overlap کرده‌اند."
                )
                recommendations.append(
                    "مدار دو قوسی R0-p(R1,CPE1)-p(R2,CPE2) را امتحان کنید."
                )

        return max(0.0, score), warnings, recommendations

    def _measurement_recommendations(
        self,
        eis_region: str,
        cv_result: CVAnalysisResult,
    ) -> list:
        """پیشنهادهای بهبود اندازه‌گیری."""
        recs = []

        if eis_region == "pre-onset":
            recs.append(
                f"پیشنهاد: EIS را در E_onset ({cv_result.e_onset:.3f} V) "
                "یا بعد از آن تکرار کنید تا R_ct واکنش را ببینید."
            )

        if cv_result.if_ib_ratio < 1.0:
            recs.append(
                "I_f/I_b < 1 نشان مسمومیت CO دارد. "
                "EIS در فرکانس‌های پایین‌تر (تا 1 mHz) بگیرید "
                "تا دینامیک CO_ads کامل‌تر دیده شود."
            )

        if cv_result.e_onset > 0.6 and self.electrolyte == "acidic":
            recs.append(
                "E_onset بالا در محیط اسیدی — "
                "محیط بازی (KOH) را امتحان کنید که معمولاً E_onset کمتری دارد."
            )

        return recs

    def compare_multiple_eis(
        self,
        cv_result: CVAnalysisResult,
        eis_fits: list[tuple[float, FitResult]],
    ) -> list[EISCVCorrelationResult]:
        """
        همبستگی چندین طیف EIS (در پتانسیل‌های مختلف) با یک CV.

        Parameters
        ----------
        cv_result : CVAnalysisResult
        eis_fits : list of (potential, FitResult)

        Returns
        -------
        list[EISCVCorrelationResult]
            نتایج مرتب‌شده بر اساس پتانسیل.
        """
        results = []
        for potential, fit in sorted(eis_fits, key=lambda x: x[0]):
            result = self.correlate(cv_result, fit, potential)
            results.append(result)
            logger.info(
                "E=%.3f V → %s (سازگاری: %.0f%%)",
                potential, result.eis_region,
                result.consistency_score * 100,
            )
        return results
