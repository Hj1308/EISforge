"""
EIS-CV Correlator — ارتباط‌دهنده EIS با CV.
نویسنده: Hoda Jafari | May 2026

از همه انواع کاتالیزور پشتیبانی می‌کند:
    - noble_metal  : Pt, Pd, Au, Rh
    - alloy        : PtRu, PtSn, PdAu, PtCu
    - metal_oxide  : NiO, Co3O4, MnO2
    - metal_free   : B4C, N-doped Carbon, CNT, rGO
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eisforge.analysis.cv_analyzer import (
    CVAnalysisResult,
    CATALYST_NOBLE_METAL,
    CATALYST_ALLOY,
    CATALYST_METAL_OXIDE,
    CATALYST_METAL_FREE,
)
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)


@dataclass
class EISCVCorrelationResult:
    eis_potential     : float
    e_onset           : float
    eis_region        : str
    r_ct              : float
    i_forward_peak    : float
    catalyst_type     : str = CATALYST_NOBLE_METAL
    consistency_score : float = 1.0
    warnings          : list = field(default_factory=list)
    recommendations   : list = field(default_factory=list)

    def report(self) -> str:
        region_map = {
            "pre-onset" : "قبل از E_onset",
            "onset"     : "روی E_onset",
            "post-onset": "بعد از E_onset",
        }
        catalyst_map = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt/Pd/Au)",
            CATALYST_ALLOY       : "Alloy (PtRu/PtSn/PdAu)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO/Co3O4)",
            CATALYST_METAL_FREE  : "Metal-Free (B4C/N-C/CNT)",
        }
        lines = [
            "═" * 64,
            "  🔗 همبستگی EIS-CV — EISForge",
            "═" * 64,
            f"  نوع کاتالیزور  = {catalyst_map.get(self.catalyst_type, self.catalyst_type)}",
            f"  پتانسیل EIS  = {self.eis_potential:.4f} V",
            f"  E_onset      = {self.e_onset:.4f} V",
            f"  ناحیه        = {region_map.get(self.eis_region, self.eis_region)}",
            f"  R_ct (EIS)   = {self.r_ct:.2f} Ω",
            f"  I_f (CV)     = {self.i_forward_peak:.4f} mA",
            f"  سازگاری     = {self.consistency_score:.0%}",
        ]
        if self.warnings:
            lines.append("  ⚠️ هشدارها:")
            for w in self.warnings:
                lines.append(f"     • {w}")
        if self.recommendations:
            lines.append("  💡 پیشنهادها:")
            for r in self.recommendations:
                lines.append(f"     • {r}")
        lines.append("═" * 64)
        return "\n".join(lines)


class EISCVCorrelator:
    """
    ارتباط‌دهنده EIS با CV — برای همه انواع کاتالیزور.

    Parameters
    ----------
    onset_tolerance : float
        بازه تحمل برای تشخیص ناحیه EIS نسبت به E_onset (V).
    electrolyte : str
        'acidic' یا 'alkaline'.
    """

    def __init__(
        self,
        onset_tolerance: float = 0.05,
        electrolyte    : str   = "acidic",
    ):
        self.onset_tolerance = onset_tolerance
        self.electrolyte     = electrolyte

    def correlate(
        self,
        cv_result     : CVAnalysisResult,
        eis_fit_result: FitResult,
        eis_potential : float,
    ) -> EISCVCorrelationResult:
        """
        همبستگی نتایج EIS و CV را بررسی می‌کند.

        Parameters
        ----------
        cv_result      : CVAnalysisResult   نتیجه آنالیز CV
        eis_fit_result : FitResult          نتیجه فیت EIS
        eis_potential  : float              پتانسیلی که EIS در آن گرفته شده (V)
        """
        warnings        = []
        recommendations = []

        ctype = getattr(cv_result, "catalyst_type", CATALYST_NOBLE_METAL)

        # ── ناحیه EIS ──────────────────────────────────────────────────────
        delta = eis_potential - cv_result.e_onset
        if delta < -self.onset_tolerance:
            region = "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            region = "onset"
        else:
            region = "post-onset"

        # ── R_ct ───────────────────────────────────────────────────────────
        r_ct = self._extract_r_ct(eis_fit_result)

        # ── بررسی سازگاری (وابسته به نوع کاتالیزور) ──────────────────────
        score = 1.0

        if not np.isnan(r_ct):
            if ctype == CATALYST_METAL_FREE:
                # برای B4C و مواد غیرفلزی، R_ct بالاتر طبیعی است
                if region == "pre-onset" and r_ct < 500:
                    score -= 0.3
                    warnings.append(
                        f"R_ct={r_ct:.1f}Ω نسبتاً کوچک برای کاتالیزور غیرفلزی در E < E_onset — "
                        f"بررسی کنید که پیک مشاهده‌شده فارادیک است نه خازنی"
                    )
                elif region == "post-onset" and r_ct > 50000:
                    score -= 0.2
                    warnings.append(
                        f"R_ct={r_ct:.1f}Ω خیلی بزرگ — ممکن است پسیواسیون سطحی رخ داده باشد"
                    )

            elif ctype == CATALYST_METAL_OXIDE:
                # اکسید فلزات: R_ct در ناحیه فعال کاهش می‌یابد
                if region == "pre-onset" and r_ct < 50:
                    score -= 0.3
                    warnings.append(
                        f"R_ct={r_ct:.1f}Ω کوچک برای اکسید فلز در E < E_onset — "
                        f"تأیید کنید که EIS در پتانسیل درست گرفته شده"
                    )

            else:
                # noble_metal و alloy — آستانه‌های کلاسیک
                if region == "pre-onset" and r_ct < 100:
                    score -= 0.4
                    warnings.append(f"R_ct={r_ct:.1f}Ω کوچک برای E < E_onset")
                elif region == "post-onset" and r_ct > 5000:
                    score -= 0.3
                    warnings.append(f"R_ct={r_ct:.1f}Ω بزرگ برای E > E_onset")

        # ── I_f/I_b — فقط برای فلزات (برای metal_free مورد کاربرد ندارد) ──
        if_ib = cv_result.if_ib_ratio
        if ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
            if not np.isnan(if_ib):
                if if_ib < 1.0:
                    recommendations.append(
                        "I_f/I_b < 1 — مدار دو قوسی R0-p(R1,CPE1)-p(R2,CPE2) را امتحان کنید"
                    )
                elif if_ib > 3.0 and not np.isnan(r_ct) and r_ct > 1000:
                    warnings.append(
                        f"I_f/I_b={if_ib:.2f} خوب اما R_ct={r_ct:.1f}Ω بزرگ است — "
                        f"بررسی کنید که EIS در پتانسیل صحیح گرفته شده"
                    )
        elif ctype == CATALYST_METAL_FREE:
            recommendations.append(
                "کاتالیزور غیرفلزی: I_f/I_b کاربرد ندارد. "
                "ECSA را از طریق C_dl تخمین بزنید. "
                "مدار R0-p(R1,CPE1)-p(R2,CPE2) برای مقاومت بین‌ذره‌ای پیشنهاد می‌شود."
            )

        # ── پیشنهاد تکرار EIS ─────────────────────────────────────────────
        if region == "pre-onset":
            recommendations.append(
                f"EIS را در E_onset ({cv_result.e_onset:.3f} V) تکرار کنید "
                f"تا R_ct در شرایط واکنش مستقیماً اندازه‌گیری شود"
            )

        # ── پیشنهاد خاص برای غیرفلزی ──────────────────────────────────────
        if ctype == CATALYST_METAL_FREE and region == "onset":
            recommendations.append(
                "برای B4C/کربن: مقاومت تماسی بین‌ذره‌ای (R_inter) و "
                "نفوذ یون در حفرات را در مدار EIS لحاظ کنید"
            )

        return EISCVCorrelationResult(
            eis_potential     = eis_potential,
            e_onset           = cv_result.e_onset,
            eis_region        = region,
            r_ct              = r_ct if not np.isnan(r_ct) else 0.0,
            i_forward_peak    = cv_result.i_forward_peak,
            catalyst_type     = ctype,
            consistency_score = max(0.0, score),
            warnings          = warnings,
            recommendations   = recommendations,
        )

    @staticmethod
    def _extract_r_ct(fit: FitResult) -> float:
        """R_ct را از پارامترهای فیت EIS استخراج می‌کند."""
        for key in ["R1", "R_ct", "R_1"]:
            if key in fit.parameters:
                return float(fit.parameters[key])
        r_vals = [v for k, v in fit.parameters.items() if k.startswith("R")]
        return float(sorted(r_vals)[1]) if len(r_vals) >= 2 else float("nan")
