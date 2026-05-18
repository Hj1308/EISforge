"""
EIS-CV Correlator — ارتباط‌دهنده EIS با CV.
نویسنده: Hoda Jafari | May 2026
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from eisforge.analysis.cv_analyzer import CVAnalysisResult
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)


@dataclass
class EISCVCorrelationResult:
    eis_potential: float
    e_onset: float
    eis_region: str
    r_ct: float
    i_forward_peak: float
    consistency_score: float
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)

    def report(self) -> str:
        region_map = {
            "pre-onset":  "قبل از E_onset",
            "onset":      "روی E_onset",
            "post-onset": "بعد از E_onset",
        }
        lines = [
            "═" * 60,
            "  🔗 همبستگی EIS-CV — EISForge",
            "═" * 60,
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
        lines.append("═" * 60)
        return "\n".join(lines)


class EISCVCorrelator:
    def __init__(self, onset_tolerance: float = 0.05, electrolyte: str = "acidic"):
        self.onset_tolerance = onset_tolerance
        self.electrolyte = electrolyte

    def correlate(
        self,
        cv_result: CVAnalysisResult,
        eis_fit_result: FitResult,
        eis_potential: float,
    ) -> EISCVCorrelationResult:
        warnings, recommendations = [], []

        # ناحیه EIS
        delta = eis_potential - cv_result.e_onset
        if delta < -self.onset_tolerance:
            region = "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            region = "onset"
        else:
            region = "post-onset"

        # R_ct
        r_ct = self._extract_r_ct(eis_fit_result)

        # بررسی سازگاری
        score = 1.0
        if not np.isnan(r_ct):
            if region == "pre-onset" and r_ct < 100:
                score -= 0.4
                warnings.append(f"R_ct={r_ct:.1f}Ω کوچک برای E < E_onset")
            elif region == "post-onset" and r_ct > 5000:
                score -= 0.3
                warnings.append(f"R_ct={r_ct:.1f}Ω بزرگ برای E > E_onset")

        if cv_result.if_ib_ratio < 1.0:
            recommendations.append(
                "I_f/I_b < 1 — مدار دو قوسی R0-p(R1,CPE1)-p(R2,CPE2) را امتحان کنید"
            )

        if region == "pre-onset":
            recommendations.append(
                f"EIS را در E_onset ({cv_result.e_onset:.3f}V) تکرار کنید"
            )

        return EISCVCorrelationResult(
            eis_potential=eis_potential,
            e_onset=cv_result.e_onset,
            eis_region=region,
            r_ct=r_ct if not np.isnan(r_ct) else 0.0,
            i_forward_peak=cv_result.i_forward_peak,
            consistency_score=max(0.0, score),
            warnings=warnings,
            recommendations=recommendations,
        )

    @staticmethod
    def _extract_r_ct(fit: FitResult) -> float:
        for key in ["R1", "R_ct", "R_1"]:
            if key in fit.parameters:
                return float(fit.parameters[key])
        r_vals = [v for k, v in fit.parameters.items() if k.startswith("R")]
        return float(sorted(r_vals)[1]) if len(r_vals) >= 2 else float("nan")
