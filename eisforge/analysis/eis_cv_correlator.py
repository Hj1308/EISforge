"""
EIS-CV Correlator — Cross-technique correlation between EIS and CV results.
Author: Hoda Jafari | May 2026

Supports all catalyst families:
    - noble_metal  : Pt, Pd, Au, Rh
    - alloy        : PtRu, PtSn, PdAu, PtCu
    - metal_oxide  : NiO, Co3O4, MnO2, Co2NiO4
    - metal_free   : B4C, N-doped Carbon, CNT, rGO
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    catalyst_type     : str   = CATALYST_NOBLE_METAL
    consistency_score : float = 1.0
    warnings          : list  = field(default_factory=list)
    recommendations   : list  = field(default_factory=list)

    def report(self) -> str:
        catalyst_map = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt / Pd / Au / Rh)",
            CATALYST_ALLOY       : "Alloy (PtRu / PtSn / PdAu / PtCu)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO / Co3O4 / MnO2)",
            CATALYST_METAL_FREE  : "Metal-Free (B4C / N-doped C / CNT)",
        }
        lines = [
            "=" * 64,
            "  EIS-CV Correlation Report — EISForge",
            "=" * 64,
            f"  Catalyst type    : {catalyst_map.get(self.catalyst_type, self.catalyst_type)}",
            f"  EIS potential    : {self.eis_potential:.4f} V",
            f"  E_onset (CV)     : {self.e_onset:.4f} V",
            f"  EIS region       : {self.eis_region}",
            f"  R_ct (EIS)       : {self.r_ct:.2f} Ohm",
            f"  I_forward (CV)   : {self.i_forward_peak:.4f} mA",
            f"  Consistency      : {self.consistency_score:.0%}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    - {w}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    - {r}")
        lines.append("=" * 64)
        return "\n".join(lines)


class EISCVCorrelator:
    """
    Cross-technique correlator between EIS fit results and CV analysis.

    Checks whether the EIS measurement potential is consistent with the
    CV-derived E_onset, and whether R_ct values make physical sense
    for the given catalyst type.

    Parameters
    ----------
    onset_tolerance : float
        Potential window around E_onset to define the 'onset' region (V).
    electrolyte : str
        'acidic' or 'alkaline'.
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
        Correlate EIS and CV results.

        Parameters
        ----------
        cv_result      : CVAnalysisResult   Result from CVAnalyzer.analyze()
        eis_fit_result : FitResult          Result from CNLSFitter.fit()
        eis_potential  : float              Potential at which EIS was measured (V)

        Returns
        -------
        EISCVCorrelationResult
        """
        warnings        = []
        recommendations = []

        ctype = getattr(cv_result, "catalyst_type", CATALYST_NOBLE_METAL)

        # ── Determine EIS region relative to E_onset ──────────────────────
        delta = eis_potential - cv_result.e_onset
        if delta < -self.onset_tolerance:
            region = "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            region = "onset"
        else:
            region = "post-onset"

        # ── Extract R_ct from EIS fit ──────────────────────────────────────
        r_ct = self._extract_r_ct(eis_fit_result)

        # ── Consistency check — catalyst-type specific ─────────────────────
        score = 1.0

        if not np.isnan(r_ct):

            if ctype == CATALYST_METAL_FREE:
                # B4C, N-doped C, CNT: higher R_ct is normal
                if region == "pre-onset" and r_ct < 500:
                    score -= 0.3
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unusually low for a metal-free "
                        f"catalyst at E < E_onset. Verify the observed peak is "
                        f"faradaic and not capacitive background."
                    )
                elif region == "post-onset" and r_ct > 50000:
                    score -= 0.2
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is very large — possible surface "
                        f"passivation or pore blocking."
                    )

            elif ctype == CATALYST_METAL_OXIDE:
                if region == "pre-onset" and r_ct < 50:
                    score -= 0.3
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unexpectedly small for a metal "
                        f"oxide at E < E_onset. Confirm EIS was measured at the "
                        f"correct potential."
                    )

            else:
                # noble_metal and alloy — classical thresholds
                if region == "pre-onset" and r_ct < 100:
                    score -= 0.4
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is small for E < E_onset "
                        f"({cv_result.e_onset:.3f} V). Expected R_ct >> 100 Ohm "
                        f"in the pre-onset region."
                    )
                elif region == "post-onset" and r_ct > 5000:
                    score -= 0.3
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is large for E > E_onset. "
                        f"Possible CO poisoning or mass-transport limitation."
                    )

        # ── I_f/I_b check — only for metal and alloy catalysts ────────────
        if_ib = cv_result.if_ib_ratio

        if ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
            if not np.isnan(if_ib):
                if if_ib < 1.0:
                    recommendations.append(
                        f"I_f/I_b = {if_ib:.2f} < 1.0 — CO poisoning suspected. "
                        f"Try two-RC circuit R0-p(R1,CPE1)-p(R2,CPE2) for EIS fit."
                    )
                elif if_ib > 3.0 and not np.isnan(r_ct) and r_ct > 1000:
                    warnings.append(
                        f"I_f/I_b = {if_ib:.2f} is excellent but R_ct = {r_ct:.1f} Ohm "
                        f"is large — confirm EIS was measured at the active potential."
                    )

        elif ctype == CATALYST_METAL_FREE:
            recommendations.append(
                "Metal-free catalyst: I_f/I_b ratio is not applicable. "
                "Estimate ECSA via C_dl method from scan-rate dependence. "
                "For EIS, consider R0-p(R1,CPE1)-p(R2,CPE2) to capture "
                "inter-particle contact resistance and pore ion diffusion."
            )

        elif ctype == CATALYST_METAL_OXIDE:
            recommendations.append(
                "Metal oxide catalyst: verify that two-RC circuit captures "
                "both the M(OH)x/MOOx redox process and direct surface oxidation."
            )

        # ── Recommend repeating EIS at E_onset ────────────────────────────
        if region == "pre-onset":
            recommendations.append(
                f"Repeat EIS at E_onset = {cv_result.e_onset:.3f} V to measure "
                f"R_ct directly under active reaction conditions."
            )

        # ── Metal-free specific EIS advice ─────────────────────────────────
        if ctype == CATALYST_METAL_FREE and region == "onset":
            recommendations.append(
                "For B4C / carbon catalysts at E_onset: include an inter-particle "
                "resistance element in the EIS circuit. Ion diffusion in pores "
                "may appear as a low-frequency Warburg tail."
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
        """Extract R_ct from EIS fit parameters."""
        for key in ["R1", "R_ct", "R_1"]:
            if key in fit.parameters:
                return float(fit.parameters[key])
        r_vals = [v for k, v in fit.parameters.items() if k.startswith("R")]
        return float(sorted(r_vals)[1]) if len(r_vals) >= 2 else float("nan")
