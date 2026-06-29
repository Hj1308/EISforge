"""
Multi-Technique Analyzer — Correlates CV, EIS, and Koutecky-Levich results.
Author: Hoda Jafari | May 2026

Provides a unified view of catalyst performance by cross-validating:
    - E_onset (CV) vs. EIS measurement potential
    - R_ct (EIS) vs. j_kinetic (KL) → intrinsic activity check
    - n_electrons (KL) vs. expected mechanism for the catalyst type
    - I_f/I_b (CV) vs. n (KL) → poisoning / incomplete oxidation flag

Design principle
----------------
This class does **not** run the individual analyses; it receives their
already-computed results and performs cross-correlations only.

Usage
-----
    from eisforge.analysis.multi_technique import ComprehensiveAnalyzer

    report = ComprehensiveAnalyzer().analyze(
        cv_result=cv_res,
        eis_fit_result=eis_res,
        kl_full_result=kl_res,
        eis_potential=0.5,
    )
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from eisforge.analysis.cv_analyzer import CVAnalysisResult
from eisforge.analysis.eis_cv_correlator import EISCVCorrelator
from eisforge.analysis.koutecky_levich import KLFullResult
from eisforge.catalogs.circuit_models import CircuitModel, get_suggested_circuit
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComprehensiveReport:
    """
    Unified report combining all three electrochemical techniques.

    Attributes are grouped by technique (CV / EIS / KL) plus cross-validation
    metadata (warnings, recommendations, consistency_score).
    """

    # ---- Identity
    catalyst_type: str
    alcohol: str
    electrolyte: str

    # ---- CV highlights
    e_onset: float
    if_ib_ratio: float
    ecsa_cm2: Optional[float] = None

    # ---- EIS highlights
    eis_potential: float = float("nan")
    r_ct: float = float("nan")
    eis_region: str = ""
    suggested_circuit: str = ""

    # ---- KL highlights
    kl_mean_n: float = float("nan")
    kl_best_jk: float = float("nan")
    kl_best_potential: float = float("nan")
    kl_r2_best: float = float("nan")

    # ---- Cross-validation output
    warnings: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    consistency_score: float = 1.0

    # ---- Full result objects for downstream inspection
    cv_result: Optional[CVAnalysisResult] = None
    eis_fit_result: Optional[FitResult] = None
    kl_full_result: Optional[KLFullResult] = None

    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted, human-readable report string."""
        lines = [
            "=" * 72,
            "  \ud83d\udcca COMPREHENSIVE ELECTROCHEMICAL REPORT",
            "=" * 72,
            f"  Catalyst       : {self.catalyst_type}",
            f"  Alcohol        : {self.alcohol}",
            f"  Electrolyte    : {self.electrolyte}",
            "-" * 72,
            f"  [CV]  E_onset       = {self.e_onset:.4f} V vs RHE",
            f"  [CV]  I_f / I_b     = {self.if_ib_ratio:.2f}",
            f"  [CV]  ECSA          = {self.ecsa_cm2 if self.ecsa_cm2 is not None else 'N/A'} cm\u00b2",
            "-" * 72,
            f"  [EIS] Potential     = {self.eis_potential:.4f} V vs RHE",
            f"  [EIS] Region        = {self.eis_region}",
            f"  [EIS] R_ct          = {self.r_ct:.2f} \u03a9",
            f"  [EIS] Suggested     = {self.suggested_circuit}",
            "-" * 72,
            f"  [KL]  Mean n        = {self.kl_mean_n:.2f}",
            f"  [KL]  Best j_k      = {self.kl_best_jk:.4f} mA/cm\u00b2  (E = {self.kl_best_potential:.3f} V)",
            f"  [KL]  Best R\u00b2       = {self.kl_r2_best:.4f}",
            "-" * 72,
            f"  \u2705 Consistency Score : {self.consistency_score:.0%}",
        ]

        if self.warnings:
            lines.append("  \u26a0\ufe0f  Warnings:")
            for w in self.warnings:
                lines.append(f"      \u2022 {w}")

        if self.recommendations:
            lines.append("  \ud83d\udca1 Recommendations:")
            for r in self.recommendations:
                lines.append(f"      \u2022 {r}")

        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class ComprehensiveAnalyzer:
    """
    Orchestrates cross-validation of CV, EIS, and KL results.

    Parameters
    ----------
    verbose : bool
        If ``True`` (default), log debug messages during analysis.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ------------------------------------------------------------------

    def analyze(
        self,
        cv_result: CVAnalysisResult,
        eis_fit_result: FitResult,
        kl_full_result: KLFullResult,
        eis_potential: float,
    ) -> ComprehensiveReport:
        """
        Cross-validate CV, EIS, and KL results and return a unified report.

        Parameters
        ----------
        cv_result : CVAnalysisResult
            Output from :class:`~eisforge.analysis.cv_analyzer.CVAnalyzer`.
        eis_fit_result : FitResult
            Output from the CNLS fitter.
        kl_full_result : KLFullResult
            Output from :class:`~eisforge.analysis.koutecky_levich.KLAnalyzer`.
        eis_potential : float
            Potential (V vs RHE) at which EIS was recorded.

        Returns
        -------
        ComprehensiveReport
        """
        # ---- Input validation ---------------------------------------------
        if not isinstance(cv_result, CVAnalysisResult):
            raise TypeError(f"cv_result must be CVAnalysisResult, got {type(cv_result).__name__}")
        if not isinstance(eis_fit_result, FitResult):
            raise TypeError(f"eis_fit_result must be FitResult, got {type(eis_fit_result).__name__}")
        if not isinstance(kl_full_result, KLFullResult):
            raise TypeError(f"kl_full_result must be KLFullResult, got {type(kl_full_result).__name__}")
        if not np.isfinite(eis_potential):
            raise ValueError(f"eis_potential must be finite, got {eis_potential}")
        if not np.isfinite(cv_result.e_onset):
            raise ValueError("cv_result.e_onset is not finite.")

        warnings: list[str] = []
        recommendations: list[str] = []
        penalty = 0.0

        _PENALTY = {"low": 0.10, "medium": 0.20, "high": 0.35}

        def warn(msg: str, severity: str = "medium") -> None:
            warnings.append(msg)
            nonlocal penalty
            penalty += _PENALTY.get(severity, 0.20)

        # ---- 1. Basic info ------------------------------------------------
        ctype       = cv_result.catalyst_type
        alcohol     = kl_full_result.alcohol
        electrolyte = kl_full_result.electrolyte

        # ---- 2. EIS region relative to CV onset ---------------------------
        delta      = eis_potential - cv_result.e_onset
        onset_tol  = 0.05  # V
        if delta < -onset_tol:
            region = "pre-onset"
        elif abs(delta) <= onset_tol:
            region = "onset"
        else:
            region = "post-onset"

        # ---- 3. Extract R_ct — delegate to the hardened correlator method --
        r_ct = EISCVCorrelator._extract_r_ct(eis_fit_result)

        # ---- 4. Suggested circuit -----------------------------------------
        suggested_circuit = get_suggested_circuit(ctype, region, electrolyte)

        # ---- 5. KL highlights ---------------------------------------------
        kl_results = kl_full_result.results_per_potential
        if kl_results:
            best         = max(kl_results, key=lambda r: r.r_squared)
            kl_best_jk   = best.j_kinetic
            kl_best_pot  = best.potential_V
            kl_r2_best   = best.r_squared
            kl_mean_n    = kl_full_result.mean_n_electrons
        else:
            kl_best_jk  = float("nan")
            kl_best_pot = float("nan")
            kl_r2_best  = float("nan")
            kl_mean_n   = float("nan")

        # ==================================================================
        # CROSS-VALIDATION CHECKS
        # ==================================================================

        # Check 1: Potential alignment between EIS and KL best fit
        if np.isfinite(kl_best_pot) and kl_best_pot > 0:
            if abs(eis_potential - kl_best_pot) > 0.1:
                warn(
                    f"EIS measured at {eis_potential:.3f} V but KL best fit is at "
                    f"{kl_best_pot:.3f} V (\u0394E = {abs(eis_potential - kl_best_pot):.3f} V). "
                    f"Consider measuring both techniques at the same potential for "
                    f"a direct R_ct vs j_kinetic comparison.",
                    severity="medium",
                )

        # Check 2: Activity consistency (high j_k should correlate with low R_ct)
        if np.isfinite(kl_best_jk) and np.isfinite(r_ct) and r_ct > 0:
            if kl_best_jk > 10.0 and r_ct > 100:
                warn(
                    f"High j_kinetic ({kl_best_jk:.2f} mA/cm\u00b2) combined with high "
                    f"R_ct ({r_ct:.1f} \u03a9). This is physically inconsistent. Verify "
                    f"that EIS was measured at the same (or corrected) potential "
                    f"as the KL analysis, and that the correct circuit element "
                    f"is being identified as R_ct.",
                    severity="high",
                )

        # Check 3: n_electrons vs expected mechanism
        if np.isfinite(kl_mean_n):
            if ctype in ("noble_metal", "alloy"):
                if kl_mean_n < 1.5:
                    warn(
                        f"Mean n = {kl_mean_n:.2f} is very low for a metal/alloy "
                        f"catalyst (expected 2–6). Check alcohol concentration and "
                        f"diffusion coefficient used in the KL calculation.",
                        severity="high",
                    )
                elif kl_mean_n > 5.0:
                    recommendations.append(
                        f"n = {kl_mean_n:.2f} suggests near-complete oxidation to CO\u2082. "
                        f"Verify with product analysis (HPLC, GC, DEMS)."
                    )
            elif ctype == "metal_oxide":
                if kl_mean_n < 1.0:
                    warn(
                        f"n = {kl_mean_n:.2f} for metal oxide is below 1. "
                        f"Confirm the process is faradaic and not purely capacitive "
                        f"(check double-layer subtraction).",
                        severity="high",
                    )
            elif ctype == "carbon_material":
                if kl_mean_n > 2.5:
                    recommendations.append(
                        f"n = {kl_mean_n:.2f} is high for a metal-free carbon catalyst. "
                        f"Ensure no metallic contamination is contributing to current."
                    )

        # Check 4: I_f/I_b (CV) vs n (KL) — CO / aldehyde poisoning flag
        if np.isfinite(cv_result.if_ib_ratio) and np.isfinite(kl_mean_n):
            if ctype in ("noble_metal", "alloy"):
                if cv_result.if_ib_ratio < 0.8 and kl_mean_n < 2.0:
                    warn(
                        f"Low I_f/I_b ({cv_result.if_ib_ratio:.2f}) and low n "
                        f"({kl_mean_n:.2f}) together indicate likely CO or aldehyde "
                        f"poisoning of active sites.",
                        severity="high",
                    )
                    recommendations.append(
                        "Perform CO-stripping CV to quantify poisoning. "
                        "Consider adding a second metal (e.g., Ru, Sn) or "
                        "increasing operating temperature."
                    )

        # ---- Measurement suggestions based on region ----------------------
        if region == "pre-onset":
            recommendations.append(
                f"EIS was measured at {eis_potential:.3f} V, which is below "
                f"E_onset ({cv_result.e_onset:.3f} V). Consider repeating EIS "
                f"at or just above E_onset to capture intrinsic charge-transfer "
                f"resistance without surface passivation contributions."
            )

        if ctype == "carbon_material" and region in ("onset", "post-onset"):
            recommendations.append(
                f"For porous carbon catalysts, include an inter-particle contact "
                f"resistance in the EIS model: {CircuitModel.POROUS_CARBON_CONTACT.value}. "
                f"This separates bulk electronic resistance from faradaic R_ct."
            )

        # ---- Consistency score --------------------------------------------
        score = max(0.0, min(1.0, 1.0 - penalty))

        # ---- ECSA (direct attribute access — no getattr fallback) ----------
        ecsa = cv_result.ecsa_cm2 if hasattr(cv_result, "ecsa_cm2") else None

        return ComprehensiveReport(
            catalyst_type     = ctype,
            alcohol           = alcohol,
            electrolyte       = electrolyte,
            e_onset           = cv_result.e_onset,
            if_ib_ratio       = cv_result.if_ib_ratio,
            ecsa_cm2          = ecsa,
            eis_potential     = eis_potential,
            r_ct              = r_ct if np.isfinite(r_ct) else float("nan"),
            eis_region        = region,
            suggested_circuit = str(suggested_circuit.value),
            kl_mean_n         = kl_mean_n,
            kl_best_jk        = kl_best_jk,
            kl_best_potential = kl_best_pot,
            kl_r2_best        = kl_r2_best,
            warnings          = warnings,
            recommendations   = recommendations,
            consistency_score = score,
            cv_result         = cv_result,
            eis_fit_result    = eis_fit_result,
            kl_full_result    = kl_full_result,
        )
