"""
eisforge.analysis.multi_technique
==================================
Cross-technique correlator: CV + EIS + Koutecky-Levich.
Author: Hoda Jafari | June 2026

Orchestrates independent analyses and performs four scientific
cross-checks that cannot be done within a single technique:

    1. EIS potential vs. KL best-fit potential alignment
    2. Activity anticorrelation  j_kinetic (KL) ↔ R_ct (EIS)
    3. n_electrons (KL) vs. expected mechanism for catalyst family
    4. I_f/I_b (CV) combined with n (KL) → CO/aldehyde poisoning flag

Design principles
-----------------
* **No duplicate logic**: circuit lookup delegates to
  ``eisforge.catalogs.circuit_models.lookup_circuit()``,
  R_ct extraction delegates to ``EISCVCorrelator._extract_r_ct()``.
* **No hard-coded strings**: all circuit recommendations come from the
  ``CIRCUIT_MAP`` registry.
* **Structured output**: ``ComprehensiveReport`` carries full result
  objects so the caller can do further processing without re-running
  the individual analyses.

Typical usage
-------------
    from eisforge.analysis.multi_technique import ComprehensiveAnalyzer

    report = ComprehensiveAnalyzer().analyze(
        cv_result      = cv_res,
        eis_fit_result = eis_res,
        kl_full_result = kl_res,
        eis_potential  = 0.50,
    )
    print(report.summary())
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
from eisforge.analysis.eis_cv_correlator import EISCVCorrelator
from eisforge.analysis.koutecky_levich import KLFullResult
from eisforge.catalogs.circuit_models import CircuitModel, lookup_circuit
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Penalty weights (shared convention with eis_cv_correlator)
# ---------------------------------------------------------------------------
_SEVERITY: dict[str, float] = {"low": 0.10, "medium": 0.20, "high": 0.35}


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class ComprehensiveReport:
    """
    Unified cross-technique report.

    Scalar highlights from each technique are stored as flat fields for
    quick inspection.  Full result objects are preserved under
    ``cv_result``, ``eis_fit_result``, and ``kl_full_result`` for
    downstream processing.
    """
    # ── Metadata ──────────────────────────────────────────────────────
    catalyst_type : str
    alcohol       : str
    electrolyte   : str

    # ── CV highlights ─────────────────────────────────────────────────
    e_onset       : float
    if_ib_ratio   : float
    ecsa_cm2      : Optional[float] = None

    # ── EIS highlights ────────────────────────────────────────────────
    eis_potential     : float = float("nan")
    r_ct              : float = float("nan")
    eis_region        : str   = ""
    suggested_circuit : Optional[CircuitModel] = None

    # ── KL highlights ─────────────────────────────────────────────────
    kl_mean_n          : float = float("nan")
    kl_best_jk         : float = float("nan")
    kl_best_potential  : float = float("nan")
    kl_r2_best         : float = float("nan")

    # ── Cross-validation output ───────────────────────────────────────
    warnings          : list[str] = field(default_factory=list)
    recommendations   : list[str] = field(default_factory=list)
    consistency_score : float     = 1.0

    # ── Full result objects ───────────────────────────────────────────
    cv_result      : Optional[CVAnalysisResult] = field(default=None, repr=False)
    eis_fit_result : Optional[FitResult]        = field(default=None, repr=False)
    kl_full_result : Optional[KLFullResult]     = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        cmap = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt / Pd / Au / Rh)",
            CATALYST_ALLOY       : "Alloy (PtRu / PtSn / PdAu / PtCu)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO / Co3O4 / MnO2)",
            CATALYST_METAL_FREE  : "Metal-Free (N-doped C / CNT / rGO)",
        }
        circ_str = (
            f"{self.suggested_circuit.notation}  [{self.suggested_circuit.name}]"
            if self.suggested_circuit else "N/A"
        )
        ecsa_str = f"{self.ecsa_cm2:.4f} cm²" if self.ecsa_cm2 is not None else "N/A"
        lines = [
            "=" * 72,
            "  COMPREHENSIVE ELECTROCHEMICAL REPORT  —  EISForge",
            "=" * 72,
            f"  Catalyst     : {cmap.get(self.catalyst_type, self.catalyst_type)}",
            f"  Alcohol      : {self.alcohol}",
            f"  Electrolyte  : {self.electrolyte}",
            "-" * 72,
            f"  [CV]  E_onset          = {self.e_onset:.4f} V",
            f"  [CV]  I_f/I_b          = {self.if_ib_ratio:.2f}",
            f"  [CV]  ECSA             = {ecsa_str}",
            "-" * 72,
            f"  [EIS] Potential        = {self.eis_potential:.4f} V",
            f"  [EIS] Region           = {self.eis_region}",
            f"  [EIS] R_ct             = {self.r_ct:.2f} Ω",
            f"  [EIS] Suggested circuit: {circ_str}",
            "-" * 72,
            f"  [KL]  Mean n           = {self.kl_mean_n:.2f}",
            f"  [KL]  Best j_kinetic   = {self.kl_best_jk:.4f} mA/cm²"
            f"  (E = {self.kl_best_potential:.3f} V)",
            f"  [KL]  Best R²          = {self.kl_r2_best:.4f}",
            "-" * 72,
            f"  Consistency score : {self.consistency_score:.0%}",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        if self.recommendations:
            lines.append("  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    ✦  {r}")
        lines.append("=" * 72)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------

class ComprehensiveAnalyzer:
    """
    Cross-technique correlator for CV, EIS, and Koutecky-Levich results.

    This class does **not** run the individual analyses; each must be
    completed beforehand.  It takes their result objects and performs
    four cross-checks that require all three techniques simultaneously.

    Parameters
    ----------
    onset_tolerance : float
        ±window (V) around E_onset that defines the ``'onset'`` EIS region.
        Default 0.05 V (consistent with ``EISCVCorrelator``).
    """

    def __init__(self, onset_tolerance: float = 0.05) -> None:
        self.onset_tolerance = onset_tolerance
        # Reuse EISCVCorrelator's validated _extract_r_ct static method
        self._extract_r_ct = EISCVCorrelator._extract_r_ct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        cv_result      : CVAnalysisResult,
        eis_fit_result : FitResult,
        kl_full_result : KLFullResult,
        eis_potential  : float,
    ) -> ComprehensiveReport:
        """
        Correlate CV, EIS, and KL results into a ``ComprehensiveReport``.

        Parameters
        ----------
        cv_result : CVAnalysisResult
            Output of ``CVAnalyzer.analyze()``.
        eis_fit_result : FitResult
            Output of ``CNLSFitter.fit()``.
        kl_full_result : KLFullResult
            Output of ``KLAnalyzer.analyze()``.
        eis_potential : float
            Potential (V vs. RHE) at which EIS was measured.

        Returns
        -------
        ComprehensiveReport

        Raises
        ------
        TypeError
            If any result argument is of the wrong type.
        ValueError
            If required attributes are missing or non-finite.
        """
        # ── Input validation ───────────────────────────────────────────
        if not isinstance(cv_result, CVAnalysisResult):
            raise TypeError(f"cv_result must be CVAnalysisResult, got {type(cv_result).__name__}")
        if not isinstance(eis_fit_result, FitResult):
            raise TypeError(f"eis_fit_result must be FitResult, got {type(eis_fit_result).__name__}")
        if not isinstance(kl_full_result, KLFullResult):
            raise TypeError(f"kl_full_result must be KLFullResult, got {type(kl_full_result).__name__}")
        if not np.isfinite(cv_result.e_onset):
            raise ValueError("cv_result.e_onset is not finite.")

        warnings: list[str]        = []
        recommendations: list[str] = []
        penalty: float             = 0.0

        ctype       = cv_result.catalyst_type
        electrolyte = kl_full_result.electrolyte
        alcohol     = kl_full_result.alcohol

        # ── EIS region ────────────────────────────────────────────────
        delta = eis_potential - cv_result.e_onset
        if delta < -self.onset_tolerance:
            region = "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            region = "onset"
        else:
            region = "post-onset"

        # ── R_ct ──────────────────────────────────────────────────────
        r_ct = self._extract_r_ct(eis_fit_result)

        # ── Suggested circuit (from registry — no hard-coded strings) ─
        suggested_circuit = lookup_circuit(ctype, region, electrolyte)

        # ── KL highlights ─────────────────────────────────────────────
        best = kl_full_result.best_result
        if best is not None:
            kl_best_jk  = best.j_kinetic
            kl_best_pot = best.potential_V
            kl_r2_best  = best.r_squared
        else:
            kl_best_jk = kl_best_pot = kl_r2_best = float("nan")
        kl_mean_n = kl_full_result.mean_n_electrons

        # ── CROSS-CHECK 1: EIS potential ↔ KL best-fit potential ──────
        # Physical rationale: R_ct and j_kinetic should be compared at
        # the *same* potential for the Butler-Volmer relationship to hold.
        # A gap > 100 mV introduces systematic error in the comparison.
        if np.isfinite(kl_best_pot) and kl_best_pot > 0:
            gap = abs(eis_potential - kl_best_pot)
            if gap > 0.10:
                penalty += _SEVERITY["medium"]
                warnings.append(
                    f"EIS measured at {eis_potential:.3f} V but KL best fit is at "
                    f"{kl_best_pot:.3f} V (Δ = {gap*1000:.0f} mV). "
                    f"R_ct and j_kinetic should be compared at the same potential "
                    f"for a valid Butler-Volmer cross-check. "
                    f"Consider repeating EIS at {kl_best_pot:.3f} V."
                )

        # ── CROSS-CHECK 2: j_kinetic ↔ R_ct anticorrelation ──────────
        # Physical rationale: via Butler-Volmer, j_k ∝ exp(-ΔG/RT) and
        # R_ct ∝ RT/(n·F·j_k).  High j_k with high R_ct is physically
        # inconsistent — either the EIS model is wrong or potentials differ.
        # Threshold: j_k > 10 mA/cm² implies R_ct < ~50 Ω at 25°C (BV).
        if np.isfinite(kl_best_jk) and np.isfinite(r_ct) and r_ct > 0:
            if kl_best_jk > 10.0 and r_ct > 100.0:
                penalty += _SEVERITY["high"]
                warnings.append(
                    f"Physically inconsistent: j_kinetic = {kl_best_jk:.2f} mA/cm² "
                    f"(high) but R_ct = {r_ct:.1f} Ω (high). "
                    f"Via Butler-Volmer, R_ct × j_k should be ~26 mV at 25°C. "
                    f"Possible causes: different measurement potentials, "
                    f"wrong EIS circuit model, or R_ct mis-assignment. "
                    f"Suggested circuit: {suggested_circuit.notation}."
                )
            elif kl_best_jk < 0.5 and r_ct < 5.0 and region == "post-onset":
                # Opposite inconsistency: very low j_k but very low R_ct
                penalty += _SEVERITY["low"]
                warnings.append(
                    f"Low j_kinetic ({kl_best_jk:.3f} mA/cm²) with very low "
                    f"R_ct ({r_ct:.2f} Ω) — activity may be mass-transport "
                    f"limited rather than kinetically limited. "
                    f"Check KL fit quality (R² = {kl_r2_best:.3f})."
                )

        # ── CROSS-CHECK 3: n_electrons vs. catalyst family ────────────
        # Physical rationale: each catalyst family has a characteristic
        # n range set by its reaction mechanism.
        if np.isfinite(kl_mean_n):
            if ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
                # Expected: 2–6 electrons for AOR on Pt/Pd/PtRu
                if kl_mean_n < 1.5:
                    penalty += _SEVERITY["medium"]
                    warnings.append(
                        f"n = {kl_mean_n:.2f} is unexpectedly low for a "
                        f"{ctype} catalyst. AOR on Pt/Pd typically gives n ≥ 2. "
                        f"Verify diffusion coefficient D and bulk concentration C."
                    )
                elif kl_mean_n > 6.5:
                    penalty += _SEVERITY["low"]
                    warnings.append(
                        f"n = {kl_mean_n:.2f} exceeds the maximum for complete "
                        f"alcohol oxidation to CO2 (n = 6 for ethanol). "
                        f"Check electrode area calibration."
                    )
                elif kl_mean_n >= 5.0:
                    recommendations.append(
                        f"n = {kl_mean_n:.2f} suggests near-complete oxidation to CO2. "
                        f"Confirm with product analysis (HPLC or GC-MS)."
                    )

            elif ctype == CATALYST_METAL_OXIDE:
                # Metal oxides in AOR: n commonly 2–4 via surface redox
                if kl_mean_n < 1.0:
                    penalty += _SEVERITY["high"]
                    warnings.append(
                        f"n = {kl_mean_n:.2f} for a metal oxide is too low. "
                        f"The M(OH)x ⇌ MOOx redox conversion itself does not "
                        f"produce Faradaic current proportional to alcohol — "
                        f"confirm the signal is not purely capacitive."
                    )
                elif kl_mean_n > 4.5:
                    recommendations.append(
                        f"n = {kl_mean_n:.2f} for a metal oxide is higher than "
                        f"typical. Verify no co-oxidation of surface species "
                        f"contributes to the limiting current."
                    )

            elif ctype == CATALYST_METAL_FREE:
                # N-doped carbons in AOR: n typically 1.5–3
                if kl_mean_n > 3.5:
                    penalty += _SEVERITY["low"]
                    recommendations.append(
                        f"n = {kl_mean_n:.2f} for a metal-free catalyst is above "
                        f"the expected 1.5–3 range. "
                        f"Rule out trace metal contamination (ICP-MS check)."
                    )

        # ── CROSS-CHECK 4: I_f/I_b (CV) combined with n (KL) ─────────
        # Physical rationale: low I_f/I_b signals CO accumulation on
        # the surface (incomplete oxidation), which should also depress
        # n.  Both being low simultaneously is a strong poisoning flag.
        if (
            ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY)
            and np.isfinite(cv_result.if_ib_ratio)
            and np.isfinite(kl_mean_n)
        ):
            if cv_result.if_ib_ratio < 0.80 and kl_mean_n < 2.0:
                penalty += _SEVERITY["high"]
                warnings.append(
                    f"Combined poisoning flag: I_f/I_b = {cv_result.if_ib_ratio:.2f} "
                    f"(CO/intermediate accumulation) AND n = {kl_mean_n:.2f} "
                    f"(incomplete oxidation). This combination strongly indicates "
                    f"aldehyde or CO poisoning of active sites."
                )
                recommendations.append(
                    "Perform CO-stripping CV to quantify poisoning. "
                    "Increase temperature by 10–20°C or switch to PtRu alloy to "
                    "improve CO tolerance via bifunctional mechanism."
                )
            elif cv_result.if_ib_ratio < 0.80:
                # Poisoning from CV alone
                penalty += _SEVERITY["medium"]
                warnings.append(
                    f"I_f/I_b = {cv_result.if_ib_ratio:.2f} indicates intermediate "
                    f"accumulation. {suggested_circuit.notation} can resolve the "
                    f"poisoning arc in the Nyquist plot."
                )

        # ── Structural recommendations ────────────────────────────────
        recommendations.append(
            f"Suggested circuit for {region} ({electrolyte}): "
            f"{suggested_circuit.notation} — {suggested_circuit.rationale}"
        )

        if region == "pre-onset":
            recommendations.append(
                f"Repeat both EIS and KL at E_onset = {cv_result.e_onset:.3f} V "
                f"to measure intrinsic kinetics under active reaction conditions."
            )

        if ctype == CATALYST_METAL_FREE and region in ("onset", "post-onset"):
            recommendations.append(
                f"For porous carbon catalysts, confirm that {suggested_circuit.notation} "
                f"includes an inter-particle contact element (R1) distinct from the "
                f"faradaic charge-transfer element (R2/CPE2)."
            )

        # ── Final score ───────────────────────────────────────────────
        consistency_score = max(0.0, 1.0 - min(penalty, 1.0))

        return ComprehensiveReport(
            catalyst_type     = ctype,
            alcohol           = alcohol,
            electrolyte       = electrolyte,
            e_onset           = cv_result.e_onset,
            if_ib_ratio       = cv_result.if_ib_ratio,
            ecsa_cm2          = getattr(cv_result, "ecsa_cm2", None),
            eis_potential     = eis_potential,
            r_ct              = r_ct if np.isfinite(r_ct) else float("nan"),
            eis_region        = region,
            suggested_circuit = suggested_circuit,
            kl_mean_n         = kl_mean_n,
            kl_best_jk        = kl_best_jk,
            kl_best_potential = kl_best_pot,
            kl_r2_best        = kl_r2_best,
            warnings          = warnings,
            recommendations   = recommendations,
            consistency_score = consistency_score,
            cv_result         = cv_result,
            eis_fit_result    = eis_fit_result,
            kl_full_result    = kl_full_result,
        )
