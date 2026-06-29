"""
EIS-CV Correlator — Cross-technique correlation between EIS and CV results.
Author: Hoda Jafari | May 2026

Supports all catalyst families:
    - noble_metal  : Pt, Pd, Au, Rh
    - alloy        : PtRu, PtSn, PdAu, PtCu
    - metal_oxide  : NiO, Co3O4, MnO2, Co2NiO4
    - carbon_material   : N-doped Carbon, CNT, rGO, graphene-based
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

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

# ---------------------------------------------------------------------------
# Default R_ct thresholds (Ohm) — keyed by catalyst type and EIS region.
# Users can override these by passing `r_ct_thresholds` to __init__.
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    CATALYST_NOBLE_METAL: {
        "pre_onset_low":  100.0,   # R_ct below this in pre-onset → suspicious
        "post_onset_high": 5000.0, # R_ct above this in post-onset → suspicious
    },
    CATALYST_ALLOY: {
        "pre_onset_low":  100.0,
        "post_onset_high": 5000.0,
    },
    CATALYST_METAL_OXIDE: {
        "pre_onset_low":   50.0,
        "post_onset_high": None,   # no upper limit defined
    },
    CATALYST_METAL_FREE: {
        "pre_onset_low":  500.0,
        "post_onset_high": 50000.0,
    },
}

# Penalty weights used by the weighted consistency scoring system.
# Severity levels: "low" → 0.1, "medium" → 0.2, "high" → 0.35
_SEVERITY_WEIGHTS = {"low": 0.1, "medium": 0.2, "high": 0.35}


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
            CATALYST_METAL_FREE  : "Metal-Free (carbon_material / N-doped C / CNT)",
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
    CV-derived E_onset, and whether R_ct values make physical sense for
    the given catalyst type and electrolyte environment.

    Parameters
    ----------
    onset_tolerance : float
        Potential window (V) around E_onset that defines the 'onset' region.
        Default is 0.05 V.
    electrolyte : str
        Electrolyte environment: ``'acidic'`` or ``'alkaline'``.
        Used to apply environment-specific R_ct thresholds and
        generate context-aware recommendations.
    r_ct_thresholds : dict, optional
        Override the built-in R_ct threshold dictionary. The expected
        structure is::

            {
                catalyst_type: {
                    "pre_onset_low":  <float or None>,
                    "post_onset_high": <float or None>,
                }
            }

        Any catalyst type not provided falls back to the built-in defaults.
    """

    def __init__(
        self,
        onset_tolerance : float = 0.05,
        electrolyte     : str   = "acidic",
        r_ct_thresholds : Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.onset_tolerance = onset_tolerance
        self.electrolyte     = electrolyte.lower().strip()

        # Merge user overrides on top of built-in defaults
        self._thresholds: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in _DEFAULT_THRESHOLDS.items()
        }
        if r_ct_thresholds:
            for cat, thr in r_ct_thresholds.items():
                if cat in self._thresholds:
                    self._thresholds[cat].update(thr)
                else:
                    self._thresholds[cat] = thr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        cv_result      : CVAnalysisResult
            Result from ``CVAnalyzer.analyze()``.
        eis_fit_result : FitResult
            Result from ``CNLSFitter.fit()``.
        eis_potential  : float
            Potential at which EIS was measured (V vs. RHE).

        Returns
        -------
        EISCVCorrelationResult

        Raises
        ------
        TypeError
            If ``cv_result`` or ``eis_fit_result`` are not the expected types.
        ValueError
            If ``eis_fit_result.parameters`` is missing or empty, or if
            required attributes on ``cv_result`` are not finite numbers.
        """
        # ── Input validation ───────────────────────────────────────────────
        if not isinstance(cv_result, CVAnalysisResult):
            raise TypeError(
                f"cv_result must be a CVAnalysisResult, got {type(cv_result).__name__}"
            )
        if not isinstance(eis_fit_result, FitResult):
            raise TypeError(
                f"eis_fit_result must be a FitResult, got {type(eis_fit_result).__name__}"
            )
        if not hasattr(eis_fit_result, "parameters") or not eis_fit_result.parameters:
            raise ValueError(
                "eis_fit_result.parameters is missing or empty. "
                "Ensure the EIS fit converged before calling correlate()."
            )
        if not np.isfinite(cv_result.e_onset):
            raise ValueError(
                "cv_result.e_onset is not a finite number. "
                "Check that the CV analysis produced a valid onset potential."
            )
        if not np.isfinite(cv_result.i_forward_peak):
            raise ValueError(
                "cv_result.i_forward_peak is not a finite number. "
                "Check that the CV analysis produced a valid forward peak current."
            )

        # ── Setup ──────────────────────────────────────────────────────────
        warnings        = []
        recommendations = []
        penalty_total   = 0.0   # accumulated weighted penalty

        ctype = cv_result.catalyst_type

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

        # ── R_ct consistency check — catalyst-type & electrolyte specific ──
        if not np.isnan(r_ct):
            thr = self._thresholds.get(ctype, {})
            low  = thr.get("pre_onset_low")
            high = thr.get("post_onset_high")

            # Alkaline electrolytes typically show lower R_ct — relax thresholds
            alkaline_factor = 0.5 if self.electrolyte == "alkaline" else 1.0
            if low is not None:
                low  = low  * alkaline_factor
            if high is not None:
                high = high * (2.0 / alkaline_factor)  # expand upper bound in alkaline

            if ctype == CATALYST_METAL_FREE:
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += _SEVERITY_WEIGHTS["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unusually low for a metal-free "
                        f"catalyst at E < E_onset ({self.electrolyte} environment). "
                        f"Verify the observed peak is faradaic and not capacitive background."
                    )
                elif region == "post-onset" and high is not None and r_ct > high:
                    penalty_total += _SEVERITY_WEIGHTS["medium"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is very large — possible surface "
                        f"passivation or pore blocking in {self.electrolyte} electrolyte."
                    )

            elif ctype == CATALYST_METAL_OXIDE:
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += _SEVERITY_WEIGHTS["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unexpectedly small for a metal "
                        f"oxide at E < E_onset ({self.electrolyte} environment). "
                        f"Confirm EIS was measured at the correct potential."
                    )

            else:  # noble_metal and alloy
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += _SEVERITY_WEIGHTS["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is small for E < E_onset "
                        f"({cv_result.e_onset:.3f} V, {self.electrolyte} environment). "
                        f"Expected R_ct >> {low:.0f} Ohm in the pre-onset region."
                    )
                elif region == "post-onset" and high is not None and r_ct > high:
                    penalty_total += _SEVERITY_WEIGHTS["medium"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is large for E > E_onset "
                        f"({self.electrolyte} environment). "
                        f"Possible CO poisoning or mass-transport limitation."
                    )

        # ── I_f/I_b check — only for metal and alloy catalysts ────────────
        if_ib = cv_result.if_ib_ratio

        if ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
            if not np.isnan(if_ib):
                if if_ib < 1.0:
                    penalty_total += _SEVERITY_WEIGHTS["medium"]
                    recommendations.append(
                        f"I_f/I_b = {if_ib:.2f} < 1.0 — CO poisoning suspected "
                        f"({self.electrolyte} environment). "
                        f"Try two-RC circuit R0-p(R1,CPE1)-p(R2,CPE2) for EIS fit."
                    )
                elif if_ib > 3.0 and not np.isnan(r_ct) and r_ct > 1000:
                    penalty_total += _SEVERITY_WEIGHTS["low"]
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
            if self.electrolyte == "alkaline":
                recommendations.append(
                    "Metal oxide catalyst (alkaline): the M(OH)x ↔ MOOx redox process "
                    "dominates in alkaline media. Use a two-RC circuit with a low-frequency "
                    "element (Warburg or finite-length diffusion) for pore ion transport. "
                    "Check that EIS was measured at the M(OH)x oxidation potential."
                )
            else:
                recommendations.append(
                    "Metal oxide catalyst (acidic/neutral): verify that two-RC circuit "
                    "captures both the surface redox process and direct surface oxidation. "
                    "Dissolution of the oxide layer is more likely in acidic media."
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
                "For carbon_material / carbon catalysts at E_onset: include an inter-particle "
                "resistance element in the EIS circuit. Ion diffusion in pores "
                "may appear as a low-frequency Warburg tail."
            )

        # ── Compute weighted consistency score ─────────────────────────────
        # Penalty is the sum of severity weights; capped so score ≥ 0.
        # This avoids the previous flat-deduction approach that could
        # produce arbitrary low scores for multi-warning cases.
        consistency_score = max(0.0, 1.0 - min(penalty_total, 1.0))

        return EISCVCorrelationResult(
            eis_potential     = eis_potential,
            e_onset           = cv_result.e_onset,
            eis_region        = region,
            r_ct              = r_ct if not np.isnan(r_ct) else 0.0,
            i_forward_peak    = cv_result.i_forward_peak,
            catalyst_type     = ctype,
            consistency_score = consistency_score,
            warnings          = warnings,
            recommendations   = recommendations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_r_ct(fit: FitResult) -> float:
        """
        Extract the charge-transfer resistance R_ct from EIS fit parameters.

        Search strategy (priority order):
        1. Exact named keys: ``'R_ct'``, ``'Rct'``, ``'R_CT'``, ``'RCT'``
           — returned immediately when found.
        2. Indexed resistances: ``'R1'``, ``'R_1'``, ``'R2'``, ``'R_2'``,
           ``'R[1]'``, ``'R[2]'`` — assumed to follow the convention that
           R0 / R_0 is the ohmic/solution resistance and R1 is R_ct.
        3. Fallback generic: collect all keys matching the pattern ``R``
           followed by a digit (case-insensitive). Sort numerically, skip
           the first element (assumed Rs), return the second.
           Returns ``float('nan')`` if fewer than two such elements exist.

        Parameters
        ----------
        fit : FitResult
            The converged EIS fit result containing a ``parameters`` dict.

        Returns
        -------
        float
            The extracted R_ct value in Ohm, or ``float('nan')`` if it
            cannot be unambiguously identified.

        Notes
        -----
        This method does **not** raise exceptions. Invalid or missing keys
        silently return ``float('nan')``, and the caller (``correlate``)
        checks for NaN before using the value.
        """
        params = fit.parameters

        # Priority 1: exact semantic names (case-insensitive scan)
        _priority_keys = ("R_ct", "Rct", "R_CT", "RCT", "r_ct", "rct")
        for key in _priority_keys:
            if key in params:
                return float(params[key])

        # Priority 2: indexed keys that conventionally map to R_ct
        _indexed_keys = ("R1", "R_1", "R[1]", "R2", "R_2", "R[2]")
        for key in _indexed_keys:
            if key in params:
                return float(params[key])

        # Priority 3: generic fallback — any R followed by one or more digits
        _r_pattern = re.compile(r"^R\D*(\d+)$", re.IGNORECASE)
        indexed: list[tuple[int, float]] = []
        for k, v in params.items():
            m = _r_pattern.match(k)
            if m:
                try:
                    indexed.append((int(m.group(1)), float(v)))
                except (ValueError, TypeError):
                    continue  # non-numeric value — skip

        if len(indexed) >= 2:
            indexed.sort(key=lambda x: x[0])  # sort by index ascending
            return indexed[1][1]  # skip index-0 (assumed Rs), return index-1

        return float("nan")
