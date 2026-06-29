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
from typing import Dict, List, Optional, Tuple

import numpy as np

from eisforge.analysis.cv_analyzer import (
    CVAnalysisResult,
    CATALYST_NOBLE_METAL,
    CATALYST_ALLOY,
    CATALYST_METAL_OXIDE,
    CATALYST_METAL_FREE,
)
from eisforge.catalogs.circuit_models import CircuitModel, lookup_circuit
from eisforge.core.fitter import FitResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default R_ct thresholds (Ohm) — keyed by catalyst type and EIS region.
# Users can override these by passing `r_ct_thresholds` to __init__.
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    CATALYST_NOBLE_METAL: {
        "pre_onset_low":   100.0,
        "post_onset_high": 5000.0,
    },
    CATALYST_ALLOY: {
        "pre_onset_low":   100.0,
        "post_onset_high": 5000.0,
    },
    CATALYST_METAL_OXIDE: {
        "pre_onset_low":    50.0,
        "post_onset_high":  None,
    },
    CATALYST_METAL_FREE: {
        "pre_onset_low":  500.0,
        "post_onset_high": 50000.0,
    },
}

# Penalty weights for weighted consistency scoring.
# Users can override per-project via `severity_weights` in __init__.
_DEFAULT_SEVERITY_WEIGHTS: Dict[str, float] = {
    "low": 0.1,
    "medium": 0.2,
    "high": 0.35,
}

# ---------------------------------------------------------------------------
# Alkaline-environment threshold correction factors.
#
# In alkaline electrolyte (KOH / NaOH), solution resistance is lower and
# ORR / alcohol-oxidation kinetics are faster than in acid.  As a result:
#   - The *lower* R_ct bound is relaxed by ALKALINE_LOW_FACTOR (×0.5),
#     because a small R_ct is no longer anomalous.
#   - The *upper* R_ct bound is widened by ALKALINE_HIGH_FACTOR (×4.0),
#     because passivation or poisoning is less common and a higher R_ct
#     must be tolerated before triggering a warning.
#
# These are empirically derived from published RRDE/EIS datasets on Pt/C
# and N-doped carbon catalysts in 0.1 M KOH vs 0.1 M HClO4.  They are
# intentionally exposed as module-level constants so users who work in
# different concentration regimes can monkey-patch them before
# instantiating EISCVCorrelator.
# ---------------------------------------------------------------------------
ALKALINE_LOW_FACTOR: float = 0.5   # low threshold ×0.5 in alkaline
ALKALINE_HIGH_FACTOR: float = 4.0  # high threshold ×4.0 in alkaline


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EISCVCorrelationResult:
    eis_potential     : float
    e_onset           : float
    eis_region        : str
    r_ct              : float
    i_forward_peak    : float
    catalyst_type     : str              = CATALYST_NOBLE_METAL
    consistency_score : float            = 1.0
    warnings          : List[str]        = field(default_factory=list)
    recommendations   : List[str]        = field(default_factory=list)
    suggested_circuit : Optional[CircuitModel] = field(default=None)

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
        # Guard: only access CircuitModel fields if the object has them,
        # so a future schema change in circuit_models.py raises a clear
        # AttributeError rather than a silent KeyError or missing output.
        if self.suggested_circuit is not None:
            notation = getattr(self.suggested_circuit, "notation", None)
            name     = getattr(self.suggested_circuit, "name", None)
            if notation:
                lines.append(f"  Suggested circuit: {notation}")
            if name:
                lines.append(f"  Circuit name     : {name}")
            if notation is None and name is None:
                lines.append(
                    "  Suggested circuit: [CircuitModel missing 'notation'/'name' fields]"
                )
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


# ---------------------------------------------------------------------------
# Correlator class
# ---------------------------------------------------------------------------

class EISCVCorrelator:
    """
    Cross-technique correlator between EIS fit results and CV analysis.

    Checks whether the EIS measurement potential is consistent with the
    CV-derived E_onset, and whether R_ct values make physical sense for
    the given catalyst type and electrolyte environment.

    The correlator also looks up the recommended equivalent-circuit topology
    from ``eisforge.catalogs.circuit_models.CIRCUIT_MAP`` based on the
    catalyst type, EIS region, and electrolyte.  Circuit recommendations
    are returned as structured ``CircuitModel`` objects — not plain strings —
    so they can be passed directly to ``CNLSFitter.fit()``.

    Parameters
    ----------
    onset_tolerance : float
        Potential window (V) around E_onset that defines the 'onset' region.
        Default is 0.05 V.
    electrolyte : str
        Electrolyte environment: ``'acidic'`` or ``'alkaline'``.
    r_ct_thresholds : dict, optional
        Override the built-in R_ct threshold dictionary. Structure::

            {
                catalyst_type: {
                    "pre_onset_low":   <float or None>,
                    "post_onset_high": <float or None>,
                }
            }

        Any catalyst type not provided falls back to built-in defaults.
    severity_weights : dict, optional
        Override the penalty weights used in consistency scoring.  Useful
        when one class of warning should dominate the score for a specific
        project.  Structure::

            {"low": <float>, "medium": <float>, "high": <float>}

        Defaults to ``_DEFAULT_SEVERITY_WEIGHTS``
        (low=0.10, medium=0.20, high=0.35).
    """

    def __init__(
        self,
        onset_tolerance  : float = 0.05,
        electrolyte      : str   = "acidic",
        r_ct_thresholds  : Optional[Dict[str, Dict[str, float]]] = None,
        severity_weights : Optional[Dict[str, float]] = None,
    ):
        self.onset_tolerance = onset_tolerance
        self.electrolyte     = electrolyte.lower().strip()

        self._thresholds: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in _DEFAULT_THRESHOLDS.items()
        }
        if r_ct_thresholds:
            for cat, thr in r_ct_thresholds.items():
                if cat in self._thresholds:
                    self._thresholds[cat].update(thr)
                else:
                    self._thresholds[cat] = thr

        # Merge user-supplied severity weights with defaults so partial
        # overrides (e.g. only "high") still work.
        self._severity_weights: Dict[str, float] = dict(_DEFAULT_SEVERITY_WEIGHTS)
        if severity_weights:
            self._severity_weights.update(severity_weights)

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
            Includes ``suggested_circuit`` (a ``CircuitModel``) that can be
            passed directly to ``CNLSFitter.fit(circuit=result.suggested_circuit.notation)``.

        Raises
        ------
        TypeError
            If ``cv_result`` or ``eis_fit_result`` are not the expected types.
        ValueError
            If ``eis_fit_result.parameters`` is missing/empty, or required
            attributes on ``cv_result`` are not finite numbers.
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
        penalty_total   = 0.0

        ctype = cv_result.catalyst_type

        # ── Determine EIS region relative to E_onset ───────────────────────
        delta = eis_potential - cv_result.e_onset
        if delta < -self.onset_tolerance:
            region = "pre-onset"
        elif abs(delta) <= self.onset_tolerance:
            region = "onset"
        else:
            region = "post-onset"

        # ── Extract R_ct from EIS fit ──────────────────────────────────────
        r_ct = self._extract_r_ct(eis_fit_result)

        # ── Lookup recommended equivalent circuit ──────────────────────────
        suggested_circuit = lookup_circuit(ctype, region, self.electrolyte)

        # Guard: verify CircuitModel has the fields we will use in the report
        # and in recommendations below.  A missing field means circuit_models.py
        # was changed without updating this module.
        for _attr in ("notation", "name", "rationale"):
            if not hasattr(suggested_circuit, _attr):
                logger.warning(
                    "CircuitModel returned by lookup_circuit() is missing "
                    "attribute '%s'. Check eisforge/catalogs/circuit_models.py.",
                    _attr,
                )

        _circuit_notation = getattr(suggested_circuit, "notation", "?")
        _circuit_name     = getattr(suggested_circuit, "name", "?")
        _circuit_rationale = getattr(suggested_circuit, "rationale", "")

        recommendations.append(
            f"Suggested equivalent circuit: {_circuit_notation} "
            f"({_circuit_name}). "
            f"{_circuit_rationale}"
        )

        # ── R_ct consistency check ─────────────────────────────────────────
        if not np.isnan(r_ct):
            thr = self._thresholds.get(ctype, {})
            low  = thr.get("pre_onset_low")
            high = thr.get("post_onset_high")

            # Apply alkaline correction using named constants.
            # See ALKALINE_LOW_FACTOR / ALKALINE_HIGH_FACTOR at module level
            # for the physical rationale and reference conditions.
            if self.electrolyte == "alkaline":
                if low  is not None: low  = low  * ALKALINE_LOW_FACTOR
                if high is not None: high = high * ALKALINE_HIGH_FACTOR

            if ctype == CATALYST_METAL_FREE:
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += self._severity_weights["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unusually low for a metal-free "
                        f"catalyst at E < E_onset ({self.electrolyte} environment). "
                        f"Verify the observed peak is faradaic and not capacitive background."
                    )
                elif region == "post-onset" and high is not None and r_ct > high:
                    penalty_total += self._severity_weights["medium"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is very large — possible surface "
                        f"passivation or pore blocking in {self.electrolyte} electrolyte."
                    )

            elif ctype == CATALYST_METAL_OXIDE:
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += self._severity_weights["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is unexpectedly small for a metal "
                        f"oxide at E < E_onset ({self.electrolyte} environment). "
                        f"Confirm EIS was measured at the correct potential."
                    )

            else:  # noble_metal and alloy
                if region == "pre-onset" and low is not None and r_ct < low:
                    penalty_total += self._severity_weights["high"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is small for E < E_onset "
                        f"({cv_result.e_onset:.3f} V, {self.electrolyte} environment). "
                        f"Expected R_ct >> {low:.0f} Ohm in the pre-onset region."
                    )
                elif region == "post-onset" and high is not None and r_ct > high:
                    penalty_total += self._severity_weights["medium"]
                    warnings.append(
                        f"R_ct = {r_ct:.1f} Ohm is large for E > E_onset "
                        f"({self.electrolyte} environment). "
                        f"Possible CO poisoning or mass-transport limitation."
                    )

        # ── I_f/I_b check ─────────────────────────────────────────────────
        if_ib = cv_result.if_ib_ratio

        if ctype in (CATALYST_NOBLE_METAL, CATALYST_ALLOY):
            if not np.isnan(if_ib):
                if if_ib < 1.0:
                    penalty_total += self._severity_weights["medium"]
                    recommendations.append(
                        f"I_f/I_b = {if_ib:.2f} < 1.0 — CO poisoning suspected "
                        f"({self.electrolyte} environment). "
                        f"The suggested circuit {_circuit_notation} is "
                        f"appropriate for resolving the poisoning arc."
                    )
                elif if_ib > 3.0:
                    # Both nan-guards are now at the top of the branch for
                    # uniform readability; the logic is identical to before
                    # because float comparisons with nan return False.
                    if not np.isnan(r_ct) and r_ct > 1000:
                        penalty_total += self._severity_weights["low"]
                        warnings.append(
                            f"I_f/I_b = {if_ib:.2f} is excellent but R_ct = {r_ct:.1f} Ohm "
                            f"is large — confirm EIS was measured at the active potential."
                        )

        elif ctype == CATALYST_METAL_FREE:
            recommendations.append(
                "Metal-free catalyst: I_f/I_b ratio is not applicable. "
                "Estimate ECSA via C_dl method from scan-rate dependence. "
                f"For EIS, use {_circuit_notation} to capture "
                "inter-particle contact resistance and pore ion diffusion."
            )

        elif ctype == CATALYST_METAL_OXIDE:
            if self.electrolyte == "alkaline":
                recommendations.append(
                    "Metal oxide catalyst (alkaline): M(OH)x ⇌ MOOx redox process dominates. "
                    f"The suggested circuit {_circuit_notation} separates outer "
                    "surface activity from inner pore-limited diffusion."
                )
            else:
                recommendations.append(
                    "Metal oxide catalyst (acidic/neutral): oxide dissolution risk. "
                    f"The suggested circuit {_circuit_notation} captures the "
                    "surface redox process. Monitor R_ct over time for dissolution evidence."
                )

        # ── Recommend repeating EIS at E_onset ─────────────────────────────
        if region == "pre-onset":
            recommendations.append(
                f"Repeat EIS at E_onset = {cv_result.e_onset:.3f} V to measure "
                f"R_ct directly under active reaction conditions."
            )

        # ── Metal-free at onset: pore diffusion advisory ───────────────────
        if ctype == CATALYST_METAL_FREE and region == "onset":
            recommendations.append(
                f"At E_onset, use {_circuit_notation}: include an inter-particle "
                "resistance element. Ion diffusion in pores may appear as a "
                "low-frequency Warburg tail — check the suggested circuit's Wo element."
            )

        # ── Weighted consistency score ──────────────────────────────────────
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
            suggested_circuit = suggested_circuit,
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
        2. Indexed resistances: ``'R1'``, ``'R_1'``, ``'R2'``, ``'R_2'``,
           ``'R[1]'``, ``'R[2]'``
        3. Fallback generic: collect all keys matching R+digit, sort numerically,
           skip index-0 (assumed Rs), return index-1.

        Parameters
        ----------
        fit : FitResult
            Converged EIS fit result with ``parameters`` dict.

        Returns
        -------
        float
            R_ct in Ohm, or ``float('nan')`` if not identifiable.
        """
        params = fit.parameters

        # Priority 1: semantic names
        for key in ("R_ct", "Rct", "R_CT", "RCT", "r_ct", "rct"):
            if key in params:
                return float(params[key])

        # Priority 2: indexed keys (R1 = R_ct convention)
        for key in ("R1", "R_1", "R[1]", "R2", "R_2", "R[2]"):
            if key in params:
                return float(params[key])

        # Priority 3: generic regex fallback
        _r_pattern = re.compile(r"^R\D*(\d+)$", re.IGNORECASE)
        indexed: list[tuple[int, float]] = []
        for k, v in params.items():
            m = _r_pattern.match(k)
            if m:
                try:
                    indexed.append((int(m.group(1)), float(v)))
                except (ValueError, TypeError):
                    continue

        if len(indexed) >= 2:
            indexed.sort(key=lambda x: x[0])
            return indexed[1][1]  # skip Rs (index 0), return R_ct (index 1)

        return float("nan")
