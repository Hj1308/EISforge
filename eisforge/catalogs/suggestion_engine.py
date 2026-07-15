"""
Circuit suggestion engine — ranks MULTIPLE candidate equivalent circuits
against real data instead of returning a single default.

Why this exists
----------------
`lookup_circuit()` in circuit_models.py returns exactly one model. For AOR
data this is scientifically risky: the same Nyquist shape (a capacitive arc
with a low-frequency upturn) can come from a second charge-transfer process,
a finite Warburg tail, OR the onset of the pseudo-inductive intermediate-
relaxation loop that is the AOR kinetic fingerprint (see
eisforge/catalogs/circuit_models.py::AOR_PSEUDOINDUCTIVE and the project
knowledge base, Section 4.2). Presenting only one option lets
the person accept a wrong topology without ever seeing the alternative.

This module fits several catalog candidates to the SAME data and ranks them
by corrected Akaike Information Criterion (AICc), the standard model-
selection statistic for CNLS/EIS fitting: it rewards goodness of fit but
penalises extra free parameters, with a small-sample correction so a
4-parameter model isn't spuriously preferred over a 6-parameter one just
because it has fewer degrees of freedom to overfit with.

    AIC  = 2k + n*ln(RSS/n)
    AICc = AIC + 2k(k+1)/(n-k-1)          (n = 2 * n_frequencies, k = n_params)

Lower AICc = better trade-off between fit quality and complexity. Models are
reported with ΔAICc relative to the best one; ΔAICc < 2 is "essentially
equivalent support", 4-7 "considerably less support", >10 "no support"
(Burnham & Anderson, standard information-theoretic convention).

Author: Claude (for Hoda Jafari) | July 2026
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from eisforge.catalogs.circuit_models import CircuitCatalog, CircuitModel
from eisforge.core.fitter import CNLSFitter, FitResult
from eisforge.parsers.base_parser import EISDataset

try:
    from eisforge.analysis.cv_analyzer import (
        CATALYST_NOBLE_METAL,
        CATALYST_ALLOY,
        CATALYST_METAL_OXIDE,
        CATALYST_METAL_FREE,
    )
except ImportError:  # pragma: no cover
    CATALYST_NOBLE_METAL = "noble_metal"
    CATALYST_ALLOY = "alloy"
    CATALYST_METAL_OXIDE = "metal_oxide"
    CATALYST_METAL_FREE = "carbon_material"


# ── Candidate sets per catalyst family ───────────────────────────────────────
# Always includes the family's default (from lookup_circuit's map) PLUS
# reasonable alternatives so the person actually gets a choice. AOR
# inductive/NDR candidates are appended when the corresponding hint is set,
# never silently substituted for the electrochemist's default set.

def _base_candidates(catalyst_type: str) -> list[CircuitModel]:
    if catalyst_type == CATALYST_METAL_FREE:
        return [
            CircuitCatalog.POROUS_CARBON,
            CircuitCatalog.POROUS_CARBON_CONTACT,
            CircuitCatalog.RANDLES_SIMPLE,
            CircuitCatalog.TWO_RC,
        ]
    if catalyst_type == CATALYST_METAL_OXIDE:
        return [
            CircuitCatalog.OXIDE_TWO_STEP,
            CircuitCatalog.RANDLES_SIMPLE,
            CircuitCatalog.RANDLES_WARBURG,
        ]
    if catalyst_type == CATALYST_ALLOY:
        return [
            CircuitCatalog.ALLOY_TWO_STEP,
            CircuitCatalog.TWO_RC,
            CircuitCatalog.TWO_RC_WARBURG,
        ]
    # noble metal / default
    return [
        CircuitCatalog.RANDLES_SIMPLE,
        CircuitCatalog.RANDLES_WARBURG,
        CircuitCatalog.TWO_RC,
        CircuitCatalog.TWO_RC_WARBURG,
    ]


def candidate_models(
    catalyst_type: str = CATALYST_METAL_FREE,
    inductive_loop: bool = False,
    negative_resistance: bool = False,
) -> list[CircuitModel]:
    """Return the deduplicated candidate list for this catalyst family.

    `inductive_loop` / `negative_resistance` are person-set flags (from
    visually inspecting the Nyquist plot for a 4th- or 2nd-quadrant
    low-frequency arc) — they ADD the AOR-specific candidates rather than
    replacing the family defaults, so a genuinely simple Randles arc is
    still offered alongside the inductive hypothesis.
    """
    models = list(_base_candidates(catalyst_type))
    if negative_resistance:
        models.append(CircuitCatalog.AOR_NDR)
    elif inductive_loop:
        models.append(CircuitCatalog.AOR_PSEUDOINDUCTIVE)
    # de-duplicate by notation, preserve order
    seen, out = set(), []
    for m in models:
        if m.notation not in seen:
            seen.add(m.notation)
            out.append(m)
    return out


# ── p0 heuristics ─────────────────────────────────────────────────────────

def _required_param_count(notation: str, max_n: int = 10) -> int:
    """Probe impedance.py for how many scalar parameters a circuit string
    needs (it validates length against a dummy initial_guess internally but
    won't report the count directly on failure in a parseable way across
    versions, so we brute-force the smallest working n)."""
    from impedance.models.circuits import CustomCircuit
    for n in range(1, max_n + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                CustomCircuit(circuit=notation, initial_guess=[1.0] * n)
            return n
        except Exception:
            continue
    raise ValueError(f"Could not determine parameter count for '{notation}'")


def default_p0(
    model: CircuitModel,
    freq: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
) -> list[float]:
    """Data-scaled initial guess, generic across circuit topologies.

    Reads the impedance.py parameter names (e.g. 'R0', 'CPE1_0', 'CPE1_1',
    'L2', 'Wo1_0') and assigns a physically-scaled starting value per
    element type:
        first R   -> high-frequency Re(Z) (series/solution resistance)
        other R   -> a fraction of the total Re(Z) span (charge transfer)
        CPE_0 (Q) -> 1e-4 (typical double-layer order of magnitude)
        CPE_1 (n) -> 0.85
        L         -> small positive inductance (10 Ω·s, mid-range)
        W / Wo/Ws -> Re(Z) span (diffusion resistance-like scale)
    """
    from impedance.models.circuits import CustomCircuit

    n_params = _required_param_count(model.notation)
    probe = CustomCircuit(circuit=model.notation, initial_guess=[1.0] * n_params)
    names = probe.get_param_names()[0]

    z_span = float(np.ptp(z_real)) or 1.0
    hf_idx = int(np.argmax(freq))          # highest frequency point
    r_series = float(max(z_real[hf_idx], 1e-6))

    p0 = []
    r_seen = False
    for name in names:
        if name.startswith("R") and "_" not in name:
            if not r_seen:
                p0.append(r_series)
                r_seen = True
            else:
                p0.append(max(z_span * 0.5, 1.0))
        elif "_0" in name and name.startswith("CPE"):
            p0.append(1e-4)
        elif "_1" in name and name.startswith("CPE"):
            p0.append(0.85)
        elif name.startswith("L"):
            p0.append(10.0)
        elif name.startswith(("W", "Wo", "Ws")) and "_0" in name:
            p0.append(max(z_span, 1.0))
        elif "_1" in name:  # Warburg time constant, etc.
            p0.append(1.0)
        elif name.startswith(("W", "Wo", "Ws")):
            p0.append(max(z_span, 1.0))
        else:
            p0.append(1.0)
    return p0


# ── Ranking ────────────────────────────────────────────────────────────────

@dataclass
class CircuitSuggestion:
    model: CircuitModel
    fit_result: Optional[FitResult]
    aicc: float
    delta_aicc: float = 0.0
    converged: bool = False
    n_params: int = 0
    error: str = ""

    def support_label(self) -> str:
        """Burnham & Anderson (2002) rule-of-thumb interpretation of ΔAICc."""
        if not self.converged:
            return "fit failed"
        if self.delta_aicc < 2:
            return "essentially equivalent support"
        if self.delta_aicc < 4:
            return "some support"
        if self.delta_aicc < 7:
            return "considerably less support"
        return "no support"

    def __str__(self) -> str:
        if not self.converged:
            return f"{self.model.name} — fit failed ({self.error})"
        return (f"{self.model.name} ({self.model.notation}) — "
                f"AICc={self.aicc:.2f}, ΔAICc={self.delta_aicc:.2f} "
                f"[{self.support_label()}], k={self.n_params}")


def _aicc(rss: float, n_data: int, k_params: int) -> float:
    """Corrected AIC for least-squares regression with Gaussian errors.
    n_data counts REAL scalars (2 per frequency point: Re and Im)."""
    rss = max(rss, 1e-300)
    aic = 2 * k_params + n_data * np.log(rss / n_data)
    denom = n_data - k_params - 1
    if denom <= 0:
        # too few points for the small-sample correction; fall back to AIC
        return aic
    return aic + (2 * k_params * (k_params + 1)) / denom


def _fit_one(model: CircuitModel, dataset: EISDataset,
            allow_negative_r: bool) -> CircuitSuggestion:
    try:
        p0 = default_p0(model, dataset.frequency, dataset.z_real, dataset.z_imag)
    except Exception as e:
        return CircuitSuggestion(model=model, fit_result=None, aicc=float("inf"),
                                 error=f"p0 generation failed: {e}")

    fitter = CNLSFitter(model.notation, p0, allow_negative_r=allow_negative_r,
                        remove_outliers=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fitter.fit(dataset)
    except Exception as e:
        return CircuitSuggestion(model=model, fit_result=None, aicc=float("inf"),
                                 n_params=len(p0), error=str(e))

    if not res.converged or res.z_fit is None:
        return CircuitSuggestion(model=model, fit_result=res, aicc=float("inf"),
                                 n_params=len(p0), converged=False,
                                 error="did not converge")

    n_data = 2 * len(dataset.frequency)
    k = len(p0)
    resid = dataset.z_complex - res.z_fit
    rss = float(np.sum(resid.real ** 2 + resid.imag ** 2))
    aicc = _aicc(rss, n_data, k)
    return CircuitSuggestion(model=model, fit_result=res, aicc=aicc,
                             n_params=k, converged=True)


def suggest_circuits(
    frequency: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    catalyst_type: str = CATALYST_METAL_FREE,
    inductive_loop: bool = False,
    negative_resistance: bool = False,
    extra_models: Optional[list[CircuitModel]] = None,
) -> list[CircuitSuggestion]:
    """Fit every candidate circuit to the SAME data and return them ranked
    best-first by AICc, each annotated with ΔAICc vs. the best fit.

    `z_imag` follows the EISDataset / project convention: −Im(Z).
    Set negative_resistance=True (implies allow_negative_r on the fitter)
    when the low-frequency Nyquist arc visibly enters the 2nd quadrant.
    """
    dataset = EISDataset(frequency=np.asarray(frequency, dtype=float),
                         z_real=np.asarray(z_real, dtype=float),
                         z_imag=np.asarray(z_imag, dtype=float),
                         metadata={})
    dataset.validate_shapes()

    models = candidate_models(catalyst_type, inductive_loop, negative_resistance)
    if extra_models:
        seen = {m.notation for m in models}
        models += [m for m in extra_models if m.notation not in seen]

    allow_neg = bool(negative_resistance)
    results = [_fit_one(m, dataset, allow_neg) for m in models]

    finite = [r for r in results if np.isfinite(r.aicc)]
    if finite:
        best = min(r.aicc for r in finite)
        for r in results:
            r.delta_aicc = (r.aicc - best) if np.isfinite(r.aicc) else float("inf")
    results.sort(key=lambda r: r.aicc)
    return results


__all__ = [
    "CircuitSuggestion",
    "candidate_models",
    "default_p0",
    "suggest_circuits",
]
