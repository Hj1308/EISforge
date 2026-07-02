"""
EISForge — Rule-Based EIS Fit Interpretation
Author: Hoda Jafari | July 2026

Deterministic, physics-based interpretation of CNLS fit results.
No ML involved: every statement below follows from standard EIS theory,
so the output is transparent and reviewable — appropriate for a
publication-support tool. (The physics-informed transformer in
eisforge/ml/eis_gpt is implemented but untrained; until it is trained
and validated, this module is the interpretation engine.)

Key relations used
------------------
Effective capacitance of a (R || CPE) sub-circuit (Brug et al. 1984,
single-time-constant form):

    C_eff = Q**(1/n) * R**((1-n)/n)

Time constant and characteristic frequency:

    tau = R * C_eff        f_c = 1 / (2*pi*tau)

Interpretation language is deliberately descriptive ("consistent with",
"suggests") rather than mechanistic — consistent with the project's
scientific-honesty standard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProcessInfo:
    """One faradaic/relaxation process extracted from the fit."""
    label: str                       # e.g. "R1 || CPE1"
    resistance: float                # Ohm
    c_eff: Optional[float] = None    # F (None if no CPE/C paired)
    cpe_n: Optional[float] = None
    tau: Optional[float] = None      # s
    f_char: Optional[float] = None   # Hz


@dataclass
class EISInterpretation:
    r_series: Optional[float] = None
    processes: List[ProcessInfo] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines: List[str] = []
        if self.r_series is not None:
            lines.append(
                f"- **Series (solution) resistance** R_s = {self.r_series:.3g} Ω "
                f"(high-frequency intercept; use as R_s for iR correction)."
            )
        for p in self.processes:
            seg = f"- **{p.label}**: R = {p.resistance:.4g} Ω"
            if p.c_eff is not None:
                seg += f", C_eff ≈ {p.c_eff:.3g} F (Brug)"
            if p.cpe_n is not None:
                seg += f", n = {p.cpe_n:.3f}"
            if p.tau is not None:
                seg += f", τ ≈ {p.tau:.3g} s (f_c ≈ {p.f_char:.3g} Hz)"
            lines.append(seg)
        for f_ in self.findings:
            lines.append(f"- {f_}")
        for w in self.warnings:
            lines.append(f"- ⚠ {w}")
        return "\n".join(lines) if lines else "- No interpretable parameters found."


def _brug_c_eff(q: float, n: float, r: float) -> Optional[float]:
    """Effective capacitance of (R || CPE), Brug single-τ formula."""
    if q <= 0 or r <= 0 or not (0.0 < n <= 1.0):
        return None
    try:
        return float(q ** (1.0 / n) * r ** ((1.0 - n) / n))
    except (OverflowError, ZeroDivisionError, ValueError):
        return None


def interpret_fit(
    parameters: Dict[str, float],
    circuit_string: str = "",
    chi_squared: Optional[float] = None,
) -> EISInterpretation:
    """Interpret a CNLS FitResult.parameters dict (impedance.py naming).

    Recognised parameter names: R0, R1, ... | CPEk_0 (Q), CPEk_1 (n) |
    Ck | Lk | Wok_0/Wok_1, Wsk_0/Wsk_1, Wk.
    """
    out = EISInterpretation()

    # ── group parameters by element ─────────────────────────────────────────
    rs: Dict[int, float] = {}
    cpe_q: Dict[int, float] = {}
    cpe_n: Dict[int, float] = {}
    caps: Dict[int, float] = {}
    inductors: Dict[int, float] = {}
    has_warburg = False

    for name, val in parameters.items():
        m = re.fullmatch(r"R(\d+)", name)
        if m:
            rs[int(m.group(1))] = float(val)
            continue
        m = re.fullmatch(r"CPE(\d+)_0", name)
        if m:
            cpe_q[int(m.group(1))] = float(val)
            continue
        m = re.fullmatch(r"CPE(\d+)_1", name)
        if m:
            cpe_n[int(m.group(1))] = float(val)
            continue
        m = re.fullmatch(r"C(\d+)", name)
        if m:
            caps[int(m.group(1))] = float(val)
            continue
        m = re.fullmatch(r"L(\d+)", name)
        if m:
            inductors[int(m.group(1))] = float(val)
            continue
        if re.fullmatch(r"W[os]?(\d+)(_\d+)?", name):
            has_warburg = True

    if not rs:
        out.warnings.append("No resistance parameters recognised — cannot interpret.")
        return out

    # ── series resistance = lowest-index R ───────────────────────────────────
    r_idx_sorted = sorted(rs)
    rs_idx = r_idx_sorted[0]
    out.r_series = rs[rs_idx]

    # ── faradaic processes: each remaining R, paired with CPE/C of same idx ──
    for idx in r_idx_sorted[1:]:
        r_val = rs[idx]
        p = ProcessInfo(label=f"R{idx}", resistance=r_val)
        if idx in cpe_q and idx in cpe_n:
            p.label = f"R{idx} ‖ CPE{idx}"
            p.cpe_n = cpe_n[idx]
            c_eff = _brug_c_eff(cpe_q[idx], cpe_n[idx], abs(r_val))
            if c_eff is not None:
                p.c_eff = c_eff
                p.tau = abs(r_val) * c_eff
                p.f_char = 1.0 / (2.0 * np.pi * p.tau) if p.tau > 0 else None
        elif idx in caps:
            p.label = f"R{idx} ‖ C{idx}"
            p.c_eff = caps[idx]
            p.tau = abs(r_val) * caps[idx]
            p.f_char = 1.0 / (2.0 * np.pi * p.tau) if p.tau > 0 else None
        out.processes.append(p)

    # ── findings ──────────────────────────────────────────────────────────────
    n_proc = len(out.processes)
    if n_proc >= 2:
        taus = [p.tau for p in out.processes if p.tau]
        if len(taus) >= 2 and max(taus) / max(min(taus), 1e-30) > 10:
            out.findings.append(
                f"{n_proc} relaxation processes with well-separated time constants "
                f"— consistent with distinct interfacial processes (e.g. film/contact "
                f"at high frequency, charge transfer at low frequency)."
            )
        else:
            out.findings.append(
                f"{n_proc} relaxation processes with overlapping time constants — "
                f"the two arcs may not be independently resolvable; treat individual "
                f"R/C values with caution."
            )

    dispersive = [p for p in out.processes if p.cpe_n is not None and p.cpe_n < 0.80]
    if dispersive:
        out.findings.append(
            "CPE exponent n < 0.80 — strong frequency dispersion, consistent with "
            "a porous/heterogeneous electrode surface (typical of high-surface-area "
            "carbons); Brug C_eff values are approximate in this regime."
        )

    negative_r = [p for p in out.processes if p.resistance < 0]
    if negative_r:
        out.findings.append(
            "Negative faradaic resistance — NDR fingerprint of adsorbed-intermediate "
            "relaxation (AOR); the low-frequency arc crosses into the negative-Re(Z) "
            "region. Interpret with the pseudo-inductive AOR framework."
        )

    if inductors:
        out.findings.append(
            "Inductive element present — low-frequency pseudo-inductive loop, "
            "consistent with relaxation of adsorbed reaction intermediates."
        )

    if has_warburg:
        out.findings.append(
            "Warburg element present — mass-transport (diffusion) contribution "
            "at low frequency."
        )

    # ── fit-quality note ──────────────────────────────────────────────────────
    if chi_squared is not None and np.isfinite(chi_squared):
        if chi_squared < 1e-3:
            out.findings.append(
                f"Reduced χ² = {chi_squared:.2e} (modulus-weighted) — excellent fit."
            )
        elif chi_squared < 1e-2:
            out.findings.append(
                f"Reduced χ² = {chi_squared:.2e} — acceptable fit; inspect residuals "
                f"at frequency extremes."
            )
        else:
            out.warnings.append(
                f"Reduced χ² = {chi_squared:.2e} — poor fit; the circuit topology "
                f"may be inadequate for this spectrum."
            )

    # ── sanity warnings ───────────────────────────────────────────────────────
    for p in out.processes:
        if p.c_eff is not None and p.c_eff > 1.0:
            out.warnings.append(
                f"{p.label}: C_eff = {p.c_eff:.2g} F is unphysically large for an "
                f"interfacial capacitance — check units and parameter degeneracy."
            )
    if out.r_series is not None and out.r_series > 1e4:
        out.warnings.append(
            "R_s > 10 kΩ — unusually high solution resistance; check reference "
            "electrode contact and cell setup."
        )

    return out
