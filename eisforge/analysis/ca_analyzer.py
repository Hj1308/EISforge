"""
EISForge — Chronoamperometry (i–t) Stability Analysis
Author: Hoda Jafari | July 2026

Descriptive stability metrics for a chronoamperometric hold, the way they
are reported in the alcohol-oxidation / electrocatalysis literature:

  * current retention (%)  = 100 * I_final / I_initial
  * steady-state current   (mean of the final tail)
  * retention at chosen hold times (e.g. 600 / 1000 / 1800 s)
  * initial drop over the first ~60 s

DELIBERATE OMISSION — no Cottrell / diffusion-coefficient fit.
The early current decay in a CA hold is dominated by double-layer
discharge and diffusional relaxation, NOT catalyst degradation, so a
Cottrell D would be physically misleading for a high-surface-area
metal-free carbon catalyst (consistent with the project's decision to
drop Cottrell D everywhere). Metrics here are descriptive only.

Sign convention: Ivium stores anodic current as negative; this module
works on |I| so an oxidation hold gives positive currents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CAResult:
    time: np.ndarray                 # s
    current: np.ndarray              # analysis current (|I|, possibly per-area)
    current_raw: np.ndarray          # raw signed current as loaded
    unit_label: str = "A"
    duration_s: float = 0.0
    i_initial: float = float("nan")  # mean of first tail_frac
    i_final: float = float("nan")    # mean of last tail_frac
    i_steady: float = float("nan")   # = i_final (steady-state estimate)
    retention_pct: float = float("nan")
    initial_drop_pct: float = float("nan")   # over first `drop_window_s`
    retention_at: Dict[float, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        L = [
            f"- **Duration**: {self.duration_s:.0f} s "
            f"({len(self.time)} points)",
            f"- **Initial current** (first-tail mean): "
            f"{self.i_initial:.4g} {self.unit_label}",
            f"- **Steady-state current** (final-tail mean): "
            f"{self.i_steady:.4g} {self.unit_label}",
            f"- **Current retention**: {self.retention_pct:.1f}%",
        ]
        for t_, r_ in sorted(self.retention_at.items()):
            L.append(f"  - retention @ {t_:.0f} s: {r_:.1f}%")
        L.append(
            f"- **Initial drop** (first {self._drop_win:.0f} s): "
            f"{self.initial_drop_pct:.1f}% — largely capacitive/diffusional "
            f"relaxation, not catalyst degradation."
        )
        for f_ in self.findings:
            L.append(f"- {f_}")
        for w in self.warnings:
            L.append(f"- ⚠ {w}")
        return "\n".join(L)

    _drop_win: float = 60.0


def analyze_ca(
    time: np.ndarray,
    current: np.ndarray,
    area_cm2: Optional[float] = None,
    per_area: bool = False,
    tail_frac: float = 0.05,
    drop_window_s: float = 60.0,
    retention_times: Tuple[float, ...] = (600.0, 1000.0, 1800.0),
) -> CAResult:
    """
    time, current: 1-D arrays (current may be signed; |I| is used).
    per_area: if True and area_cm2 given, convert to current density.
    tail_frac: fraction of points averaged at each end for I_initial/I_final.
    """
    t = np.asarray(time, dtype=float)
    i_raw = np.asarray(current, dtype=float)

    order = np.argsort(t)
    t, i_raw = t[order], i_raw[order]

    i_mag = np.abs(i_raw)
    unit = "A"
    if per_area and area_cm2 and area_cm2 > 0:
        i_mag = i_mag / area_cm2
        unit = "A/cm²"

    res = CAResult(time=t, current=i_mag, current_raw=i_raw, unit_label=unit)
    res._drop_win = drop_window_s
    n = len(t)
    if n < 10:
        res.warnings.append("Fewer than 10 points — metrics unreliable.")
        return res

    res.duration_s = float(t[-1] - t[0])
    ntail = max(1, int(round(n * tail_frac)))
    res.i_initial = float(np.mean(i_mag[:ntail]))
    res.i_final = float(np.mean(i_mag[-ntail:]))
    res.i_steady = res.i_final

    if res.i_initial > 0:
        res.retention_pct = 100.0 * res.i_final / res.i_initial
        # retention at chosen times
        for tt in retention_times:
            if t[0] <= tt <= t[-1]:
                idx = int(np.argmin(np.abs(t - tt)))
                res.retention_at[float(tt)] = 100.0 * i_mag[idx] / res.i_initial
        # initial drop
        if t[-1] >= drop_window_s:
            idxd = int(np.argmin(np.abs(t - (t[0] + drop_window_s))))
            res.initial_drop_pct = 100.0 * (
                res.i_initial - i_mag[idxd]) / res.i_initial
    else:
        res.warnings.append("Non-positive initial current — retention skipped.")

    # descriptive findings
    if np.isfinite(res.retention_pct):
        if res.retention_pct >= 90:
            res.findings.append(
                "High current retention (≥90%) — excellent operational "
                "stability over the tested window.")
        elif res.retention_pct >= 70:
            res.findings.append(
                "Moderate current retention (70–90%) — typical for many "
                "carbon-based AOR catalysts over this timescale.")
        else:
            res.findings.append(
                "Substantial current decay (<70% retained) — consistent with "
                "intermediate accumulation / surface poisoning commonly seen in "
                "alcohol oxidation; interpret as apparent operational stability, "
                "not a single degradation mechanism.")
    return res
