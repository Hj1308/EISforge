"""
EISForge — CVData: one container for a cyclic / linear-sweep voltammogram.

Why a dedicated class (patch37)
-------------------------------
CV/LSV data used to travel inside ``EISDataset`` (potential stored in ``z_real``, current
in ``z_imag``, a row index as "frequency"). Mature packages keep one data class per
measurement type with explicit units and the operations that belong to it:
pyimpspec ``DataSet`` (``subtract_impedances``, ``average``), ixdat ``CyclicVoltammogram``
(``select_sweep``, ``diff_with``, ``calc_capacitance``). ``CVData`` follows that pattern.

Conventions
-----------
* ``potential_V`` — potential vs the reference electrode, in V (for Ivium files recorded
  with "Apply wrt OCP=true" the parser already returns the absolute column, patch34).
* ``current_mA`` — current in mA, IUPAC sign convention (anodic positive).

Sweep splitting follows ixdat's ``find_signed_sections`` idea (sign of dE between
consecutive points). Blank subtraction follows ixdat's ``CyclicVoltammogram.diff_with``
(branch by branch, blank interpolated onto the sample's potentials) with one deliberate
difference: points outside the blank's potential range become NaN instead of being
silently clamped to the blank's end-point current by ``np.interp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np


class Branch(NamedTuple):
    """One monotonic sweep: kind is 'anodic' (dE > 0) or 'cathodic' (dE < 0)."""
    kind: str
    index: np.ndarray


@dataclass
class BlankSubtraction:
    """Result of :meth:`CVData.subtract`."""
    net_mA: np.ndarray                 # same length as the sample; NaN where undefined
    overlap_fraction: float            # shared potential span / sample span
    pairs: List[tuple] = field(default_factory=list)      # (kind, n_finite, n_points)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CVData:
    """A voltammogram with explicit units (potential in V vs reference, current in mA)."""
    potential_V: np.ndarray
    current_mA: np.ndarray
    scan_rate_mV_s: float = float("nan")
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.potential_V = np.asarray(self.potential_V, dtype=float)
        self.current_mA = np.asarray(self.current_mA, dtype=float)
        if self.potential_V.shape != self.current_mA.shape or self.potential_V.ndim != 1:
            raise ValueError("potential_V and current_mA must be 1-D arrays of equal length, "
                             f"got {self.potential_V.shape} and {self.current_mA.shape}")
        if len(self.potential_V) < 2:
            raise ValueError("a voltammogram needs at least 2 points")

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_ivium(cls, path, label: Optional[str] = None) -> "CVData":
        """Read an Ivium .idf CV/LSV through the package parser (single source of truth)."""
        from eisforge.parsers.ivium_parser import IviumIDFParser

        ds = IviumIDFParser().parse(path)
        md = dict(ds.metadata)
        if md.get("data_type") not in ("CV", "LSV"):
            raise ValueError(f"{Path(path).name} is not a CV/LSV file (data_type="
                             f"{md.get('data_type')!r})")
        try:
            sr = float(md.get("scanrate", "nan")) * 1e3            # V/s in the file -> mV/s
        except (TypeError, ValueError):
            sr = float("nan")
        return cls(potential_V=ds.z_real, current_mA=ds.z_imag, scan_rate_mV_s=sr,
                   label=label or Path(path).stem, meta=md)

    # ── sweeps ────────────────────────────────────────────────────────────────
    def branches(self, min_points: int = 3) -> List[Branch]:
        """Split into monotonic sweeps by the sign of dE (ixdat find_signed_sections idea).

        Zero potential steps inherit the previous direction; the vertex point belongs to
        both adjacent branches. Runs shorter than ``min_points`` are ignored.
        """
        d = np.sign(np.diff(self.potential_V))
        nonzero = np.flatnonzero(d)
        if len(nonzero) == 0:
            return []
        d[: nonzero[0]] = d[nonzero[0]]
        for k in range(1, len(d)):
            if d[k] == 0:
                d[k] = d[k - 1]
        out, start = [], 0
        for k in range(1, len(d) + 1):
            if k == len(d) or d[k] != d[start]:
                idx = np.arange(start, k + 1)
                if len(idx) >= min_points:
                    out.append(Branch("anodic" if d[start] > 0 else "cathodic", idx))
                start = k
        return out

    # ── blank subtraction ────────────────────────────────────────────────────
    def subtract(self, blank: "CVData") -> BlankSubtraction:
        """Subtract ``blank`` branch by branch (ixdat diff_with), NaN outside its range.

        Branches are paired in order of appearance within each kind (1st anodic with
        1st anodic, ...). Warnings flag a scan-rate mismatch, less than 80% shared
        potential window, an OCP difference above 50 mV, and unpaired branches.
        """
        net = np.full(len(self.current_mA), np.nan)
        res = BlankSubtraction(net_mA=net, overlap_fraction=0.0)
        mine, theirs = self.branches(), blank.branches()
        pool = {"anodic": [b for b in theirs if b.kind == "anodic"],
                "cathodic": [b for b in theirs if b.kind == "cathodic"]}
        for br in mine:
            if not pool[br.kind]:
                res.warnings.append(f"the blank has no {br.kind} branch left to pair with")
                continue
            bb = pool[br.kind].pop(0)
            eb, ib = blank.potential_V[bb.index], blank.current_mA[bb.index]
            order = np.argsort(eb, kind="stable")
            eb, ib = eb[order], ib[order]
            e = self.potential_V[br.index]
            inside = (e >= eb[0]) & (e <= eb[-1])
            net[br.index[inside]] = self.current_mA[br.index][inside] - np.interp(e[inside], eb, ib)
            res.pairs.append((br.kind, int(inside.sum()), len(br.index)))
        left = sum(len(v) for v in pool.values())
        if left:
            res.warnings.append(f"{left} blank branch(es) were not used")

        span = float(np.ptp(self.potential_V))
        lo = max(float(self.potential_V.min()), float(blank.potential_V.min()))
        hi = min(float(self.potential_V.max()), float(blank.potential_V.max()))
        res.overlap_fraction = max(0.0, hi - lo) / span if span > 0 else 0.0
        if res.overlap_fraction < 0.8:
            res.warnings.append(f"only {res.overlap_fraction:.0%} of the potential window is "
                                "shared with the blank")
        s1, s2 = self.scan_rate_mV_s, blank.scan_rate_mV_s
        if np.isfinite(s1) and np.isfinite(s2) and abs(s1 - s2) > 0.01 * max(abs(s2), 1e-12):
            res.warnings.append(f"different scan rates ({s1:g} vs {s2:g} mV/s): the capacitive "
                                "current does not cancel")
        o1, o2 = self.meta.get("ocp_V"), blank.meta.get("ocp_V")
        if o1 is not None and o2 is not None and abs(o1 - o2) > 0.05:
            res.warnings.append(f"OCP differs by {abs(o1 - o2) * 1e3:.0f} mV: the scans were applied "
                                "relative to OCP, so their absolute windows differ")
        return res

    # ── export ────────────────────────────────────────────────────────────────
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({"potential_V": self.potential_V, "current_mA": self.current_mA})
