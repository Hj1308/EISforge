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
    blank_dominant_fraction: float = 0.0   # share of shared points with |I_blank| > 2|I_sample|


@dataclass
class BlankComparison:
    """Result of :meth:`CVData.compare_to_blank` (currents in mA, potentials in V vs ref)."""
    verdict: str = "undefined"         # increase | decrease | no significant difference | undefined
    text: str = ""
    i_max_sample_mA: float = float("nan")   # max current on the shared anodic sweep
    i_max_blank_mA: float = float("nan")
    ratio_max: float = float("nan")         # i_max_sample / i_max_blank (blank max > 0 only)
    max_net_mA: float = float("nan")
    e_at_max_net_V: float = float("nan")
    e_significant_V: float = float("nan")   # first potential of a significant increase
    noise_mA: float = float("nan")


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

    # ── cycles ────────────────────────────────────────────────────────────────
    def select_cycle(self, idx: int = -1) -> "CVData":
        """Return one cycle (ixdat ``select_cycle`` idea); ``idx=-1`` = last complete cycle.

        Cycles are cut at every second potential vertex, starting from the first scan
        direction (the rule of the former app loader, kept unchanged in patch45). The
        result's meta gets ``n_cycles`` and ``cycle_used`` (1-based).
        """
        E = self.potential_V
        v = np.where(np.diff(np.sign(np.diff(E))) != 0)[0] + 1
        if len(v) < 2:
            start, end, n, k = 0, len(E), 1, 0
        else:
            up = np.median(np.diff(E[:min(50, len(E))])) > 0
            b = v[1::2] if up else v[0::2]
            if len(b) == 0:
                b = v[:1]
            starts, ends = [0] + list(b + 1), list(b + 1) + [len(E)]
            n = len(starts) - 1
            k = (max(0, n - 2) if n >= 2 else 0) if idx == -1 else max(0, min(idx, n - 1))
            start, end = starts[k], ends[k]
        return CVData(potential_V=E[start:end], current_mA=self.current_mA[start:end],
                      scan_rate_mV_s=self.scan_rate_mV_s, label=self.label,
                      meta={**self.meta, "n_cycles": n, "cycle_used": k + 1})

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
        # patch38: a blank that is much larger than the sample over most of the window is not
        # a background of the same electrode (real pairs: valid 0.11 / 0.14, invalid 0.91).
        shared = np.isfinite(net)
        if shared.any():
            i_s = self.current_mA[shared]
            i_b = i_s - net[shared]
            res.blank_dominant_fraction = float(np.mean(np.abs(i_b) > 2.0 * np.abs(i_s)))
            if res.blank_dominant_fraction > 0.5:
                res.warnings.append(
                    f"the blank current is more than twice the sample current on "
                    f"{res.blank_dominant_fraction:.0%} of the shared window: the two scans are "
                    "probably not comparable (different electrode, contact or day?)")
        return res

    # ── activity vs blank (metal-free catalysts) ──────────────────────────────
    def compare_to_blank(self, blank: "CVData", res: Optional[BlankSubtraction] = None,
                         k_noise: float = 3.0, rel_min: float = 0.10,
                         n_points: int = 5) -> "BlankComparison":
        """Judge oxidation activity against a blank, without needing a peak (patch48).

        Metal-free catalysts usually show no anodic peak; activity appears as extra
        anodic current over the blank. On the anodic sweep(s) the net current
        (``subtract``) is significant only where it exceeds BOTH ``k_noise`` x the noise
        of the net current and ``rel_min`` x |I_blank| for ``n_points`` consecutive
        points. Noise = std(diff(net)) / sqrt(2) (point-to-point scatter). The relative
        floor guards against run-to-run differences that the noise cannot see; replicate
        scans remain the better test.
        """
        res = res if res is not None else self.subtract(blank)
        net = res.net_mA
        an = np.zeros(len(net), dtype=bool)
        for br in self.branches():
            if br.kind == "anodic":
                an[br.index] = True
        ok = an & np.isfinite(net)
        out = BlankComparison()
        if not ok.any():
            out.verdict = "undefined"
            out.text = "The alcohol and blank anodic sweeps do not overlap: no comparison."
            return out
        idx = np.flatnonzero(ok)
        i_s = self.current_mA[idx]
        d = net[idx]
        i_b = i_s - d
        out.i_max_sample_mA = float(i_s.max())
        out.i_max_blank_mA = float(i_b.max())
        if out.i_max_blank_mA > 0:
            out.ratio_max = out.i_max_sample_mA / out.i_max_blank_mA
        k = int(np.argmax(d))
        out.max_net_mA, out.e_at_max_net_V = float(d[k]), float(self.potential_V[idx[k]])
        out.noise_mA = float(np.std(np.diff(d)) / np.sqrt(2.0)) if len(d) > 2 else 0.0
        thr = np.maximum(k_noise * out.noise_mA, rel_min * np.abs(i_b))

        def _first_run(mask):
            run = 0
            for j, m in enumerate(mask):
                run = run + 1 if m else 0
                if run >= n_points:
                    return j - n_points + 1
            return None

        up, down = _first_run(d > thr), _first_run(d < -thr)
        if up is not None:
            out.verdict = "increase"
            out.e_significant_V = float(self.potential_V[idx[up]])
            out.text = (f"Anodic current above the blank from E = {out.e_significant_V:.3f} V "
                        "(vs ref): oxidation activity over the blank. No distinct peak is "
                        "expected for metal-free catalysts.")
        elif down is not None:
            out.verdict = "decrease"
            out.text = ("The anodic current is BELOW the blank: no oxidation activity over the "
                        "blank (alcohol adsorption blocking sites, or the two scans are not "
                        "comparable).")
        else:
            out.verdict = "no significant difference"
            out.text = (f"No significant difference from the blank (threshold: "
                        f"{k_noise:g}x noise and {rel_min:.0%} of the blank current).")
        return out

    # ── export ────────────────────────────────────────────────────────────────
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({"potential_V": self.potential_V, "current_mA": self.current_mA})
