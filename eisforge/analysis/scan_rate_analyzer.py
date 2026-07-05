"""
EISForge — Scan-Rate Kinetics Analysis
Author: Hoda Jafari | July 2026

Three classical scan-rate diagnostics for a redox/electrooxidation wave:

  1. Overlay of all CVs (raw)
  2. log(Ipa) vs log(nu) — the "b-value" (power-law exponent):
         Ipa ∝ nu^b
     b ≈ 0.5  -> diffusion-controlled
     b ≈ 1.0  -> surface/adsorption-controlled
     0.5<b<1  -> mixed control
  3. Randles–Ševčík:  Ipa vs sqrt(nu)
     Linearity indicates a diffusion contribution. The diffusion
     coefficient D is *optional* and returned only on explicit request,
     because for high-surface-area / mesoporous carbon electrodes with a
     mixed-control wave (b<1) the Randles–Ševčík assumptions (pure
     semi-infinite diffusion, reversibility, well-defined planar area) do
     not strictly hold, and the resulting D is "apparent" at best.

The analyzer never invents an electrode area or concentration; if D is
requested the caller must supply n, A, C explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress


@dataclass
class ScanRateResult:
    rates_mV: np.ndarray                 # scan rates (mV/s), sorted
    ipa: np.ndarray                      # anodic peak current per rate (same order)
    ipa_potential: np.ndarray            # potential at which Ipa was found
    # log-log (b-value)
    b_value: float = float("nan")
    b_intercept: float = float("nan")
    b_r2: float = float("nan")
    # Randles–Ševčík
    rs_slope: float = float("nan")
    rs_intercept: float = float("nan")
    rs_r2: float = float("nan")
    # optional apparent D
    diffusion_coeff: Optional[float] = None
    findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def mechanism_label(self) -> str:
        b = self.b_value
        if not np.isfinite(b):
            return "undetermined"
        if b < 0.60:
            return "predominantly diffusion-controlled"
        if b > 0.90:
            return "predominantly surface/adsorption-controlled"
        return "mixed control (both diffusion and adsorption contribute)"


def _anodic_peak(
    E: np.ndarray, I: np.ndarray,
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """Return (Ipa, E_at_peak). If window given, search only within it."""
    if window is not None:
        lo, hi = min(window), max(window)
        mask = (E >= lo) & (E <= hi)
        if not np.any(mask):
            mask = np.ones_like(E, dtype=bool)
    else:
        mask = np.ones_like(E, dtype=bool)
    Ew, Iw = E[mask], I[mask]
    idx = int(np.argmax(Iw))
    return float(Iw[idx]), float(Ew[idx])


def analyze_scan_rates(
    data: Dict[float, Tuple[np.ndarray, np.ndarray]],
    peak_window: Optional[Tuple[float, float]] = None,
    compute_D: bool = False,
    n_electrons: Optional[int] = None,
    area_cm2: Optional[float] = None,
    conc_mol_cm3: Optional[float] = None,
) -> ScanRateResult:
    """
    data: {scan_rate_mV_per_s: (E_array_V, I_array)}  — I in whatever unit;
          slopes scale with it, mechanism (b-value) is unit-independent.
    peak_window: optional (E_low, E_high) to restrict the anodic-peak search.
    compute_D: if True, also compute an apparent diffusion coefficient from
               the Randles–Ševčík slope (requires n_electrons, area_cm2,
               conc_mol_cm3, and I in AMPERES).
    """
    rates = np.array(sorted(data.keys()), dtype=float)
    ipa = np.empty(len(rates))
    epa = np.empty(len(rates))
    for k, r in enumerate(rates):
        E, I = data[r]
        ipa[k], epa[k] = _anodic_peak(np.asarray(E), np.asarray(I), peak_window)

    res = ScanRateResult(rates_mV=rates, ipa=ipa, ipa_potential=epa)

    nu = rates / 1000.0  # V/s
    # need >=3 points and positive Ipa for meaningful regressions
    good = ipa > 0
    if np.sum(good) < 3:
        res.warnings.append(
            "Fewer than 3 scan rates with positive anodic peak current — "
            "regressions skipped. Check the peak window."
        )
        return res

    nu_g, ipa_g = nu[good], ipa[good]

    # 1. log-log b-value
    lb = linregress(np.log10(nu_g), np.log10(ipa_g))
    res.b_value, res.b_intercept, res.b_r2 = lb.slope, lb.intercept, lb.rvalue ** 2

    # 2. Randles–Ševčík
    lr = linregress(np.sqrt(nu_g), ipa_g)
    res.rs_slope, res.rs_intercept, res.rs_r2 = lr.slope, lr.intercept, lr.rvalue ** 2

    # findings
    res.findings.append(
        f"b-value (log Ipa vs log ν) = {res.b_value:.3f} "
        f"(R² = {res.b_r2:.4f}) → {res.mechanism_label()}."
    )
    res.findings.append(
        f"Randles–Ševčík (Ipa vs √ν): R² = {res.rs_r2:.4f}, "
        f"slope = {res.rs_slope:.4g} — a linear trend indicates a diffusion "
        f"contribution to the overall response."
    )
    if 0.6 <= res.b_value <= 0.9:
        res.warnings.append(
            "Mixed control (0.6 < b < 0.9): the process is NOT purely "
            "diffusion-limited. Randles–Ševčík D is therefore apparent only."
        )

    # 3. optional apparent D
    if compute_D:
        missing = [nm for nm, v in
                   (("n_electrons", n_electrons), ("area_cm2", area_cm2),
                    ("conc_mol_cm3", conc_mol_cm3)) if v in (None, 0)]
        if missing:
            res.warnings.append(
                "Apparent D not computed — missing/zero: " + ", ".join(missing) + "."
            )
        else:
            # Randles–Ševčík (298 K, reversible):
            #   Ipa = 2.69e5 · n^1.5 · A · sqrt(D) · C · sqrt(nu)
            # slope (Ipa vs sqrt(nu)) = 2.69e5 · n^1.5 · A · C · sqrt(D)
            k = 2.69e5 * (n_electrons ** 1.5) * area_cm2 * conc_mol_cm3
            if k > 0 and res.rs_slope > 0:
                D = (res.rs_slope / k) ** 2
                res.diffusion_coeff = float(D)
                res.warnings.append(
                    f"Apparent D = {D:.3e} cm²/s — valid ONLY under pure-diffusion, "
                    f"reversible assumptions with a well-defined planar area. For "
                    f"mesoporous/high-area carbon this value is indicative, not "
                    f"definitive; report as 'apparent' if used."
                )
            else:
                res.warnings.append("Apparent D not computed — non-positive slope.")

    return res
