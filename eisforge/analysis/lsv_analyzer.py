"""
LSV Analyzer — Automatic Linear Sweep Voltammetry Analysis for AOR.
Author: Hoda Jafari | May 2026

Supports all catalyst families:
    - noble_metal  : Pt, Pd, Au, Rh
    - alloy        : PtRu, PtSn, PdAu, PtCu
    - metal_oxide  : NiO, Co3O4, Co2NiO4, MnO2
    - carbon_material   : N-doped Carbon, CNT, rGO, graphene-based

Electrolyte specifics:
    - Acid type    : H2SO4, HClO4, HCl, HNO3
    - Base type    : KOH, NaOH, Na2CO3, NH3
    - Concentration: float in mol/L (M)

iR Compensation:
    E_corrected = E_measured - I(A) × R_s(Ω)
    Tafel slopes and E_onset are computed on iR-corrected potentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from eisforge.analysis.cv_analyzer import (
    ElectrolyteInfo,
    CATALYST_NOBLE_METAL,
    CATALYST_ALLOY,
    CATALYST_METAL_OXIDE,
    CATALYST_METAL_FREE,
    ACID_H2SO4, ACID_HClO4, ACID_HCl, ACID_HNO3,
    BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3,
)

logger = logging.getLogger(__name__)

FARADAY   = 96485.33    # C/mol
GAS_CONST = 8.31446     # J/mol/K


@dataclass
class LSVAnalysisResult:
    """Results of complete LSV analysis."""

    # Catalyst and electrolyte info
    catalyst_type : str = CATALYST_NOBLE_METAL
    electrolyte   : ElectrolyteInfo = field(default_factory=ElectrolyteInfo)

    # E_onset
    e_onset        : float = 0.0
    e_onset_method : str   = ""

    # Tafel
    tafel_slope           : float = 0.0   # mV/dec
    tafel_slope_std       : float = 0.0
    exchange_current_density : float = 0.0  # j0 (mA/cm²) — valid only if j0_is_valid
    j0_is_valid           : bool  = False   # True only when an equilibrium potential was supplied
    tafel_r_squared       : float = 0.0
    tafel_region          : tuple = (0.0, 0.0)   # (E_low, E_high) of the fitted window, V
    tafel_eta_region      : tuple = (float("nan"), float("nan"))  # overpotential span, V (needs E_eq)
    tafel_n_points        : int   = 0       # points used in the fit
    tafel_decades         : float = 0.0     # decades of current spanned by the fit
    tafel_method          : str   = ""      # "auto-detected" | "user current window" | "failed"
    equilibrium_potential : Optional[float] = None  # E_eq used for j0 (same frame as e_onset)
    tafel_warnings        : list  = field(default_factory=list)  # diagnostic flags

    # Overpotential
    overpotential_10  : float = 0.0   # V
    overpotential_50  : float = 0.0
    overpotential_100 : float = 0.0

    # Activity
    j_at_onset      : float = 0.0
    j_limiting      : float = 0.0
    e_half_wave     : float = 0.0
    mass_activity   : float = 0.0   # mA/mg
    specific_activity: float = 0.0  # mA/cm²_ECSA or mA/cm²_BET

    # iR compensation
    ir_compensated : bool  = False
    r_s_used       : float = 0.0

    # Measurement parameters
    scan_rate        : float = 5.0
    electrode_area   : float = 1.0
    ecsa             : float = 0.0
    catalyst_loading : float = 0.0

    mechanism_interpretation : str = ""
    performance_rating       : str = ""

    def summary(self) -> str:
        el            = self.electrolyte
        is_carbon_material = self.catalyst_type == CATALYST_METAL_FREE
        is_oxide      = self.catalyst_type == CATALYST_METAL_OXIDE
        ecsa_label    = "cm²_BET" if is_carbon_material else "cm²_metal"

        lines = [
            "=" * 68,
            "  LSV Analysis Results — EISForge",
            "=" * 68,
            f"  Catalyst type       : {self._catalyst_label()}",
            f"  Electrolyte         : {el.label()}  ({el.media})",
            "-" * 68,
        ]
        if self.ir_compensated:
            lines.append(f"  iR Compensation     : APPLIED (R_s = {self.r_s_used:.3f} Ω)")
        else:
            lines.append(f"  iR Compensation     : not applied")

        lines += [
            f"  E_onset             = {self.e_onset:.4f} V  ({self.e_onset_method})",
            "-" * 68,
            f"  Tafel slope         = {self.tafel_slope:.1f} ± {self.tafel_slope_std:.1f} mV/dec",
            f"  Tafel fit window    = {self.tafel_region[0]:.3f}–{self.tafel_region[1]:.3f} V"
            f"  |  {self.tafel_decades:.2f} dec  |  n = {self.tafel_n_points}  ({self.tafel_method})",
            f"  R² (Tafel fit)      = {self.tafel_r_squared:.4f}",
        ]
        if self.j0_is_valid:
            lines.append(
                f"  j0                  = {self.exchange_current_density:.4e} mA/cm²"
                f"  (extrapolated to η = 0)"
            )
        else:
            lines.append(
                "  j0                  = n/a  (supply an equilibrium potential to compute a true j0)"
            )

        if is_carbon_material:
            lines.append(
                "  NOTE: Tafel > 120 mV/dec can be normal for metal-free catalysts,"
            )
            lines.append(
                "        but verify the fit lies on the activation-controlled branch."
            )

        for w in self.tafel_warnings:
            lines.append(f"  ! {w}")

        lines += [
            "-" * 68,
            f"  η @ 10 mA/cm²       = {self.overpotential_10*1000:.1f} mV"
            if not np.isnan(self.overpotential_10) else "  η @ 10 mA/cm²       = not reached",
            f"  η @ 50 mA/cm²       = {self.overpotential_50*1000:.1f} mV"
            if not np.isnan(self.overpotential_50) else "  η @ 50 mA/cm²       = not reached",
            f"  η @ 100 mA/cm²      = {self.overpotential_100*1000:.1f} mV"
            if not np.isnan(self.overpotential_100) else "  η @ 100 mA/cm²      = not reached",
            "-" * 68,
            f"  Limiting current    = {self.j_limiting:.3f} mA/cm²",
            f"  Half-wave potential = {self.e_half_wave:.4f} V"
            if not np.isnan(self.e_half_wave) else "  Half-wave potential = not determined",
        ]

        if self.catalyst_loading > 0:
            lines.append(f"  Mass activity       = {self.mass_activity:.3f} mA/mg_cat")
        if self.ecsa > 0:
            lines.append(f"  Specific activity   = {self.specific_activity:.4f} mA/{ecsa_label}")

        lines += [
            "-" * 68,
            f"  Mechanism : {self.mechanism_interpretation}",
            f"  Rating    : {self.performance_rating}",
            "=" * 68,
        ]
        return "\n".join(lines)

    def _catalyst_label(self) -> str:
        labels = {
            CATALYST_NOBLE_METAL : "Noble Metal (Pt / Pd / Au / Rh)",
            CATALYST_ALLOY       : "Alloy (PtRu / PtSn / PdAu / PtCu ...)",
            CATALYST_METAL_OXIDE : "Metal Oxide (NiO / Co3O4 / MnO2 ...)",
            CATALYST_METAL_FREE  : "Metal-Free (carbon_material / N-doped C / CNT ...)",
        }
        return labels.get(self.catalyst_type, self.catalyst_type)


class LSVAnalyzer:
    """
    Automatic LSV analyzer for AOR — supports all catalyst families.

    Parameters
    ----------
    scan_rate : float
        Scan rate in mV/s (LSV typically 1–10 mV/s; default: 5).
    electrode_area : float
        Geometric electrode area in cm² (default: 1.0).
    ecsa : float
        ECSA in cm²_metal (noble/alloy) or cm²_BET (metal-free).
    catalyst_loading : float
        Catalyst loading in mg/cm².
    electrolyte : str or ElectrolyteInfo
        Electrolyte. Can be:
          - ElectrolyteInfo object (recommended)
          - simple string: 'acidic', 'alkaline', 'KOH', 'H2SO4'
    electrolyte_concentration : float
        Concentration in mol/L — used when electrolyte is a string.
    catalyst_type : str
        One of: 'noble_metal', 'alloy', 'metal_oxide', 'carbon_material'
    e_ref_vs_rhe : float
        Reference electrode offset to RHE (V).
    tafel_current_range : tuple
        (j_min, j_max) mA/cm² for Tafel fit.
    current_unit : str
        Input current unit: 'mA', 'A', 'uA', 'nA'.
    """

    _UNIT_TO_MA = {"A": 1000.0, "mA": 1.0, "uA": 1e-3, "μA": 1e-3, "nA": 1e-6}

    # Tafel current ranges per catalyst type (mA/cm²)
    _TAFEL_RANGES = {
        CATALYST_NOBLE_METAL : (0.1,  2.0),
        CATALYST_ALLOY       : (0.1,  2.0),
        CATALYST_METAL_OXIDE : (0.05, 1.0),   # lower currents, higher onset
        CATALYST_METAL_FREE  : (0.02, 1.0),   # gentle slope, lower currents
    }

    def __init__(
        self,
        scan_rate                : float = 5.0,
        electrode_area           : float = 1.0,
        ecsa                     : float = 0.0,
        catalyst_loading         : float = 0.0,
        electrolyte                      = "acidic",
        electrolyte_concentration: float = 0.5,
        catalyst_type            : str   = CATALYST_NOBLE_METAL,
        e_ref_vs_rhe             : float = 0.0,
        tafel_current_range      : tuple = None,
        current_unit             : str   = "mA",
        onset_method             : str   = "tangent",
        equilibrium_potential    : Optional[float] = None,
        auto_tafel_region        : bool  = True,
        min_tafel_decades        : float = 0.8,
        tafel_r2_target          : float = 0.99,
    ) -> None:
        self.scan_rate        = scan_rate
        self.electrode_area   = max(electrode_area, 1e-10)
        self.ecsa             = ecsa
        self.catalyst_loading = catalyst_loading
        self.catalyst_type    = catalyst_type
        self.e_ref_vs_rhe     = e_ref_vs_rhe
        self.current_unit     = current_unit
        self._unit_factor     = self._UNIT_TO_MA.get(current_unit, 1.0)
        self.onset_method     = (str(onset_method) or "tangent").lower()

        # Tafel configuration
        # equilibrium_potential is interpreted in the SAME frame as the analysed
        # potential (i.e. after e_ref_vs_rhe is applied). It is required to report
        # a physically meaningful exchange current density (j0).
        self.equilibrium_potential = equilibrium_potential
        self.auto_tafel_region     = auto_tafel_region
        self.min_tafel_decades     = float(min_tafel_decades)
        self.tafel_r2_target       = float(tafel_r2_target)
        # (E_lo, E_hi) in the analysis (RHE) frame. When set, the Tafel
        # line is fitted directly on this window (overrides auto-detection).
        self.tafel_potential_range = None

        # Electrolyte
        if isinstance(electrolyte, ElectrolyteInfo):
            self.electrolyte_info = electrolyte
        else:
            self.electrolyte_info = ElectrolyteInfo.from_string(
                str(electrolyte), concentration=electrolyte_concentration
            )

        # Tafel range: use provided or default per catalyst type
        if tafel_current_range is not None:
            self.tafel_current_range = tafel_current_range
        else:
            self.tafel_current_range = self._TAFEL_RANGES.get(
                catalyst_type, (0.1, 2.0)
            )

    # ── iR Compensation ───────────────────────────────────────────────────────

    @staticmethod
    def apply_ir_compensation(
        potential  : np.ndarray,
        current_ma : np.ndarray,
        r_s_ohms   : float,
    ) -> np.ndarray:
        """E_corrected = E_measured - I(A) × R_s(Ω)"""
        if r_s_ohms <= 0:
            return potential.copy()
        current_a = current_ma * 1e-3
        return potential - current_a * r_s_ohms

    # ── Main analyze ──────────────────────────────────────────────────────────

    def analyze(
        self,
        potential : np.ndarray,
        current   : np.ndarray,
        r_s_ohms  : float = 0.0,
    ) -> LSVAnalysisResult:
        """
        Full LSV analysis — adapts to catalyst_type and electrolyte.

        Parameters
        ----------
        potential : np.ndarray   Potential (V) — forward scan only.
        current   : np.ndarray   Current (in current_unit, default mA).
        r_s_ohms  : float        R_s for iR compensation (Ω). 0 = skip.
        """
        potential = np.asarray(potential, dtype=float)
        current   = np.asarray(current,   dtype=float)

        if len(potential) < 20:
            raise ValueError(f"Insufficient data: {len(potential)} points.")

        current_ma = current * self._unit_factor
        potential  = potential + self.e_ref_vs_rhe

        ir_compensated = r_s_ohms > 0
        if ir_compensated:
            potential = self.apply_ir_compensation(potential, current_ma, r_s_ohms)
            logger.info(f"iR compensation applied: R_s = {r_s_ohms:.3f} Ω")

        current_ma = self._smooth(current_ma)

        if potential[-1] < potential[0]:
            potential  = potential[::-1]
            current_ma = current_ma[::-1]

        j = current_ma / self.electrode_area    # mA/cm²

        # find upper boundary of AOR wave (before OER or second wave)
        e_aor_limit = self._find_aor_upper_limit(potential, j)

        e_onset, onset_method = self._detect_onset(
            potential[potential <= e_aor_limit],
            j[potential <= e_aor_limit],
        )
        tafel = self._tafel_analysis(
            potential[potential <= e_aor_limit],
            j[potential <= e_aor_limit],
            e_onset,
        )
        eta_10  = self._overpotential_at_j(potential, j, 10.0,  e_onset)
        eta_50  = self._overpotential_at_j(potential, j, 50.0,  e_onset)
        eta_100 = self._overpotential_at_j(potential, j, 100.0, e_onset)
        j_lim   = float(np.max(j))
        e_half  = self._half_wave_potential(potential, j, j_lim)
        j_at_onset = float(np.interp(e_onset, potential, j))

        mass_act = (
            j_at_onset / self.catalyst_loading
            if self.catalyst_loading > 0 else 0.0
        )
        spec_act = (
            j_at_onset * self.electrode_area / self.ecsa
            if self.ecsa > 0 else 0.0
        )

        _aor_note = (
            f"AOR wave clipped at E={e_aor_limit:.3f} V (valley/plateau detected)"
            if e_aor_limit < potential[-1] - 0.05
            else "No second wave detected — full curve used"
        )

        return LSVAnalysisResult(
            catalyst_type            = self.catalyst_type,
            electrolyte              = self.electrolyte_info,
            e_onset                  = e_onset,
            e_onset_method           = onset_method + (" (iR-corrected)" if ir_compensated else ""),
            tafel_slope              = tafel["slope"],
            tafel_slope_std          = tafel["slope_std"],
            exchange_current_density = tafel["j0"],
            j0_is_valid              = tafel["j0_valid"],
            tafel_r_squared          = tafel["r2"],
            tafel_region             = tafel["region"],
            tafel_eta_region         = tafel["eta_region"],
            tafel_n_points           = tafel["n_points"],
            tafel_decades            = tafel["decades"],
            tafel_method             = tafel["method"],
            equilibrium_potential    = self.equilibrium_potential,
            tafel_warnings           = tafel["warnings"] + [_aor_note],
            overpotential_10         = eta_10,
            overpotential_50         = eta_50,
            overpotential_100        = eta_100,
            j_at_onset               = j_at_onset,
            j_limiting               = j_lim,
            e_half_wave              = e_half,
            mass_activity            = mass_act,
            specific_activity        = spec_act,
            ir_compensated           = ir_compensated,
            r_s_used                 = r_s_ohms,
            scan_rate                = self.scan_rate,
            electrode_area           = self.electrode_area,
            ecsa                     = self.ecsa,
            catalyst_loading         = self.catalyst_loading,
            mechanism_interpretation = self._interpret_tafel(tafel["slope"]),
            performance_rating       = self._rate_performance(e_onset, tafel["slope"], eta_10),
        )

    # ── Wave segmentation ────────────────────────────────────────────────────

    @staticmethod
    def _find_aor_upper_limit(potential: np.ndarray, j: np.ndarray) -> float:
        # Find upper potential boundary of AOR wave.
        # Locates valley (min dj/dE) separating AOR from OER/second wave.
        # Returns E_upper (V). If no valley found, returns last potential.
        E  = np.asarray(potential, dtype=float)
        jj = np.asarray(j, dtype=float)
        n  = len(E)

        # orient so activation branch is positive-going
        seg = max(int(0.15 * n), 3)
        if np.mean(np.abs(jj[-seg:])) < np.mean(np.abs(jj[:seg])):
            jj = -jj

        dj = np.gradient(jj, E)

        # search for valley only after the first 25% of scan
        # (avoid the initial noisy region)
        search_start = max(int(0.25 * n), 5)

        # valley = first index where dj has a local minimum AND is below
        # 20% of its maximum value in the search region
        dj_search = dj[search_start:]
        E_search  = E[search_start:]

        dj_max = float(np.max(dj_search))
        # threshold: dj must drop to < 20% of peak slope
        threshold = 0.20 * dj_max

        below = np.where(dj_search < threshold)[0]
        if len(below) == 0:
            return float(E[-1])   # no valley → use full curve

        # among points below threshold, find the one with minimum dj
        # (the flattest point = centre of plateau)
        valley_idx = below[int(np.argmin(dj_search[below]))]
        e_valley   = float(E_search[valley_idx])

        # sanity: valley must be at least 0.1V after scan start
        if e_valley < E[0] + 0.10:
            return float(E[-1])

        # sanity: a genuine AOR wave-END valley follows a current PEAK
        # (current rose, peaked, then fell into the valley). If the current
        # only rises monotonically up to the valley, the 'valley' is the
        # pre-wave background gap (double-layer plateau before the real
        # oxidation wave), not a wave end -- do not clip there.
        global_valley = search_start + valley_idx
        peak_before   = float(np.max(jj[:global_valley + 1]))
        j_valley_val  = float(jj[global_valley])
        if peak_before <= 1.10 * j_valley_val:
            return float(E[-1])

        return e_valley

    # ── E_onset ───────────────────────────────────────────────────────────────

    def _detect_onset(self, potential, j) -> tuple[float, str]:
        """
        E_onset -- hybrid (zero-crossing anchor + sign-robust).
        Combines our zero-crossing tangent with fallbacks.
        """
        n  = int(len(potential))
        E  = np.asarray(potential, dtype=float)
        jj = np.asarray(j,         dtype=float)

        # --- sign detection using |j| magnitude ------------------------------
        seg = max(int(0.15 * n), 3)
        j_abs_lo = float(np.mean(np.abs(jj[:seg])))   # low-E end
        j_abs_hi = float(np.mean(np.abs(jj[-seg:])))  # high-E end
        # anodic AOR: magnitude should be LARGER at high-E (oxidation current grows)
        # if low-E end is larger → current stored negative → flip
        jw = jj.copy() if j_abs_hi >= j_abs_lo else -jj

        # shift baseline to ~0
        bl_end  = max(int(0.15 * n), 5)
        jw      = jw - float(np.min(jw[:bl_end]))

        baseline = float(np.median(jw[:bl_end]))
        std_base = float(np.std(jw[:bl_end])) or 1e-12
        j_span   = float(np.max(jw) - baseline) or 1e-12
        method   = getattr(self, "onset_method", "tangent").lower()

        def _tangent():
            if bl_end >= n - 4:
                return None
            dj = np.gradient(jw, E)
            k  = int(np.argmax(dj[bl_end:])) + bl_end
            w  = max(int(0.05 * n), 3)
            r0 = max(k - w, bl_end); r1 = min(k + w + 1, n)
            if r1 - r0 < 3:
                return None
            try:
                m_r, b_r = np.polyfit(E[r0:r1], jw[r0:r1], 1)
                m_b, b_b = np.polyfit(E[:bl_end], jw[:bl_end], 1)
            except Exception:
                return None
            denom = m_r - m_b
            if abs(denom) < 1e-12:
                return None
            return float(np.clip((b_b - b_r) / denom, E.min(), E.max()))

        def _threshold():
            thr   = baseline + 5.0 * std_base
            above = np.where(jw > thr)[0]
            above = above[above >= bl_end]
            if len(above) == 0:
                return None
            return float(E[int(above[0])])

        def _derivative():
            if n <= bl_end + 2:
                return None
            dj     = np.gradient(jw, E)
            dj_max = float(np.max(dj[bl_end:]))
            if dj_max <= 0:
                return None
            knee = np.where(dj[bl_end:] >= 0.10 * dj_max)[0]
            if len(knee) == 0:
                return None
            return float(E[bl_end + int(knee[0])])

        order = {
            "tangent":    [_tangent,    _threshold,  _derivative],
            "threshold":  [_threshold,  _tangent,    _derivative],
            "derivative": [_derivative, _tangent,    _threshold],
        }.get(method, [_tangent, _threshold, _derivative])

        for fn in order:
            try:
                val = fn()
            except Exception:
                val = None
            if val is not None and np.isfinite(val):
                tag = fn.__name__.strip("_")
                label = tag if tag == method else f"{method}->{tag} (fallback)"
                return float(val), label

        # last resort: 5% of current span
        cross = np.where(jw - baseline >= 0.05 * j_span)[0]
        cross = cross[cross >= bl_end]
        idx   = int(cross[0]) if len(cross) else n // 2
        return float(E[idx]), "fallback-5%"

    # ── Tafel analysis ────────────────────────────────────────────────────────

    def _tafel_analysis(self, potential, j, e_onset) -> dict:
        warnings: list = []

        def _failed(msg):
            return {
                "slope": float("nan"), "slope_std": float("nan"),
                "j0": float("nan"), "j0_valid": False, "r2": 0.0,
                "region": (e_onset, e_onset),
                "eta_region": (float("nan"), float("nan")),
                "n_points": 0, "decades": 0.0, "method": "failed",
                "warnings": warnings + [msg],
            }

        j_abs = np.abs(j)
        j_peak_idx = int(np.argmax(j_abs))
        e_peak = float(potential[j_peak_idx])
        j_floor = max(1e-8, 0.005 * float(np.max(j_abs)))

        # ── Manual potential window (highest priority) ─────────────────────
        # Robust for curves with no current peak (monotonic exponential rise)
        # where auto-detection of the activation branch is unreliable.
        if getattr(self, "tafel_potential_range", None) is not None:
            e_lo, e_hi = sorted(self.tafel_potential_range)
            mwin = ((potential >= e_lo) & (potential <= e_hi)
                    & (j > 0) & (j_abs > j_floor))
            if int(np.sum(mwin)) >= 4:
                E_m    = potential[mwin]
                logj_m = np.log10(j_abs[mwin])
                slope, intercept, r_val, _, slope_std = linregress(logj_m, E_m)
                if slope < 0:
                    warnings.append("Negative Tafel slope -- using |slope|.")
                    slope = abs(slope)
                tafel_mv = slope * 1000.0
                r2  = r_val ** 2
                dec = float(logj_m.max() - logj_m.min())
                j0, j0_valid = float("nan"), False
                eta_region = (float("nan"), float("nan"))
                if self.equilibrium_potential is not None:
                    e_eq = float(self.equilibrium_potential)
                    try:
                        j0 = float(10 ** (-(intercept - e_eq) / slope))
                        j0_valid = bool(np.isfinite(j0) and j0 > 0)
                    except Exception:
                        pass
                    eta_region = (float(E_m.min() - e_eq), float(E_m.max() - e_eq))
                else:
                    warnings.append("j0 not computed: no equilibrium potential supplied.")
                if r2 < 0.99:
                    warnings.append(f"R2 = {r2:.4f} (<0.99)")
                if dec < 1.0:
                    warnings.append(f"Only {dec:.2f} decades spanned (<1).")
                return {
                    "slope": tafel_mv, "slope_std": slope_std * 1000.0,
                    "j0": j0, "j0_valid": j0_valid, "r2": r2,
                    "region": (float(E_m.min()), float(E_m.max())),
                    "eta_region": eta_region,
                    "n_points": int(np.sum(mwin)), "decades": dec,
                    "method": "user potential window", "warnings": warnings,
                }
            warnings.append("Manual potential window has <4 valid points -> auto.")

        # Hybrid Tafel domain (ChatGPT OER detection + Grok noise floor + our valley)
        _dj_full = np.gradient(j, potential)
        _zc_idx  = int(np.argmin(np.abs(j)))
        _after   = _dj_full[_zc_idx:]; _E_after = potential[_zc_idx:]
        _dj_thr  = float(np.quantile(np.abs(_dj_full), 0.80))
        _E_oer   = None
        for _k in range(len(_after)-2):
            if _after[_k] >= _dj_thr and _after[_k+1] >= _dj_thr and _after[_k+2] >= _dj_thr:
                _E_oer = float(_E_after[_k]); break
        if _E_oer is None:
            _E_oer = float(np.quantile(potential, 0.85))
        _n   = len(potential)
        _ss  = max(int(0.25*_n), 5)
        _djs = _dj_full[_ss:]; _Es = potential[_ss:]
        _djmax = float(np.max(_djs))
        _below = np.where(_djs < 0.15*_djmax)[0]
        _e_valley = float(_Es[_below[int(np.argmin(_djs[_below]))]]) \
                    if len(_below) else _E_oer
        E_upper = min(_E_oer - 0.03, _e_valley)
        if E_upper <= e_onset:
            E_upper = _e_valley
        j_max     = float(j_abs.max())
        j_upper_v = abs(float(np.interp(E_upper, potential, j)))
        j_act_lim = max(0.40 * j_upper_v, 0.01 * j_max)
        j_noise   = 0.01 * j_max
        if e_onset <= E_upper:
            domain = (
                (potential >= e_onset) &
                (potential <= E_upper) &
                (j > 0) &
                (j_abs > j_noise) &
                (j_abs <= j_act_lim)
            )
        else:
            domain = (potential <= E_upper) & (j > 0) & (j_abs > j_noise)
            warnings.append("E_onset > E_upper; domain set to [start, E_upper].")
        if np.sum(domain) < 6:
            domain = (potential <= E_upper) & (j > 0) & (j_abs > j_noise)
            warnings.append("Activation domain relaxed.")
        if np.sum(domain) < 4:
            return _failed("Insufficient points in activation region.")
        E_dom     = potential[domain]
        j_dom     = j[domain]
        j_abs_dom = j_abs[domain]
        _diffs = np.diff(j_dom)
        _nm    = np.where(_diffs <= 0)[0]
        if len(_nm) and _nm[0] + 1 >= 4:
            E_dom     = E_dom[:_nm[0]+1]
            j_abs_dom = j_abs_dom[:_nm[0]+1]
            warnings.append("Tafel window trimmed at first non-monotonic point.")
        if len(E_dom) < 4:
            return _failed("Insufficient monotonic points in activation region.")
        logj_dom = np.log10(j_abs_dom)

        region_idx = None
        method = ""

        if (self.tafel_current_range is not None) and (not self.auto_tafel_region):
            j_min, j_max = sorted([abs(x) for x in self.tafel_current_range])
            win = (j_abs[domain] >= j_min) & (j_abs[domain] <= j_max)
            if np.sum(win) >= self._min_tafel_points(len(E_dom)):
                idx = np.where(win)[0]
                xs, ys = logj_dom[idx], E_dom[idx]
                _, _, rv, _, _ = linregress(xs, ys)
                dec = float(xs.max() - xs.min())
                if (rv ** 2 >= 0.97) and (dec >= 0.5):
                    region_idx = idx
                    method = "user current window (absolute)"
            if region_idx is None:
                warnings.append("Manual window not linear enough → fallback to auto")

        if region_idx is None:
            region_idx = self._find_linear_tafel_region(E_dom, logj_dom)
            method = "auto-detected"

        if region_idx is None or len(region_idx) < 3:
            return _failed("Could not find linear Tafel region.")

        E_fit = E_dom[region_idx]
        logj_fit = logj_dom[region_idx]

        slope, intercept, r_val, _, slope_std = linregress(logj_fit, E_fit)
        if slope < 0:
            warnings.append("Negative Tafel slope -- using |slope|.")
            slope = abs(slope)
        tafel_mv = slope * 1000.0
        r2 = r_val ** 2
        decades = float(logj_fit.max() - logj_fit.min())
        n_pts = int(len(region_idx))

        j0, j0_valid = float("nan"), False
        eta_region = (float("nan"), float("nan"))
        if self.equilibrium_potential is not None:
            e_eq = float(self.equilibrium_potential)
            try:
                j0 = float(10 ** (-(intercept - e_eq) / slope))
                j0_valid = bool(np.isfinite(j0) and j0 > 0)
            except Exception:
                pass
            eta_region = (float(E_fit.min() - e_eq), float(E_fit.max() - e_eq))
        else:
            warnings.append("j0 not computed: no equilibrium potential supplied.")

        if r2 < 0.99:
            warnings.append(f"R2 = {r2:.4f} (<0.99)")
        if decades < 1.0:
            warnings.append(f"Only {decades:.2f} decades spanned (<1).")
        if abs(tafel_mv) > 120.0:
            warnings.append(f"Slope = {tafel_mv:.0f} mV/dec (>120)")
        if E_fit.max() > e_peak - 0.02:
            warnings.append("Upper edge near current peak → possible mass-transport.")

        return {
            "slope": tafel_mv, "slope_std": slope_std * 1000.0,
            "j0": j0, "j0_valid": j0_valid, "r2": r2,
            "region": (float(E_fit.min()), float(E_fit.max())),
            "eta_region": eta_region,
            "n_points": n_pts, "decades": decades,
            "method": method, "warnings": warnings,
        }

    def _min_tafel_points(self, n_domain: int) -> int:
        """Minimum points for a trustworthy linear fit."""
        return max(6, int(0.08 * n_domain))

    def _find_linear_tafel_region(self, E, logj):
        """
        Find the contiguous window of (E vs log10 j) that represents the
        activation-controlled Tafel line.

        Physical basis: on an AOR polarisation curve the apparent slope can only
        *increase* as mass-transport mixes in at higher overpotential, so the true
        activation slope is the SMALLEST linear slope found at the foot of the
        wave. The routine therefore scans candidate windows of width between
        min_tafel_decades and ~1.2 decades, keeps those that are genuinely linear
        (R^2 >= gate), and returns the one with the smallest |slope| (tie-broken by
        more points, then lower foot). This rejects windows that creep into the
        mixed/mass-transport region. If no window clears the linearity gate, the
        highest-R^2 window meeting the span requirement is returned.

        Regression statistics are computed in O(1) per window via prefix sums.
        Assumes E is ordered by ascending potential and log10(j) is near-monotonic
        on this branch. Returns an int index array into E/logj, or None.
        """
        x = np.asarray(logj, dtype=float)
        y = np.asarray(E, dtype=float)
        n = len(x)
        min_pts = self._min_tafel_points(n)
        if n < min_pts:
            return None

        max_decades = max(self.min_tafel_decades + 0.2, 1.2)
        r2_gate     = min(self.tafel_r2_target, 0.997)

        cx  = np.concatenate(([0.0], np.cumsum(x)))
        cy  = np.concatenate(([0.0], np.cumsum(y)))
        cxx = np.concatenate(([0.0], np.cumsum(x * x)))
        cyy = np.concatenate(([0.0], np.cumsum(y * y)))
        cxy = np.concatenate(([0.0], np.cumsum(x * y)))

        stride = 1 if n <= 800 else int(np.ceil(n / 800.0))

        best     = None   # (abs_slope, -n_points, i, k)  -> minimise
        fallback = None   # (r2, i, k)

        for i in range(0, n - min_pts + 1, stride):
            for k in range(i + min_pts - 1, n, stride):
                span = x[k] - x[i]
                if span < self.min_tafel_decades:
                    continue
                if span > max_decades:
                    break  # windows only widen as k grows (monotonic logj)
                m = k - i + 1
                sx  = cx[k + 1]  - cx[i]
                sy  = cy[k + 1]  - cy[i]
                sxx = cxx[k + 1] - cxx[i]
                syy = cyy[k + 1] - cyy[i]
                sxy = cxy[k + 1] - cxy[i]

                Sxx = sxx - sx * sx / m
                if Sxx <= 1e-12:
                    continue
                Syy = syy - sy * sy / m
                if Syy <= 1e-12:
                    continue
                Sxy = sxy - sx * sy / m
                r2  = (Sxy * Sxy) / (Sxx * Syy)
                slope = Sxy / Sxx

                if (fallback is None) or (r2 > fallback[0]):
                    fallback = (r2, i, k)
                if r2 >= r2_gate:
                    cand = (abs(slope), -m, i, k)
                    if (best is None) or (cand < best):
                        best = cand

        if best is not None:
            return np.arange(best[2], best[3] + 1)
        if fallback is not None:
            return np.arange(fallback[1], fallback[2] + 1)
        return None

    # ── Overpotential ─────────────────────────────────────────────────────────

    @staticmethod
    def _overpotential_at_j(potential, j, j_target, e_onset) -> float:
        if j_target > np.max(j):
            return float("nan")
        return float(np.interp(j_target, j, potential)) - e_onset

    @staticmethod
    def _half_wave_potential(potential, j, j_lim) -> float:
        j_half = j_lim / 2.0
        if j_half > np.max(j) or j_half < np.min(j):
            return float("nan")
        return float(np.interp(j_half, j, potential))

    # ── Interpretation ────────────────────────────────────────────────────────

    def _interpret_tafel(self, slope_mv: float) -> str:
        if np.isnan(slope_mv):
            return "Tafel slope could not be determined"

        abs_s     = abs(slope_mv)
        ctype     = self.catalyst_type
        el        = self.electrolyte_info
        conc_note = f" [{el.concentration} M {el.compound}]"

        # Metal-free: completely different thresholds
        if ctype == CATALYST_METAL_FREE:
            if abs_s < 120:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Good kinetics for metal-free catalyst. "
                    f"Rate-limiting: surface defect activation or C-H bond breaking."
                )
            elif abs_s < 250:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Normal range for metal-free / ceramic catalyst (carbon_material, CNT, N-doped C). "
                    f"Multi-step mechanism without d-band facilitation."
                )
            else:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Very high slope: strong mass-transport or surface passivation. "
                    f"Check electrode preparation and electrolyte diffusion."
                )

        # Metal oxide
        if ctype == CATALYST_METAL_OXIDE:
            if abs_s < 60:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Excellent for metal oxide. M(OH)x/MOOx redox mediation active."
                )
            elif abs_s < 120:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"Typical for oxide catalyst. First electron transfer limiting."
                )
            else:
                return (
                    f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                    f"High slope for oxide: check OH- supply "
                    f"({'low' if el.concentration < 0.5 else 'normal'} concentration)."
                )

        # Noble metal and alloy
        conc_warning = ""
        if el.is_acidic() and el.compound == ACID_HCl:
            conc_warning = " [Cl- adsorption may inflate Tafel slope]"
        elif el.is_alkaline() and el.concentration < 0.1:
            conc_warning = " [Low OH- — insufficient for bifunctional mechanism]"

        if abs_s < 40:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"Chemical step rate-limiting{conc_warning}"
            )
        elif abs_s < 75:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"Langmuir adsorption limiting — PtRu / PdAu bifunctional mechanism{conc_warning}"
            )
        elif abs_s < 120:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"First electron transfer limiting (Volmer / Butler-Volmer){conc_warning}"
            )
        else:
            return (
                f"Tafel = {slope_mv:.1f} mV/dec{conc_note} — "
                f"High value: CO poisoning or diffusion limiting. "
                f"Consider alloy catalyst or higher {el.compound} concentration.{conc_warning}"
            )

    def _rate_performance(self, e_onset, tafel_slope, eta_10) -> str:
        """
        Performance rating adapted per catalyst type and electrolyte.
        Metal-free catalysts are judged on a different scale.
        """
        el    = self.electrolyte_info
        ctype = self.catalyst_type
        score = 0

        # E_onset thresholds per catalyst type and electrolyte
        onset_thresholds = {
            CATALYST_NOBLE_METAL : (0.20, 0.40) if el.is_alkaline() else (0.40, 0.60),
            CATALYST_ALLOY       : (0.10, 0.30) if el.is_alkaline() else (0.30, 0.50),
            CATALYST_METAL_OXIDE : (1.30, 1.50) if el.is_alkaline() else (1.50, 1.70),
            CATALYST_METAL_FREE  : (0.40, 0.70) if el.is_alkaline() else (0.55, 0.85),
        }
        lo, hi = onset_thresholds.get(ctype, (0.40, 0.60))

        if e_onset < lo:   score += 3
        elif e_onset < hi: score += 1

        # Tafel slope thresholds — metal-free has different scale
        if not np.isnan(tafel_slope):
            if ctype == CATALYST_METAL_FREE:
                if abs(tafel_slope) < 120:   score += 3
                elif abs(tafel_slope) < 200: score += 2
                elif abs(tafel_slope) < 300: score += 1
            elif ctype == CATALYST_METAL_OXIDE:
                if abs(tafel_slope) < 60:   score += 3
                elif abs(tafel_slope) < 100: score += 2
                elif abs(tafel_slope) < 130: score += 1
            else:
                if abs(tafel_slope) < 60:   score += 3
                elif abs(tafel_slope) < 90:  score += 2
                elif abs(tafel_slope) < 120: score += 1

        # Overpotential at 10 mA/cm²
        if not np.isnan(eta_10):
            if ctype == CATALYST_METAL_FREE:
                if eta_10 < 0.3:   score += 3
                elif eta_10 < 0.5: score += 2
                elif eta_10 < 0.7: score += 1
            else:
                if eta_10 < 0.1:   score += 3
                elif eta_10 < 0.2: score += 2
                elif eta_10 < 0.3: score += 1

        # Electrolyte-specific bonus/penalty
        if el.compound == BASE_Na2CO3:
            score = max(0, score - 1)  # Na2CO3 inherently slower
        if el.is_alkaline() and el.concentration >= 1.0 and ctype == CATALYST_METAL_FREE:
            score += 1  # high OH- helps metal-free

        catalyst_label = {
            CATALYST_NOBLE_METAL : "noble metal",
            CATALYST_ALLOY       : "alloy",
            CATALYST_METAL_OXIDE : "metal oxide",
            CATALYST_METAL_FREE  : "metal-free",
        }.get(ctype, "catalyst")

        el_label = f"{el.concentration} M {el.compound}"

        if score >= 8:
            return f"Excellent {catalyst_label} in {el_label} — publishable in high-IF journals"
        elif score >= 5:
            return f"Good {catalyst_label} in {el_label} — suitable catalyst"
        elif score >= 3:
            return f"Moderate {catalyst_label} in {el_label} — optimization needed"
        else:
            return f"Poor performance in {el_label} — consider different composition or electrolyte"

    # ── Smoothing ─────────────────────────────────────────────────────────────

    @staticmethod
    def _smooth(current: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter
            w = min(11, len(current) // 3)
            if w % 2 == 0: w += 1
            if w < 5: return current.copy()
            return savgol_filter(current, window_length=w, polyorder=3)
        except Exception:
            try:
                from scipy.ndimage import gaussian_filter1d
                return gaussian_filter1d(current, sigma=2)
            except Exception:
                return current.copy()

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def for_noble_metal(cls, electrolyte_compound="H2SO4", concentration=0.5, **kw):
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_NOBLE_METAL, **kw)

    @classmethod
    def for_alloy(cls, electrolyte_compound="H2SO4", concentration=0.5, **kw):
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_ALLOY, **kw)

    @classmethod
    def for_carbon_material(cls, electrolyte_compound="KOH", concentration=1.0, **kw):
        """For N-doped Carbon, CNT, rGO, graphene-based — uses wider Tafel thresholds."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_FREE, **kw)

    @classmethod
    def for_metal_oxide(cls, electrolyte_compound="KOH", concentration=1.0, **kw):
        """For NiO, Co3O4, Co2NiO4, MnO2."""
        media = "alkaline" if electrolyte_compound in (BASE_KOH, BASE_NaOH, BASE_Na2CO3, BASE_NH3) else "acidic"
        el    = ElectrolyteInfo(media=media, compound=electrolyte_compound, concentration=concentration)
        return cls(electrolyte=el, catalyst_type=CATALYST_METAL_OXIDE, **kw)

    @staticmethod
    def load_csv(filepath: str) -> tuple[np.ndarray, np.ndarray]:
        for enc in ["latin-1", "cp1252", "utf-8"]:
            try:
                df = pd.read_csv(filepath, comment="#", encoding=enc,
                                 sep=None, engine="python")
                c  = df.columns.tolist()
                return df[c[0]].to_numpy(float), df[c[1]].to_numpy(float)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot read: {filepath}")
