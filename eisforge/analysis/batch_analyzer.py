"""
Batch Statistical Analyzer — n=3 Reproducibility for AOR.
Author: Hoda Jafari | May 2026

Standard in top journals (Nature Catalysis, JACS, Electrochimica Acta):
    E_onset = 0.452 ± 0.008 V  (n=3)
    I_f/I_b = 2.31  ± 0.15     (n=3)
    Tafel   = 78.3  ± 4.2 mV/dec

Without Mean ± SD, reviewers reject the paper.

Usage:
------
    from eisforge.analysis.batch_analyzer import BatchCVAnalyzer, BatchLSVAnalyzer

    # CV batch
    ana = BatchCVAnalyzer(scan_rate=50, electrode_area=0.196,
                          electrolyte='KOH', catalyst_type='noble_metal')
    result = ana.analyze_files(['cv1.csv', 'cv2.csv', 'cv3.csv'])
    print(result.summary())
    print(result.to_markdown_table())

    # LSV batch
    ana_lsv = BatchLSVAnalyzer(scan_rate=5, electrode_area=0.196)
    result_lsv = ana_lsv.analyze_files(['lsv1.csv', 'lsv2.csv', 'lsv3.csv'])
    print(result_lsv.to_latex_table())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _stat(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) for a list of values. Handles NaN gracefully."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    return float(np.mean(clean)), float(np.std(clean, ddof=1) if len(clean) > 1 else 0.0)


def _stat_filtered(values: list[float], outlier_idx: list[int]) -> tuple[float, float]:
    """
    Return (mean, std) excluding outlier indices.
    Outlier indices refer to positions in the original `values` list.
    """
    clean = [v for i, v in enumerate(values)
             if i not in outlier_idx and v is not None and not np.isnan(v)]
    if not clean:
        return _stat(values)   # fallback: use all if everything is flagged
    return float(np.mean(clean)), float(np.std(clean, ddof=1) if len(clean) > 1 else 0.0)


def _grubbs_outliers(values: list[float], alpha: float = 0.05) -> list[int]:
    """
    Iterative Grubbs test for outlier detection (ASTM E178 compliant).
    Returns indices of outlier values at given significance level.
    Only meaningful for n >= 3.
    Iterates until no more outliers are found or fewer than 3 points remain.
    """
    clean = [(i, v) for i, v in enumerate(values) if v is not None and not np.isnan(v)]
    if len(clean) < 3:
        return []

    outlier_indices: list[int] = []
    remaining = list(clean)

    while len(remaining) >= 3:
        vals = np.array([v for _, v in remaining])
        mean, std = vals.mean(), vals.std()
        if std < 1e-10:
            break

        G = np.abs(vals - mean) / std
        G_max_local = int(G.argmax())
        G_max = float(G[G_max_local])

        n = len(vals)
        try:
            t_crit = sp_stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        except Exception:
            G_crit = 2.5   # fallback for very small n

        if G_max > G_crit:
            original_idx = remaining[G_max_local][0]
            outlier_indices.append(original_idx)
            remaining.pop(G_max_local)
        else:
            break

    return outlier_indices


def _to_latex_safe(s: str) -> str:
    """Convert Unicode characters in parameter names to LaTeX equivalents."""
    replacements = {
        "η": r"$\eta$",
        "²": r"$^2$",
        "°": r"$^\circ$",
        "±": r"$\pm$",
        "%": r"\%",
        "_": r"\_",
        "&": r"\&",
        "#": r"\#",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def _fmt_j0(val: float) -> str:
    """Auto-scale exchange current density for readable display."""
    if np.isnan(val):
        return "nan"
    if abs(val) < 1e-3:
        return f"{val * 1e6:.3f} μA/cm²"
    return f"{val:.3e} mA/cm²"


# ── CV Batch Result ───────────────────────────────────────────────────────────

@dataclass(eq=False)
class BatchCVResult:
    """
    Statistical results from batch CV analysis (n=3 or more).

    All metrics reported as (mean, std) pairs, with outliers excluded.
    Also contains the averaged curve with error band for Plotly.
    """
    n_files        : int
    n_valid        : int
    outlier_indices: list[int] = field(default_factory=list)
    catalyst_type  : str = "noble_metal"
    electrolyte    : str = "unknown"

    # ── Key metrics (mean, std — outliers excluded) ──────────────────────────
    e_onset_mean   : float = float("nan")
    e_onset_std    : float = float("nan")
    e_fwd_peak_mean: float = float("nan")
    e_fwd_peak_std : float = float("nan")
    i_fwd_mean     : float = float("nan")
    i_fwd_std      : float = float("nan")
    i_bwd_mean     : float = float("nan")
    i_bwd_std      : float = float("nan")
    if_ib_mean     : float = float("nan")
    if_ib_std      : float = float("nan")
    j_fwd_mean     : float = float("nan")
    j_fwd_std      : float = float("nan")
    cdl_mean       : float = float("nan")
    cdl_std        : float = float("nan")

    # ── Averaged curve (for Plotly) — repr=False to avoid huge print output ──
    potential_common   : Optional[np.ndarray] = field(default=None, repr=False)
    current_mean_curve : Optional[np.ndarray] = field(default=None, repr=False)
    current_std_curve  : Optional[np.ndarray] = field(default=None, repr=False)

    # ── Raw per-file results ─────────────────────────────────────────────────
    individual_results : list = field(default_factory=list, repr=False)

    def summary(self) -> str:
        is_mf = self.catalyst_type == "carbon_material"
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [
            "=" * 65,
            f"  Batch CV Analysis — n={self.n_valid}/{self.n_files} valid  "
            f"(n_eff={n_eff} after outlier removal)",
            f"  Catalyst: {self.catalyst_type} | Electrolyte: {self.electrolyte}",
            "=" * 65,
            f"  E_onset     = {self.e_onset_mean:.4f} ± {self.e_onset_std:.4f} V",
            f"  I_forward   = {self.i_fwd_mean:.4f} ± {self.i_fwd_std:.4f} mA",
            f"  j_forward   = {self.j_fwd_mean:.4f} ± {self.j_fwd_std:.4f} mA/cm²",
        ]
        if is_mf:
            lines.append(
                f"  C_dl        = {self.cdl_mean:.4f} ± {self.cdl_std:.4f} mF/cm²"
            )
        else:
            lines.append(
                f"  I_f/I_b     = {self.if_ib_mean:.3f} ± {self.if_ib_std:.3f}"
            )
        if self.outlier_indices:
            lines.append(f"  Outliers    : files {self.outlier_indices} (excluded from stats)")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        is_mf = self.catalyst_type == "carbon_material"
        rows = [
            ("E_onset (V)",          f"{self.e_onset_mean:.4f}",    f"{self.e_onset_std:.4f}"),
            ("E_forward_peak (V)",   f"{self.e_fwd_peak_mean:.4f}", f"{self.e_fwd_peak_std:.4f}"),
            ("I_forward (mA)",       f"{self.i_fwd_mean:.4f}",      f"{self.i_fwd_std:.4f}"),
            ("j_forward (mA/cm²)",  f"{self.j_fwd_mean:.4f}",      f"{self.j_fwd_std:.4f}"),
        ]
        if is_mf:
            rows.append(("C_dl (mF/cm²)", f"{self.cdl_mean:.4f}", f"{self.cdl_std:.4f}"))
        else:
            rows += [
                ("I_backward (mA)",  f"{self.i_bwd_mean:.4f}", f"{self.i_bwd_std:.4f}"),
                ("I_f / I_b",        f"{self.if_ib_mean:.3f}",  f"{self.if_ib_std:.3f}"),
            ]
        return pd.DataFrame(rows, columns=["Parameter", "Mean", "SD"])

    def to_markdown_table(self) -> str:
        df = self.to_dataframe()
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [
            f"| Parameter | Mean ± SD (n={n_eff}) |",
            "|---|---|",
        ]
        for _, row in df.iterrows():
            lines.append(f"| {row['Parameter']} | {row['Mean']} ± {row['SD']} |")
        return "\n".join(lines)

    def to_latex_table(self) -> str:
        df = self.to_dataframe()
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Electrochemical performance metrics (mean $\pm$ SD, n="
            + str(n_eff) + r")}",
            r"\begin{tabular}{lc}",
            r"\hline",
            r"Parameter & Mean $\pm$ SD \\ \hline",
        ]
        for _, row in df.iterrows():
            param = _to_latex_safe(row["Parameter"])
            lines.append(f"{param} & ${row['Mean']} \\pm {row['SD']}$ \\\\")
        lines += [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)


@dataclass(eq=False)
class BatchLSVResult:
    """Statistical results from batch LSV analysis."""
    n_files        : int
    n_valid        : int
    outlier_indices: list[int] = field(default_factory=list)
    catalyst_type  : str = "noble_metal"
    electrolyte    : str = "unknown"

    e_onset_mean   : float = float("nan")
    e_onset_std    : float = float("nan")
    tafel_mean     : float = float("nan")
    tafel_std      : float = float("nan")
    j0_mean        : float = float("nan")
    j0_std         : float = float("nan")
    eta10_mean     : float = float("nan")
    eta10_std      : float = float("nan")
    eta50_mean     : float = float("nan")
    eta50_std      : float = float("nan")
    eta100_mean    : float = float("nan")
    eta100_std     : float = float("nan")
    mass_act_mean  : float = float("nan")
    mass_act_std   : float = float("nan")
    spec_act_mean  : float = float("nan")
    spec_act_std   : float = float("nan")

    potential_common  : Optional[np.ndarray] = field(default=None, repr=False)
    j_mean_curve      : Optional[np.ndarray] = field(default=None, repr=False)
    j_std_curve       : Optional[np.ndarray] = field(default=None, repr=False)
    individual_results: list = field(default_factory=list, repr=False)

    def summary(self) -> str:
        is_mf = self.catalyst_type == "carbon_material"
        tafel_note = " (normal for metal-free)" if is_mf else ""
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [
            "=" * 65,
            f"  Batch LSV Analysis — n={self.n_valid}/{self.n_files} valid  "
            f"(n_eff={n_eff} after outlier removal)",
            f"  Catalyst: {self.catalyst_type} | Electrolyte: {self.electrolyte}",
            "=" * 65,
            f"  E_onset     = {self.e_onset_mean:.4f} ± {self.e_onset_std:.4f} V",
            f"  Tafel slope = {self.tafel_mean:.1f} ± {self.tafel_std:.1f} mV/dec{tafel_note}",
            f"  j0          = {_fmt_j0(self.j0_mean)} ± {_fmt_j0(self.j0_std)}",
            f"  η @ 10      = {self.eta10_mean*1000:.1f} ± {self.eta10_std*1000:.1f} mV",
            f"  η @ 50      = {self.eta50_mean*1000:.1f} ± {self.eta50_std*1000:.1f} mV",
            f"  η @ 100     = {self.eta100_mean*1000:.1f} ± {self.eta100_std*1000:.1f} mV",
        ]
        if not np.isnan(self.mass_act_mean):
            lines.append(f"  Mass act.   = {self.mass_act_mean:.3f} ± {self.mass_act_std:.3f} mA/mg")
        if self.outlier_indices:
            lines.append(f"  Outliers    : files {self.outlier_indices} (excluded from stats)")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        is_mf = self.catalyst_type == "carbon_material"
        rows = [
            ("E_onset (V vs RHE)",   f"{self.e_onset_mean:.4f}",        f"{self.e_onset_std:.4f}"),
            ("Tafel slope (mV/dec)", f"{self.tafel_mean:.1f}",           f"{self.tafel_std:.1f}"),
            ("j0",                   _fmt_j0(self.j0_mean),              _fmt_j0(self.j0_std)),
            ("η @ 10 mA/cm² (mV)",  f"{self.eta10_mean*1000:.1f}",     f"{self.eta10_std*1000:.1f}"),
            ("η @ 50 mA/cm² (mV)",  f"{self.eta50_mean*1000:.1f}",     f"{self.eta50_std*1000:.1f}"),
            ("η @ 100 mA/cm² (mV)", f"{self.eta100_mean*1000:.1f}",    f"{self.eta100_std*1000:.1f}"),
        ]
        if not np.isnan(self.mass_act_mean):
            rows.append(("Mass activity (mA/mg)",
                         f"{self.mass_act_mean:.3f}", f"{self.mass_act_std:.3f}"))
        if not np.isnan(self.spec_act_mean):
            label = ("Specific activity (mA/cm²_BET)"
                     if is_mf else "Specific activity (mA/cm²_Pt)")
            rows.append((label,
                         f"{self.spec_act_mean:.4f}", f"{self.spec_act_std:.4f}"))
        return pd.DataFrame(rows, columns=["Parameter", "Mean", "SD"])

    def to_markdown_table(self) -> str:
        df = self.to_dataframe()
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [f"| Parameter | Mean ± SD (n={n_eff}) |", "|---|---|"]
        for _, row in df.iterrows():
            lines.append(f"| {row['Parameter']} | {row['Mean']} ± {row['SD']} |")
        return "\n".join(lines)

    def to_latex_table(self) -> str:
        df = self.to_dataframe()
        n_eff = self.n_valid - len(self.outlier_indices)
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{LSV performance metrics (mean $\pm$ SD, n="
            + str(n_eff) + r")}",
            r"\begin{tabular}{lc}",
            r"\hline",
            r"Parameter & Mean $\pm$ SD \\ \hline",
        ]
        for _, row in df.iterrows():
            param = _to_latex_safe(row["Parameter"])
            lines.append(f"{param} & ${row['Mean']} \\pm {row['SD']}$ \\\\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)


# ── Potential axis alignment ──────────────────────────────────────────────────

def _align_to_common_axis(
    potentials: list[np.ndarray],
    currents  : list[np.ndarray],
    n_points  : int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate all CV/LSV curves to a common potential axis.
    NaN-safe: uses left/right=np.nan so out-of-range points are excluded
    from mean/std via a valid-point mask.

    Returns (potential_common, current_mean, current_std).
    """
    e_min = max(p.min() for p in potentials)
    e_max = min(p.max() for p in potentials)

    if e_min >= e_max:
        e_min = min(p.min() for p in potentials)
        e_max = max(p.max() for p in potentials)

    potential_common = np.linspace(e_min, e_max, n_points)
    interpolated: list[np.ndarray] = []

    for pot, cur in zip(potentials, currents):
        order = np.argsort(pot)
        pot_s, cur_s = pot[order], cur[order]
        _, unique_idx = np.unique(pot_s, return_index=True)
        pot_u, cur_u = pot_s[unique_idx], cur_s[unique_idx]
        interp = np.interp(potential_common, pot_u, cur_u,
                           left=np.nan, right=np.nan)
        interpolated.append(interp)

    stack = np.stack(interpolated)          # (n_files, n_points)
    valid = ~np.isnan(stack).any(axis=0)    # columns where ALL files have data

    if not valid.any():
        # fallback: use all columns even if some NaN
        valid = np.ones(n_points, dtype=bool)

    mean_curve = np.nanmean(stack[:, valid], axis=0)
    std_curve = (
        np.nanstd(stack[:, valid], axis=0, ddof=1)
        if len(interpolated) > 1
        else np.zeros(valid.sum())
    )
    return potential_common[valid], mean_curve, std_curve


# ── Batch CV Analyzer ────────────────────────────────────────────────────────

class BatchCVAnalyzer:
    """
    Batch CV analysis for n=3 (or more) reproducibility.

    Accepts numpy arrays OR file paths. For file paths, uses CVAnalyzer internally.

    Parameters — same as CVAnalyzer.
    """

    def __init__(
        self,
        scan_rate              : float = 50.0,
        electrode_area         : float = 1.0,
        ecsa                   : float = 0.0,
        catalyst_type          : str   = "noble_metal",
        electrolyte                    = "acidic",
        electrolyte_concentration: float = 0.5,
        onset_method           : str   = "tangent",
        current_unit           : str   = "mA",
        catalyst_loading       : float = 0.0,
        r_s_ohms               : float = 0.0,
        detect_outliers        : bool  = True,
    ) -> None:
        self.scan_rate                = scan_rate
        self.electrode_area           = electrode_area
        self.ecsa                     = ecsa
        self.catalyst_type            = catalyst_type
        self.electrolyte              = electrolyte
        self.electrolyte_concentration = electrolyte_concentration
        self.onset_method             = onset_method
        self.current_unit             = current_unit
        self.catalyst_loading         = catalyst_loading
        self.r_s_ohms                 = r_s_ohms
        self.detect_outliers          = detect_outliers

    def analyze_arrays(
        self,
        potentials: list[np.ndarray],
        currents  : list[np.ndarray],
    ) -> BatchCVResult:
        """
        Analyze a list of (potential, current) arrays.

        Parameters
        ----------
        potentials : list of np.ndarray — one per measurement
        currents   : list of np.ndarray — one per measurement
        """
        from eisforge.analysis.cv_analyzer import CVAnalyzer, ElectrolyteInfo

        el = (
            self.electrolyte
            if isinstance(self.electrolyte, ElectrolyteInfo)
            else ElectrolyteInfo.from_string(
                str(self.electrolyte), concentration=self.electrolyte_concentration
            )
        )

        results = []
        errors  = []

        for i, (pot, cur) in enumerate(zip(potentials, currents)):
            try:
                ana = CVAnalyzer(
                    scan_rate=self.scan_rate,
                    electrode_area=self.electrode_area,
                    ecsa=self.ecsa,
                    catalyst_loading=self.catalyst_loading,
                    onset_method=self.onset_method,
                    electrolyte=el,
                    catalyst_type=self.catalyst_type,
                    current_unit=self.current_unit,
                )
                r = ana.analyze(pot, cur, r_s_ohms=self.r_s_ohms)
                results.append(r)
            except Exception as exc:
                logger.warning(f"File {i} failed: {exc}")
                errors.append(i)

        n_valid = len(results)
        if n_valid == 0:
            raise ValueError("All files failed to analyze.")

        # ── Extract raw metrics ───────────────────────────────────────────────
        e_onsets    = [r.e_onset         for r in results]
        e_fwd_peaks = [r.e_forward_peak  for r in results]
        i_fwds      = [r.i_forward_peak  for r in results]
        i_bwds      = [r.i_backward_peak for r in results]
        if_ibs      = [r.if_ib_ratio     for r in results]
        j_fwds      = [r.j_forward_peak  for r in results]
        cdls        = [r.cdl_mF_cm2      for r in results]

        # ── Outlier detection (iterative Grubbs) ──────────────────────────────
        outliers = _grubbs_outliers(e_onsets) if self.detect_outliers else []

        # ── Averaged curve ────────────────────────────────────────────────────
        pots_arr = [np.asarray(pot) for pot in potentials]
        curs_arr = [np.asarray(cur) for cur in currents]
        try:
            pot_common, cur_mean, cur_std = _align_to_common_axis(pots_arr, curs_arr)
        except Exception:
            pot_common = cur_mean = cur_std = None

        return BatchCVResult(
            n_files             = len(potentials),
            n_valid             = n_valid,
            outlier_indices     = outliers,
            catalyst_type       = self.catalyst_type,
            electrolyte         = el.label() if hasattr(el, "label") else str(self.electrolyte),
            # ── Stats use _stat_filtered to exclude outliers ──
            e_onset_mean        = _stat_filtered(e_onsets,    outliers)[0],
            e_onset_std         = _stat_filtered(e_onsets,    outliers)[1],
            e_fwd_peak_mean     = _stat_filtered(e_fwd_peaks, outliers)[0],
            e_fwd_peak_std      = _stat_filtered(e_fwd_peaks, outliers)[1],
            i_fwd_mean          = _stat_filtered(i_fwds,      outliers)[0],
            i_fwd_std           = _stat_filtered(i_fwds,      outliers)[1],
            i_bwd_mean          = _stat_filtered(i_bwds,      outliers)[0],
            i_bwd_std           = _stat_filtered(i_bwds,      outliers)[1],
            if_ib_mean          = _stat_filtered(if_ibs,      outliers)[0],
            if_ib_std           = _stat_filtered(if_ibs,      outliers)[1],
            j_fwd_mean          = _stat_filtered(j_fwds,      outliers)[0],
            j_fwd_std           = _stat_filtered(j_fwds,      outliers)[1],
            cdl_mean            = _stat_filtered(cdls,        outliers)[0],
            cdl_std             = _stat_filtered(cdls,        outliers)[1],
            potential_common    = pot_common,
            current_mean_curve  = cur_mean,
            current_std_curve   = cur_std,
            individual_results  = results,
        )

    def analyze_files(self, file_paths: list[str]) -> BatchCVResult:
        """Analyze a list of CV file paths (CSV format: potential, current columns)."""
        from eisforge.analysis.cv_analyzer import CVAnalyzer
        potentials, currents = [], []
        for path in file_paths:
            pot, cur = CVAnalyzer.load_csv(path)
            potentials.append(pot)
            currents.append(cur)
        return self.analyze_arrays(potentials, currents)


# ── Batch LSV Analyzer ───────────────────────────────────────────────────────

class BatchLSVAnalyzer:
    """Batch LSV analysis for n=3 (or more) reproducibility."""

    def __init__(
        self,
        scan_rate              : float = 5.0,
        electrode_area         : float = 1.0,
        ecsa                   : float = 0.0,
        catalyst_type          : str   = "noble_metal",
        electrolyte                    = "acidic",
        electrolyte_concentration: float = 0.5,
        catalyst_loading       : float = 0.0,
        e_ref_vs_rhe           : float = 0.0,
        tafel_current_range    : tuple = None,
        current_unit           : str   = "mA",
        r_s_ohms               : float = 0.0,
        detect_outliers        : bool  = True,
    ) -> None:
        self.scan_rate                = scan_rate
        self.electrode_area           = electrode_area
        self.ecsa                     = ecsa
        self.catalyst_type            = catalyst_type
        self.electrolyte              = electrolyte
        self.electrolyte_concentration = electrolyte_concentration
        self.catalyst_loading         = catalyst_loading
        self.e_ref_vs_rhe             = e_ref_vs_rhe
        self.tafel_current_range      = tafel_current_range
        self.current_unit             = current_unit
        self.r_s_ohms                 = r_s_ohms
        self.detect_outliers          = detect_outliers

    def analyze_arrays(
        self,
        potentials: list[np.ndarray],
        currents  : list[np.ndarray],
    ) -> BatchLSVResult:
        from eisforge.analysis.lsv_analyzer import LSVAnalyzer, ElectrolyteInfo

        el = (
            self.electrolyte
            if isinstance(self.electrolyte, ElectrolyteInfo)
            else ElectrolyteInfo.from_string(
                str(self.electrolyte), concentration=self.electrolyte_concentration
            )
        )

        results = []
        for i, (pot, cur) in enumerate(zip(potentials, currents)):
            try:
                la = LSVAnalyzer(
                    scan_rate=self.scan_rate,
                    electrode_area=self.electrode_area,
                    ecsa=self.ecsa,
                    catalyst_loading=self.catalyst_loading,
                    electrolyte=el,
                    catalyst_type=self.catalyst_type,
                    e_ref_vs_rhe=self.e_ref_vs_rhe,
                    tafel_current_range=self.tafel_current_range or (0.1, 2.0),
                    current_unit=self.current_unit,
                )
                results.append(la.analyze(pot, cur, r_s_ohms=self.r_s_ohms))
            except Exception as exc:
                logger.warning(f"File {i} failed: {exc}")

        n_valid = len(results)
        if n_valid == 0:
            raise ValueError("All files failed to analyze.")

        e_onsets  = [r.e_onset                 for r in results]
        tafels    = [r.tafel_slope              for r in results]
        j0s       = [r.exchange_current_density for r in results]
        eta10s    = [r.overpotential_10         for r in results]
        eta50s    = [r.overpotential_50         for r in results]
        eta100s   = [r.overpotential_100        for r in results]
        mass_acts = [r.mass_activity            for r in results]
        spec_acts = [r.specific_activity        for r in results]

        outliers = _grubbs_outliers(e_onsets) if self.detect_outliers else []

        pots_arr = [np.asarray(p) for p in potentials]
        curs_arr = [np.asarray(c) / self.electrode_area for c in currents]
        try:
            pot_c, j_mean, j_std = _align_to_common_axis(pots_arr, curs_arr)
        except Exception:
            pot_c = j_mean = j_std = None

        return BatchLSVResult(
            n_files             = len(potentials),
            n_valid             = n_valid,
            outlier_indices     = outliers,
            catalyst_type       = self.catalyst_type,
            electrolyte         = el.label() if hasattr(el, "label") else str(self.electrolyte),
            e_onset_mean        = _stat_filtered(e_onsets,  outliers)[0],
            e_onset_std         = _stat_filtered(e_onsets,  outliers)[1],
            tafel_mean          = _stat_filtered(tafels,    outliers)[0],
            tafel_std           = _stat_filtered(tafels,    outliers)[1],
            j0_mean             = _stat_filtered(j0s,       outliers)[0],
            j0_std              = _stat_filtered(j0s,       outliers)[1],
            eta10_mean          = _stat_filtered(eta10s,    outliers)[0],
            eta10_std           = _stat_filtered(eta10s,    outliers)[1],
            eta50_mean          = _stat_filtered(eta50s,    outliers)[0],
            eta50_std           = _stat_filtered(eta50s,    outliers)[1],
            eta100_mean         = _stat_filtered(eta100s,   outliers)[0],
            eta100_std          = _stat_filtered(eta100s,   outliers)[1],
            mass_act_mean       = _stat_filtered(mass_acts, outliers)[0],
            mass_act_std        = _stat_filtered(mass_acts, outliers)[1],
            spec_act_mean       = _stat_filtered(spec_acts, outliers)[0],
            spec_act_std        = _stat_filtered(spec_acts, outliers)[1],
            potential_common    = pot_c,
            j_mean_curve        = j_mean,
            j_std_curve         = j_std,
            individual_results  = results,
        )

    def analyze_files(self, file_paths: list[str]) -> BatchLSVResult:
        """Analyze a list of LSV file paths (CSV format: potential, current columns)."""
        from eisforge.analysis.lsv_analyzer import LSVAnalyzer
        potentials, currents = [], []
        for path in file_paths:
            pot, cur = LSVAnalyzer.load_csv(path)
            potentials.append(pot)
            currents.append(cur)
        return self.analyze_arrays(potentials, currents)
