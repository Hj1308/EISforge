"""
Batch EIS Fitter — patch19
======================================================
Warm-start sequential CNLS fitting for multi-condition
EIS series (e.g., different alcohol concentrations).

Key idea (from ZScope):  use the best-fit parameters of
condition i as the initial guess for condition i+1.  This
converges ~60-80 % faster than independent cold starts and
also avoids landing in different local minima across the series.

Usage
-----
from eisforge.core.batch_fitter import BatchFitter

fitter = BatchFitter(circuit="R0-p(R1,CPE1)-p(R2,W1)",
                     p0=p0_dict, bounds=bounds_dict)
results = fitter.fit_series(freq_list, zreal_list, zimag_list)
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


# ── circuit element library ───────────────────────────────────────────────────

def _Z_R(p: float, freq: np.ndarray) -> np.ndarray:
    return np.full(len(freq), complex(p, 0))


def _Z_C(p: float, freq: np.ndarray) -> np.ndarray:
    omega = 2 * np.pi * freq
    return 1.0 / (1j * omega * p)


def _Z_L(p: float, freq: np.ndarray) -> np.ndarray:
    omega = 2 * np.pi * freq
    return 1j * omega * p


def _Z_CPE(Q: float, n: float, freq: np.ndarray) -> np.ndarray:
    """CPE: Z = 1 / (Q * (jω)^n)"""
    omega = 2 * np.pi * freq
    return 1.0 / (Q * (1j * omega) ** n)


def _Z_W(sigma: float, freq: np.ndarray) -> np.ndarray:
    """Semi-infinite Warburg: Z = σ/√ω * (1-j)"""
    omega = 2 * np.pi * freq
    return (sigma / np.sqrt(omega)) * (1 - 1j)


def _Z_Wo(R: float, tau: float, freq: np.ndarray) -> np.ndarray:
    """Finite-length (open) Warburg: Z = R * coth(√(jω·τ)) / √(jω·τ)"""
    omega = 2 * np.pi * freq
    x = np.sqrt(1j * omega * tau)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return R / (x * np.tanh(x))


# ── simple recursive circuit parser ──────────────────────────────────────────

class CircuitEvaluator:
    """
    Evaluates a circuit string such as:
        "R0-p(R1,CPE1)-p(R2,Wo1)"
    given a flat parameter vector.

    Supported elements (case-insensitive tokens):
        R#, C#, L#, CPE# (2 params: Q, n), W# (sigma), Wo# (R, tau)
    Series: A-B-C
    Parallel: p(A,B)
    """

    ELEM_NPARAMS = {"R": 1, "C": 1, "L": 1, "CPE": 2, "W": 1, "WO": 2, "WS": 2}

    def __init__(self, circuit_str: str):
        self.circuit_str = circuit_str.strip()
        self._tokens = self._parse_tokens(circuit_str)
        self.n_params = sum(
            self.ELEM_NPARAMS.get(t["kind"].upper(), 1)
            for t in self._tokens
        )

    # -- public API ---

    def impedance(self, params: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """Return complex impedance array (same length as freq)."""
        z, _ = self._eval(self.circuit_str, list(params), freq, 0)
        return z

    def param_names(self) -> List[str]:
        names = []
        for t in self._tokens:
            k = t["kind"].upper()
            idx = t["idx"]
            if k in ("CPE",):
                names += [f"Q{idx}", f"n{idx}"]
            elif k in ("WO", "WS"):
                names += [f"R_W{idx}", f"tau{idx}"]
            else:
                names.append(f"{k}{idx}")
        return names

    # -- internals ---

    def _parse_tokens(self, s: str) -> List[dict]:
        """Very simple tokeniser — just extract element kinds and indices."""
        import re
        tokens = []
        for m in re.finditer(r"(CPE|WO|WS|W|R|C|L)(\d+)", s, re.IGNORECASE):
            tokens.append({"kind": m.group(1).upper(), "idx": m.group(2)})
        return tokens

    def _eval(self, s: str, params: list, freq: np.ndarray,
               p_idx: int) -> Tuple[np.ndarray, int]:
        """Recursively evaluate series/parallel expressions."""
        import re
        s = s.strip()

        # parallel: p(A,B,...)
        if s.lower().startswith("p(") and s.endswith(")"):
            inner = s[2:-1]
            parts = self._split_top_level(inner)
            z_total = None
            for part in parts:
                z_part, p_idx = self._eval(part, params, freq, p_idx)
                if z_total is None:
                    z_total = 1.0 / z_part
                else:
                    z_total = z_total + 1.0 / z_part
            return 1.0 / z_total, p_idx

        # series: A-B-C  (top-level dashes only)
        parts = self._split_top_level(s, delimiter="-")
        if len(parts) > 1:
            z_total = np.zeros(len(freq), dtype=complex)
            for part in parts:
                z_part, p_idx = self._eval(part, params, freq, p_idx)
                z_total = z_total + z_part
            return z_total, p_idx

        # single element
        m = re.match(r"(CPE|WO|WS|W|R|C|L)(\d+)$", s, re.IGNORECASE)
        if not m:
            raise ValueError(f"Unknown circuit element: '{s}'")
        kind = m.group(1).upper()
        if kind == "R":
            z = _Z_R(params[p_idx], freq); p_idx += 1
        elif kind == "C":
            z = _Z_C(params[p_idx], freq); p_idx += 1
        elif kind == "L":
            z = _Z_L(params[p_idx], freq); p_idx += 1
        elif kind == "CPE":
            z = _Z_CPE(params[p_idx], params[p_idx + 1], freq); p_idx += 2
        elif kind == "W":
            z = _Z_W(params[p_idx], freq); p_idx += 1
        elif kind in ("WO", "WS"):
            z = _Z_Wo(params[p_idx], params[p_idx + 1], freq); p_idx += 2
        else:
            raise ValueError(f"Unsupported element kind: {kind}")
        return z, p_idx

    @staticmethod
    def _split_top_level(s: str, delimiter: str = ",") -> List[str]:
        """Split *s* by *delimiter* only at the top nesting level."""
        parts, depth, buf = [], 0, []
        for ch in s:
            if ch == "(":
                depth += 1; buf.append(ch)
            elif ch == ")":
                depth -= 1; buf.append(ch)
            elif ch == delimiter and depth == 0:
                parts.append("".join(buf).strip()); buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf).strip())
        return parts


# ── data classes ─────────────────────────────────────────────────────────────

@dataclasses.dataclass
class EISFitResult:
    """Result of a single CNLS fit."""
    success: bool
    params: np.ndarray           # best-fit parameter vector
    param_names: List[str]       # human-readable names
    chi2: float                  # modulus-weighted chi-squared
    residuals: np.ndarray        # complex residuals
    z_fit: np.ndarray            # fitted impedance
    message: str = ""
    label: str = ""              # e.g. "0.25 M MeOH"

    @property
    def params_dict(self) -> Dict[str, float]:
        return dict(zip(self.param_names, self.params))


@dataclasses.dataclass
class BatchEISResult:
    """Collection of EIS fits over a series of conditions."""
    results: List[EISFitResult]
    labels: List[str]
    conditions: Optional[List[float]] = None   # e.g. concentrations

    @property
    def success_mask(self) -> List[bool]:
        return [r.success for r in self.results]

    @property
    def n_success(self) -> int:
        return sum(self.success_mask)

    def get_param_series(self, name: str) -> np.ndarray:
        """Return array of one parameter across all results (NaN for failed fits)."""
        vals = []
        for r in self.results:
            v = r.params_dict.get(name, np.nan)
            vals.append(v if r.success else np.nan)
        return np.array(vals)

    def to_dataframe(self):
        """Returns a pandas DataFrame — one row per condition."""
        import pandas as pd
        if not self.results:
            return pd.DataFrame()
        param_names = self.results[0].param_names
        rows = []
        for i, r in enumerate(self.results):
            row = {
                "label": self.labels[i],
                "success": r.success,
                "chi2": r.chi2,
            }
            if self.conditions is not None:
                row["condition"] = self.conditions[i]
            for j, name in enumerate(param_names):
                row[name] = r.params[j] if r.success else np.nan
            rows.append(row)
        return pd.DataFrame(rows)


# ── core fitter ───────────────────────────────────────────────────────────────

class BatchFitter:
    """
    Warm-start sequential CNLS fitter for a series of EIS spectra.

    Parameters
    ----------
    circuit : str
        Circuit string, e.g. "R0-p(R1,CPE1)-p(R2,W1)".
    p0 : array-like
        Initial parameter guess for the *first* spectrum.
    bounds : tuple of two array-likes, optional
        (lower_bounds, upper_bounds).  Defaults to (0, +inf).
    method : str
        scipy least_squares method: "trf" (default) or "lm".
    warm_start : bool
        If True (default), propagate best-fit params as p0 for next fit.
    loss : str
        Loss function for least_squares: "linear" (default) or "soft_l1".
    """

    def __init__(
        self,
        circuit: str,
        p0: np.ndarray,
        bounds: Optional[Tuple] = None,
        method: str = "trf",
        warm_start: bool = True,
        loss: str = "linear",
    ):
        self.evaluator = CircuitEvaluator(circuit)
        self._p0_init = np.asarray(p0, dtype=float)
        self._bounds = bounds if bounds is not None else (
            np.zeros(len(self._p0_init)),
            np.full(len(self._p0_init), np.inf),
        )
        self.method = method
        self.warm_start = warm_start
        self.loss = loss

    # ---- single fit ----------------------------------------------------------

    def fit_one(
        self,
        freq: np.ndarray,
        z_real: np.ndarray,
        z_imag: np.ndarray,
        p0: Optional[np.ndarray] = None,
        label: str = "",
    ) -> EISFitResult:
        """
        Fit a single spectrum.  Uses *p0* if provided, else self._p0_init.
        Returns an EISFitResult.
        """
        freq = np.asarray(freq, dtype=float)
        z_exp = np.asarray(z_real, dtype=float) + 1j * np.asarray(z_imag, dtype=float)
        p0_use = np.asarray(p0, dtype=float) if p0 is not None else self._p0_init.copy()

        def _residuals(params):
            try:
                z_fit = self.evaluator.impedance(params, freq)
                res = (z_fit - z_exp) / (np.abs(z_exp) + 1e-30)
                return np.concatenate([res.real, res.imag])
            except Exception:
                return np.full(2 * len(freq), 1e6)

        try:
            out = least_squares(
                _residuals,
                p0_use,
                bounds=self._bounds,
                method=self.method,
                loss=self.loss,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=5000,
            )
            z_fit = self.evaluator.impedance(out.x, freq)
            chi2 = float(np.sum(np.abs((z_fit - z_exp) / np.abs(z_exp)) ** 2))
            return EISFitResult(
                success=out.success,
                params=out.x,
                param_names=self.evaluator.param_names(),
                chi2=chi2,
                residuals=z_fit - z_exp,
                z_fit=z_fit,
                message=out.message,
                label=label,
            )
        except Exception as exc:
            return EISFitResult(
                success=False,
                params=p0_use,
                param_names=self.evaluator.param_names(),
                chi2=np.inf,
                residuals=np.zeros(len(freq), dtype=complex),
                z_fit=np.zeros(len(freq), dtype=complex),
                message=str(exc),
                label=label,
            )

    # ---- batch (warm-start) fit ----------------------------------------------

    def fit_series(
        self,
        freq_list: List[np.ndarray],
        zreal_list: List[np.ndarray],
        zimag_list: List[np.ndarray],
        labels: Optional[List[str]] = None,
        conditions: Optional[List[float]] = None,
    ) -> BatchEISResult:
        """
        Fit a series of EIS spectra with optional warm-starting.

        Parameters
        ----------
        freq_list, zreal_list, zimag_list :
            Lists of equal length; each element is one spectrum.
        labels : list of str, optional
            Human-readable label for each spectrum.
        conditions : list of float, optional
            Physical condition (e.g. concentration in M) for each spectrum.
            Used for x-axis of trend plots.

        Returns
        -------
        BatchEISResult
        """
        n = len(freq_list)
        if labels is None:
            labels = [f"Spectrum {i + 1}" for i in range(n)]

        results = []
        p0 = self._p0_init.copy()

        for i in range(n):
            result = self.fit_one(
                freq_list[i], zreal_list[i], zimag_list[i],
                p0=p0, label=labels[i],
            )
            results.append(result)

            # warm-start: propagate successful fit to next spectrum
            if self.warm_start and result.success:
                p0 = result.params.copy()
            # if fit failed, keep the last good p0 (no change)

        return BatchEISResult(
            results=results,
            labels=labels,
            conditions=conditions,
        )
