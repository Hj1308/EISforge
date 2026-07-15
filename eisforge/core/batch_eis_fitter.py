"""
EISForge — patch19: Warm-Start Batch EIS Fitter
================================================
Sequential CNLS fitting across a series of EIS spectra (e.g. different
alcohol concentrations, temperatures, or potentials).  Each successful
fit seeds the initial parameters of the next spectrum — "warm-starting" —
which dramatically reduces convergence failures and speeds up fitting
for slowly-varying datasets (typical in AOR / fuel-cell studies).

Usage
-----
>>> from eisforge.core.batch_eis_fitter import WarmStartBatchFitter, BatchEISDataset
>>> datasets = [
...     BatchEISDataset(freq, zr, zi, label="0.1 M MeOH"),
...     BatchEISDataset(freq2, zr2, zi2, label="0.5 M MeOH"),
...     BatchEISDataset(freq3, zr3, zi3, label="1.0 M MeOH"),
... ]
>>> fitter = WarmStartBatchFitter(circuit="R0-p(Rct,CPE1)", bounds_low=[0]*5, bounds_high=[1e6]*4+[1.0])
>>> results = fitter.fit(datasets, initial_params=[10, 50, 1e-5, 0.85])
>>> for r in results:
...     print(r.label, r.params)

Reference
---------
Barsoukov & Macdonald (2005) *Impedance Spectroscopy*, §3.5
Author: Hoda Jafari | EISForge 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.optimize import least_squares


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchEISDataset:
    """Single EIS spectrum for batch processing."""
    frequency: np.ndarray          # Hz
    z_real: np.ndarray             # Ω
    z_imag: np.ndarray             # Ω  (positive = capacitive convention)
    label: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def Z_complex(self) -> np.ndarray:
        return self.z_real - 1j * self.z_imag


@dataclass
class BatchEISResult:
    """Fit result for one EIS spectrum in the batch."""
    label: str
    params: Optional[np.ndarray]   # best-fit parameters (None if failed)
    param_errors: Optional[np.ndarray]  # 1-sigma std from covariance diagonal
    chi2: float                    # reduced chi-squared
    circuit: str
    success: bool
    n_iter: int
    message: str
    # convenience: named-parameter dict (populated by fitter if names supplied)
    named_params: Dict[str, float] = field(default_factory=dict)
    named_errors: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit evaluator  (forward model)
# ─────────────────────────────────────────────────────────────────────────────

class _CircuitEvaluator:
    """
    Minimal circuit evaluator that understands the EISForge circuit string
    format: elements separated by '-' (series) and 'p(A,B)' for parallel.

    Supported elements
    ------------------
    R<n>      — resistor,        1 param: R
    C<n>      — capacitor,       1 param: C
    L<n>      — inductor,        1 param: L
    W<n>      — finite Warburg,  1 param: Aw  (Z = Aw/sqrt(jω))
    WO<n>     — Warburg open,    2 params: Aw, τ
    WS<n>     — Warburg short,   2 params: Aw, τ
    CPE<n>    — CPE,             2 params: Q, n
    G<n>      — Gerischer,       2 params: Y0, k
    """

    _ELEM_PARAMS = {
        'R': 1, 'C': 1, 'L': 1, 'W': 1,
        'WO': 2, 'WS': 2, 'CPE': 2, 'G': 2,
    }

    def __init__(self, circuit_str: str):
        self.circuit_str = circuit_str.strip()
        self._elem_list = self._parse_elements(self.circuit_str)

    # ── public ────────────────────────────────────────────────────────────
    def n_params(self) -> int:
        return sum(self._ELEM_PARAMS.get(e, 1) for e in self._elem_list)

    def impedance(self, params: np.ndarray, freq: np.ndarray) -> np.ndarray:
        """Return complex impedance array Z(f) for given parameter vector."""
        omega = 2 * np.pi * freq
        p_iter = iter(params)
        return self._eval_expr(self.circuit_str, omega, p_iter)

    # ── parsing ───────────────────────────────────────────────────────────
    def _parse_elements(self, expr: str) -> List[str]:
        """Flat list of element types for parameter counting."""
        import re
        tokens = re.findall(r'CPE|WO|WS|[RCLWG]', expr.upper())
        return tokens

    # ── evaluation ────────────────────────────────────────────────────────
    def _eval_expr(self, expr: str, omega: np.ndarray, p_iter) -> np.ndarray:
        import re
        expr = expr.strip()
        # Handle parallel p(A,B,...)
        if expr.lower().startswith('p(') and expr.endswith(')'):
            inner = expr[2:-1]
            parts = self._split_top_level(inner)
            Z_inv = sum(1.0 / self._eval_expr(p, omega, p_iter) for p in parts)
            return 1.0 / Z_inv
        # Handle series A-B-C
        if '-' in expr:
            parts = self._split_series(expr)
            if len(parts) > 1:
                return sum(self._eval_expr(p, omega, p_iter) for p in parts)
        # Single element
        m = re.match(r'^(CPE|WO|WS|[RCLWG])', expr.upper())
        if not m:
            raise ValueError(f"Unknown circuit element: {expr!r}")
        etype = m.group(1)
        return self._elem_impedance(etype, omega, p_iter)

    def _split_series(self, expr: str) -> List[str]:
        """Split by '-' at depth 0 (not inside parentheses)."""
        parts, depth, buf = [], 0, []
        for ch in expr:
            if ch == '(':
                depth += 1; buf.append(ch)
            elif ch == ')':
                depth -= 1; buf.append(ch)
            elif ch == '-' and depth == 0:
                parts.append(''.join(buf)); buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append(''.join(buf))
        return parts

    def _split_top_level(self, expr: str) -> List[str]:
        """Split by ',' at depth 0."""
        parts, depth, buf = [], 0, []
        for ch in expr:
            if ch == '(':
                depth += 1; buf.append(ch)
            elif ch == ')':
                depth -= 1; buf.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(buf)); buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append(''.join(buf))
        return parts

    def _elem_impedance(self, etype: str, omega: np.ndarray, p_iter) -> np.ndarray:
        jw = 1j * omega
        if etype == 'R':
            R = next(p_iter)
            return np.full_like(omega, R, dtype=complex)
        elif etype == 'C':
            C = next(p_iter)
            return 1.0 / (jw * C)
        elif etype == 'L':
            L = next(p_iter)
            return jw * L
        elif etype == 'W':
            Aw = next(p_iter)
            return Aw / np.sqrt(jw)
        elif etype == 'WO':
            Aw, tau = next(p_iter), next(p_iter)
            return Aw * np.tanh(np.sqrt(jw * tau)) / np.sqrt(jw)
        elif etype == 'WS':
            Aw, tau = next(p_iter), next(p_iter)
            return Aw / (np.tanh(np.sqrt(jw * tau)) * np.sqrt(jw))
        elif etype == 'CPE':
            Q, n = next(p_iter), next(p_iter)
            return 1.0 / (Q * (jw ** n))
        elif etype == 'G':
            Y0, k = next(p_iter), next(p_iter)
            return 1.0 / (Y0 * np.sqrt(k + jw))
        else:
            raise ValueError(f"Unknown element: {etype}")


# ─────────────────────────────────────────────────────────────────────────────
# Cost function  (modulus-weighted CNLS — IUPAC recommendation)
# ─────────────────────────────────────────────────────────────────────────────

def _modulus_residuals(
    params: np.ndarray,
    evaluator: _CircuitEvaluator,
    freq: np.ndarray,
    Z_exp: np.ndarray,
) -> np.ndarray:
    """
    Modulus-weighted residuals for CNLS fitting (Boukamp 1995).
    Returned as [real_residuals, imag_residuals] stacked (shape 2N).
    Weights  w_i = 1 / |Z_exp_i|  so low-impedance and
    high-impedance points contribute equally.
    """
    Z_fit = evaluator.impedance(params, freq)
    mod = np.abs(Z_exp)
    mod = np.where(mod < 1e-30, 1e-30, mod)  # guard against division by zero
    res_r = (Z_fit.real - Z_exp.real) / mod
    res_i = (Z_fit.imag - Z_exp.imag) / mod
    return np.concatenate([res_r, res_i])


def _chi2_reduced(
    residuals: np.ndarray, n_params: int
) -> float:
    dof = max(len(residuals) - n_params, 1)
    return float(np.sum(residuals ** 2) / dof)


# ─────────────────────────────────────────────────────────────────────────────
# Main fitter class
# ─────────────────────────────────────────────────────────────────────────────

class WarmStartBatchFitter:
    """
    Warm-start sequential CNLS batch fitter.

    Parameters
    ----------
    circuit : str
        Circuit string, e.g. ``"R0-p(Rct,CPE1)"`` or
        ``"R0-p(Rct,CPE1)-WO1"``.
    bounds_low, bounds_high : list or array-like
        Lower and upper bounds for every parameter.
    param_names : list[str], optional
        Human-readable names for each parameter (used in results).
    method : str
        Scipy ``least_squares`` method: ``'trf'`` (default) or ``'dogbox'``.
    loss : str
        Scipy loss function: ``'linear'`` (default) or ``'soft_l1'``
        for robustness to outlier frequencies.
    max_nfev : int
        Maximum function evaluations per spectrum (default 2000).
    ftol, xtol, gtol : float
        Convergence tolerances (defaults: 1e-10 each).
    verbose : bool
        Print per-spectrum fit summary to stdout.
    """

    def __init__(
        self,
        circuit: str,
        bounds_low: list,
        bounds_high: list,
        param_names: Optional[List[str]] = None,
        method: str = 'trf',
        loss: str = 'linear',
        max_nfev: int = 2000,
        ftol: float = 1e-10,
        xtol: float = 1e-10,
        gtol: float = 1e-10,
        verbose: bool = False,
    ):
        self.circuit = circuit
        self.evaluator = _CircuitEvaluator(circuit)
        self.bounds = (np.asarray(bounds_low, dtype=float),
                       np.asarray(bounds_high, dtype=float))
        self.param_names = param_names
        self.method = method
        self.loss = loss
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.verbose = verbose

    # ── single-spectrum fit ───────────────────────────────────────────────

    def fit_single(
        self,
        dataset: BatchEISDataset,
        initial_params: np.ndarray,
    ) -> BatchEISResult:
        """Fit one EIS spectrum; returns a BatchEISResult."""
        freq = dataset.frequency
        Z_exp = dataset.Z_complex

        # Clip initial guess to bounds
        p0 = np.clip(
            np.asarray(initial_params, dtype=float),
            self.bounds[0] + 1e-30,
            self.bounds[1] - 1e-30,
        )

        try:
            sol = least_squares(
                _modulus_residuals,
                p0,
                bounds=self.bounds,
                args=(self.evaluator, freq, Z_exp),
                method=self.method,
                loss=self.loss,
                max_nfev=self.max_nfev,
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
            )
            success = sol.success or sol.cost < 1e-4

            # Parameter uncertainties from Jacobian covariance
            try:
                _, sv, Vt = np.linalg.svd(sol.jac, full_matrices=False)
                threshold = np.finfo(float).eps * max(sol.jac.shape) * sv[0]
                sv_inv = np.where(sv > threshold, 1.0 / sv, 0.0)
                cov = (Vt.T * sv_inv[np.newaxis, :]) @ Vt
                param_errors = np.sqrt(np.diag(cov) * sol.cost / max(len(sol.fun) - len(p0), 1))
            except Exception:
                param_errors = np.full_like(p0, np.nan)

            chi2 = _chi2_reduced(sol.fun, len(p0))
            params = sol.x
            msg = sol.message

        except Exception as exc:
            success = False
            params = None
            param_errors = None
            chi2 = np.inf
            msg = str(exc)
            sol = type('_', (), {'nfev': 0})()

        # Build named-param dicts
        named_params: Dict[str, float] = {}
        named_errors: Dict[str, float] = {}
        if params is not None and self.param_names:
            for i, name in enumerate(self.param_names):
                if i < len(params):
                    named_params[name] = float(params[i])
                    named_errors[name] = float(param_errors[i]) if param_errors is not None else np.nan

        result = BatchEISResult(
            label=dataset.label,
            params=params,
            param_errors=param_errors,
            chi2=chi2,
            circuit=self.circuit,
            success=success,
            n_iter=getattr(sol, 'nfev', 0),
            message=msg,
            named_params=named_params,
            named_errors=named_errors,
        )

        if self.verbose:
            status = '✓' if success else '✗'
            print(f"  [{status}] {dataset.label or '—':30s}  "
                  f"χ²={chi2:.4e}  nfev={result.n_iter}")
        return result

    # ── warm-start batch fit  ─────────────────────────────────────────────

    def fit(
        self,
        datasets: List[BatchEISDataset],
        initial_params: list,
    ) -> List[BatchEISResult]:
        """
        Fit all datasets sequentially with warm-starting.

        Parameters
        ----------
        datasets : list[BatchEISDataset]
            Ordered list of EIS spectra (e.g. low → high concentration).
        initial_params : list or array-like
            Starting parameters for the first spectrum.

        Returns
        -------
        list[BatchEISResult]
            One result per input dataset.
        """
        p0 = np.asarray(initial_params, dtype=float)
        results: List[BatchEISResult] = []

        if self.verbose:
            print(f"WarmStartBatchFitter — circuit: {self.circuit}")
            print(f"  {len(datasets)} spectra | warm-start enabled")
            print("-" * 60)

        for ds in datasets:
            result = self.fit_single(ds, p0)
            results.append(result)
            # Warm-start: use this result as the next initial guess
            # Only update p0 if the fit was successful
            if result.success and result.params is not None:
                p0 = result.params.copy()
            # else: keep the previous p0 (fall back to last good fit)

        if self.verbose:
            n_ok = sum(r.success for r in results)
            print("-" * 60)
            print(f"  Done: {n_ok}/{len(results)} fits converged.")

        return results

    # ── convenience: extract a parameter column ───────────────────────────

    def extract_param(
        self,
        results: List[BatchEISResult],
        param_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a single parameter across all results.

        Returns
        -------
        values : ndarray  — NaN for failed fits
        errors : ndarray  — NaN for failed fits or unavailable covariance
        """
        values = np.array([
            r.params[param_index] if (r.success and r.params is not None) else np.nan
            for r in results
        ])
        errors = np.array([
            r.param_errors[param_index]
            if (r.success and r.param_errors is not None) else np.nan
            for r in results
        ])
        return values, errors

    def extract_named_param(
        self,
        results: List[BatchEISResult],
        name: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Same as extract_param but using the parameter name string."""
        values = np.array([r.named_params.get(name, np.nan) for r in results])
        errors = np.array([r.named_errors.get(name, np.nan) for r in results])
        return values, errors
