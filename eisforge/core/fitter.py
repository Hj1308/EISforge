"""
CNLS Fitter — Direct scipy.optimize implementation (bypasses impedance.py NaN bug).
Author: Hoda Jafari | May 2026

Uses impedance.py ONLY for circuit model evaluation, not for optimization.
The optimization is done directly with scipy.optimize.least_squares (LM method).
This avoids the NaN parameters bug seen with impedance.py's internal optimizer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from eisforge.parsers.base_parser import EISDataset


@dataclass
class FitResult:
    circuit_string: str
    parameters: dict
    parameter_errors: dict
    chi_squared: float
    z_fit: Optional[np.ndarray]
    converged: bool
    n_outliers_removed: int = 0
    n_points_used: int = 0
    _circuit_obj: Optional[object] = field(default=None, repr=False)

    def parameter_table(self) -> str:
        lines = [f"{'Parameter':<16} {'Value':>16} {'±Error':>16}", "-"*50]
        for name, val in self.parameters.items():
            err = self.parameter_errors.get(name, float("nan"))
            lines.append(f"{name:<16} {val:>16.4e} {err:>16.2e}")
        lines.append(f"\nReduced χ² = {self.chi_squared:.6f}")
        return "\n".join(lines)


class CNLSFitter:
    """
    CNLS fitter using direct scipy.optimize.least_squares.

    Strategy:
        1. Use impedance.py CustomCircuit ONLY to evaluate Z(ω) for given params
        2. Run scipy.optimize.least_squares directly (Levenberg-Marquardt or TRF)
        3. Extract uncertainties from Jacobian covariance matrix
        4. This avoids impedance.py's internal optimizer which can return NaN
    """

    def __init__(
        self,
        circuit_string: str,
        initial_guess: list,
        bounds: Optional[tuple] = None,
        weight_by_modulus: bool = True,
        remove_outliers: bool = True,
        outlier_threshold: float = 3.0,
        neighbor_ratio: float = 5.0,
    ) -> None:
        self.circuit_string    = circuit_string
        self.initial_guess     = list(initial_guess)
        self.bounds            = bounds
        self.weight_by_modulus = weight_by_modulus
        self.remove_outliers   = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.neighbor_ratio    = neighbor_ratio

    # ── Outlier Detection ─────────────────────────────────────────────────────

    def _detect_outliers(self, freq: np.ndarray, Z: np.ndarray) -> np.ndarray:
        n = len(freq)
        if n < 10:
            return np.ones(n, dtype=bool)

        order_desc = np.argsort(freq)[::-1]
        log_m_sorted = np.log10(np.abs(Z[order_desc]).clip(min=1e-15))

        keep_sorted = np.ones(n, dtype=bool)
        log_ratio = np.log10(self.neighbor_ratio)

        for i in range(1, n - 1):
            d_prev = abs(log_m_sorted[i] - log_m_sorted[i-1])
            d_next = abs(log_m_sorted[i] - log_m_sorted[i+1])
            if d_prev > log_ratio and d_next > log_ratio:
                neighbor_avg = (log_m_sorted[i-1] + log_m_sorted[i+1]) / 2
                if abs(log_m_sorted[i] - neighbor_avg) > log_ratio:
                    keep_sorted[i] = False

        try:
            from scipy.signal import medfilt
            window = max(3, min(7, n // 5))
            if window % 2 == 0:
                window += 1
            trend = medfilt(log_m_sorted, kernel_size=window)
            residuals = log_m_sorted - trend
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            if mad > 1e-10:
                z_scores = np.abs(residuals - median_res) / mad
                keep_sorted = keep_sorted & (z_scores < self.outlier_threshold)
        except Exception:
            pass

        keep = np.ones(n, dtype=bool)
        for k, idx in enumerate(order_desc):
            keep[idx] = keep_sorted[k]
        return keep

    # ── Main Fit ──────────────────────────────────────────────────────────────

    def fit(self, dataset: EISDataset) -> FitResult:
        """Fit EIS data using direct scipy.optimize.least_squares."""
        freq   = dataset.frequency
        Z_meas = dataset.z_complex

        # Remove outliers
        if self.remove_outliers:
            keep      = self._detect_outliers(freq, Z_meas)
            n_removed = int(np.sum(~keep))
            if n_removed > 0:
                warnings.warn(f"Removed {n_removed} outlier point(s).", stacklevel=2)
            freq   = freq[keep]
            Z_meas = Z_meas[keep]
        else:
            n_removed = 0

        # Build circuit model for Z evaluation
        try:
            from impedance.models.circuits import CustomCircuit
            circuit = CustomCircuit(
                circuit=self.circuit_string,
                initial_guess=self.initial_guess,
            )
        except Exception as e:
            return self._failed_result(f"Invalid circuit: {e}", n_removed, len(freq))

        # Get parameter names
        try:
            pnames = circuit.get_param_names()
            if isinstance(pnames, tuple):
                pnames = pnames[0]
            if isinstance(pnames, str):
                pnames = list(pnames)
            param_names = list(pnames)
        except Exception:
            param_names = [f"p{i}" for i in range(len(self.initial_guess))]

        # ── Weights ─────────────────────────────────────────────────────────
        Z_mod = np.abs(Z_meas)
        Z_mod_safe = np.where(Z_mod > 1e-10, Z_mod, 1e-10)
        weights = 1.0 / Z_mod_safe if self.weight_by_modulus else np.ones(len(freq))

        # ── Define residual function for scipy ───────────────────────────────
        def residuals(params):
            """Returns weighted residuals [Re(ΔZ)·w, Im(ΔZ)·w]."""
            try:
                circuit.parameters_ = np.array(params)
                Z_pred = circuit.predict(freq)
                r_real = (Z_meas.real - Z_pred.real) * weights
                r_imag = (Z_meas.imag - Z_pred.imag) * weights
                return np.concatenate([r_real, r_imag])
            except Exception:
                return np.ones(2 * len(freq)) * 1e10

        # ── Run scipy.optimize.least_squares ─────────────────────────────────
        p0 = np.array(self.initial_guess, dtype=float)

        # Choose method based on bounds
        if self.bounds is not None:
            lb = np.array(self.bounds[0], dtype=float)
            ub = np.array(self.bounds[1], dtype=float)
            method = "trf"
        else:
            lb = 0.0
            ub = np.inf
            method = "trf"
            # Ensure p0 is strictly positive for TRF
            p0 = np.where(p0 > 0, p0, 1e-10)

        try:
            result = least_squares(
                residuals,
                x0=p0,
                bounds=(lb if self.bounds else (0.0, np.inf), ub if self.bounds else np.inf),
                method=method,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=10000,
            )

            # Check result validity
            fitted = result.x
            if not np.all(np.isfinite(fitted)):
                raise ValueError("Optimizer returned NaN/Inf parameters")

            converged = result.success or result.cost < 1e-3

        except Exception as e:
            warnings.warn(f"TRF failed ({e}), trying Levenberg-Marquardt...", stacklevel=2)
            try:
                result = least_squares(
                    residuals,
                    x0=p0,
                    method="lm",    # No bounds, most robust
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=5000,
                )
                fitted = result.x
                if not np.all(np.isfinite(fitted)):
                    raise ValueError("LM also returned NaN")
                converged = result.success
            except Exception as e2:
                warnings.warn(f"Both optimizers failed: {e2}. Using initial guess.", stacklevel=2)
                fitted    = p0
                converged = False
                result    = None

        # ── Build parameters dict ────────────────────────────────────────────
        n = min(len(param_names), len(fitted))
        parameters = {param_names[i]: float(fitted[i]) for i in range(n)}

        # ── Compute parameter errors from Jacobian ───────────────────────────
        errors = {name: float("nan") for name in param_names[:n]}
        if result is not None and hasattr(result, "jac"):
            try:
                J   = result.jac
                cov = np.linalg.pinv(J.T @ J)
                diag = np.diag(cov)
                std_devs = np.sqrt(np.abs(diag)) * np.sqrt(result.cost / max(len(result.fun) - len(fitted), 1))
                for i in range(min(n, len(std_devs))):
                    if np.isfinite(std_devs[i]):
                        errors[param_names[i]] = float(std_devs[i])
            except Exception:
                pass

        # ── Compute z_fit and chi² ───────────────────────────────────────────
        z_fit = None
        chi2  = float("inf")
        try:
            circuit.parameters_ = fitted
            z_fit = circuit.predict(freq)
            dof   = max(2 * len(freq) - len(fitted), 1)
            res_r = (Z_meas.real - z_fit.real) * weights
            res_i = (Z_meas.imag - z_fit.imag) * weights
            chi2  = float(np.sum(res_r**2 + res_i**2) / dof)
        except Exception:
            pass

        # Final convergence check
        all_finite = all(np.isfinite(v) for v in parameters.values())
        converged  = converged and np.isfinite(chi2) and all_finite

        return FitResult(
            circuit_string=self.circuit_string,
            parameters=parameters,
            parameter_errors=errors,
            chi_squared=chi2,
            z_fit=z_fit,
            converged=converged,
            n_outliers_removed=n_removed,
            n_points_used=len(freq),
            _circuit_obj=circuit,
        )

    def _failed_result(self, msg, n_removed, n_used) -> FitResult:
        warnings.warn(msg, stacklevel=3)
        params = {f"p{i}": v for i, v in enumerate(self.initial_guess)}
        return FitResult(
            circuit_string=self.circuit_string,
            parameters=params,
            parameter_errors={k: float("nan") for k in params},
            chi_squared=float("inf"),
            z_fit=None,
            converged=False,
            n_outliers_removed=n_removed,
            n_points_used=n_used,
        )
