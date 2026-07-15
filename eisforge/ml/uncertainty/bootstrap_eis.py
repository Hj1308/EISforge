"""
Bootstrap EIS Uncertainty — percentile CI + correlation matrix.

Author: Hoda Jafari | July 2026

Noise-injection bootstrap: fits the circuit once via TRF least_squares
to get best-fit parameters and residuals, then resamples residuals with
added Gaussian noise, refits each bootstrap sample with TRF (matching
fitter.py's CNLSFitter optimizer), and computes percentile confidence
intervals and a parameter correlation matrix from the bootstrap ensemble.

Implements the standard residual bootstrap for EIS:
  1. Best-fit → Z_fit, residuals δ = Z_meas - Z_fit
  2. Resample residuals with replacement + Gaussian noise σ ~ std(δ)
  3. Z_boot = Z_fit + δ_resampled + ε(σ)
  4. Refit Z_boot → parameters[k]
  5. CI = percentiles of bootstrap parameter distributions
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BootstrapResult:
    """Container for bootstrap uncertainty analysis."""

    n_bootstrap: int
    n_converged: int
    best_parameters: dict[str, float]
    ci_50: dict[str, tuple[float, float]]   # 50% CI (25th, 75th percentiles)
    ci_90: dict[str, tuple[float, float]]   # 90% CI (5th, 95th percentiles)
    ci_95: dict[str, tuple[float, float]]   # 95% CI (2.5th, 97.5th percentiles)
    correlation_matrix: dict[str, dict[str, float]]
    samples: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _failed: int = 0
    _warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Bootstrap EIS — {self.n_converged}/{self.n_bootstrap} converged",
            f"  param       best       50% CI              90% CI",
            f"  {'-' * 60}",
        ]
        for name in self.best_parameters:
            best = self.best_parameters[name]
            lo50, hi50 = self.ci_50.get(name, (np.nan, np.nan))
            lo90, hi90 = self.ci_90.get(name, (np.nan, np.nan))
            lines.append(
                f"  {name:<12} {best:>8.4f}  [{lo50:>8.4f}, {hi50:>8.4f}]  "
                f"[{lo90:>8.4f}, {hi90:>8.4f}]"
            )
        return "\n".join(lines)


def bootstrap_eis_uncertainty(
    freq: np.ndarray,
    z_real: np.ndarray,
    z_imag: np.ndarray,
    circuit_string: str,
    initial_guess: list[float],
    n_bootstrap: int = 1000,
    noise_scale: float = 1.0,
    bounds: Optional[list[tuple[float, float]]] = None,
    weight_by_modulus: bool = True,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap uncertainty for EIS circuit fitting.

    Parameters
    ----------
    freq : np.ndarray
        Frequency points (Hz), descending.
    z_real, z_imag : np.ndarray
        Measured impedance (Ω).
    circuit_string : str
        Circuit description string (e.g. "R0-p(R1,CPE1)").
    initial_guess : list[float]
        Initial parameter values for the fit.
    n_bootstrap : int
        Number of bootstrap iterations.
    noise_scale : float
        Multiplier for injected noise σ (1.0 = residual std dev).
    bounds : list[tuple[float, float]] or None
        Per-parameter (min, max) bounds for Nelder-Mead refit.
        If None, fits are unbounded and a warning is emitted.
    weight_by_modulus : bool
        If True, use 1/|Z| weighting in the cost function.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    rng = np.random.default_rng(seed)
    freq = np.asarray(freq, dtype=float)
    z_real = np.asarray(z_real, dtype=float)
    z_imag = np.asarray(z_imag, dtype=float)
    z_meas = z_real + 1j * z_imag

    n_data = len(freq)
    n_params = len(initial_guess)

    # ── Warn if no bounds ─────────────────────────────────────────────────
    if bounds is None:
        warnings.warn(
            "bootstrap_eis_uncertainty: bounds=None — bootstrap refits are "
            "unbounded. CIs may include unphysical values (e.g. negative "
            "resistance). Pass bounds=list[tuple] to constrain parameter ranges.",
            stacklevel=2,
        )

    # ── Initial fit ───────────────────────────────────────────────────────
    try:
        from impedance.models.circuits import CustomCircuit
    except ImportError:
        raise ImportError(
            "bootstrap_eis_uncertainty requires the 'impedance' package "
            "(pip install impedance)"
        )

    best_params, z_fit, residuals, pnames = _initial_fit(
        freq, z_meas, circuit_string, initial_guess, weight_by_modulus, bounds
    )

    # ── Bounds sanity check ────────────────────────────────────────────────
    if bounds is not None:
        for j, name in enumerate(pnames):
            lo, hi = bounds[j]
            val = float(best_params[j])
            if val < lo or val > hi:
                raise ValueError(
                    f"bootstrap_eis_uncertainty: best-fit parameter '{name}'="
                    f"{val:.6f} is outside its specified bound ({lo}, {hi}). "
                    f"Widen the bounds or fix the initial guess / circuit model."
                )

    # Residual noise level (per frequency, as complex)
    res_std_real = float(np.std(residuals.real))
    res_std_imag = float(np.std(residuals.imag))
    noise_sigma = max(res_std_real, res_std_imag) * noise_scale

    if noise_sigma < 1e-15:
        warnings.warn(
            "Residuals are near-zero — noise_scale artificially inflated.",
            stacklevel=2,
        )
        noise_sigma = 1e-6

    # ── Bootstrap loop ────────────────────────────────────────────────────
    samples = {name: [] for name in pnames}
    n_failed = 0

    for i in range(n_bootstrap):
        # Resample residuals (with replacement) + Gaussian noise
        idx        = rng.integers(0, n_data, size=n_data)
        delta_boot = (
            residuals[idx]
            + noise_sigma * (rng.standard_normal(n_data) + 1j * rng.standard_normal(n_data))
        )

        # Construct bootstrap dataset
        z_boot = z_fit + delta_boot
        Z_mod  = np.abs(z_boot)
        Z_mod_safe = np.where(Z_mod > 1e-10, Z_mod, 1e-10)
        w = 1.0 / Z_mod_safe if weight_by_modulus else np.ones(n_data)

        # TRF refit (matching fitter.py — bounds handled natively)
        try:
            x_opt = _trf_refit(
                circuit_string, freq, z_boot, w,
                x0=best_params, bounds=bounds,
            )
            if x_opt is not None:
                for j, name in enumerate(pnames):
                    samples[name].append(float(x_opt[j]))
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    # ── Build results ─────────────────────────────────────────────────────
    best_dict = {pnames[i]: float(best_params[i]) for i in range(n_params)}

    ci_50 = {}
    ci_90 = {}
    ci_95 = {}
    for name in pnames:
        arr = np.array(samples[name])
        if len(arr) < 3:
            ci_50[name] = (np.nan, np.nan)
            ci_90[name] = (np.nan, np.nan)
            ci_95[name] = (np.nan, np.nan)
        else:
            ci_50[name] = (float(np.percentile(arr, 25)), float(np.percentile(arr, 75)))
            ci_90[name] = (float(np.percentile(arr, 5)),  float(np.percentile(arr, 95)))
            ci_95[name] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    # Correlation matrix from bootstrap samples
    corr_mat = {}
    for name_i in pnames:
        corr_mat[name_i] = {}
        arr_i = np.array(samples[name_i])
        for name_j in pnames:
            arr_j = np.array(samples[name_j])
            if len(arr_i) > 2 and len(arr_j) > 2:
                c = np.corrcoef(arr_i, arr_j)[0, 1]
                corr_mat[name_i][name_j] = float(c) if np.isfinite(c) else 0.0
            else:
                corr_mat[name_i][name_j] = 0.0

    return BootstrapResult(
        n_bootstrap=n_bootstrap,
        n_converged=n_bootstrap - n_failed,
        best_parameters=best_dict,
        ci_50=ci_50,
        ci_90=ci_90,
        ci_95=ci_95,
        correlation_matrix=corr_mat,
        samples={name: np.array(v) for name, v in samples.items()},
        _failed=n_failed,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _initial_fit(
    freq: np.ndarray,
    z_meas: np.ndarray,
    circuit_string: str,
    initial_guess: list[float],
    weight_by_modulus: bool,
    bounds: Optional[list[tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Fit circuit via TRF least_squares (matching fitter.py CNLSFitter).

    Uses user-provided bounds if given; defaults to (0, inf) for
    physically constrained parameters.
    """
    from impedance.models.circuits import CustomCircuit
    from scipy.optimize import least_squares

    circuit = CustomCircuit(initial_guess=initial_guess, circuit=circuit_string)

    # Get parameter names
    pnames = circuit.get_param_names()
    if isinstance(pnames, tuple):
        pnames = pnames[0]
    if isinstance(pnames, str):
        pnames = list(pnames)
    pnames = list(pnames)

    Z_mod = np.abs(z_meas)
    Z_mod_safe = np.where(Z_mod > 1e-10, Z_mod, 1e-10)
    w = 1.0 / Z_mod_safe if weight_by_modulus else np.ones(len(freq))

    def residuals(params):
        circuit.parameters_ = np.array(params)
        Z_pred = circuit.predict(freq)
        return np.concatenate([
            (z_meas.real - Z_pred.real) * w,
            (z_meas.imag - Z_pred.imag) * w,
        ])

    p0 = np.array(initial_guess, dtype=float)

    if bounds is not None:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
    else:
        lb = 0.0
        ub = np.inf

    result = least_squares(
        residuals, x0=p0,
        bounds=(lb, ub), method="trf",
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10000,
    )

    best_params = result.x
    circuit.parameters_ = best_params
    z_fit    = circuit.predict(freq)
    residuals = z_meas - z_fit

    return best_params, z_fit, residuals, pnames


def _trf_refit(
    circuit_string: str,
    freq: np.ndarray,
    z_boot: np.ndarray,
    weights: np.ndarray,
    x0: np.ndarray,
    bounds: Optional[list[tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """TRF least-squares refit of a single bootstrap sample.

    Uses the same optimizer (scipy.least_squares, method='trf') as
    the main CNLSFitter in fitter.py, ensuring the bootstrap refits
    are consistent with the original fit and bounds are handled natively.
    """
    from impedance.models.circuits import CustomCircuit
    from scipy.optimize import least_squares

    circuit = CustomCircuit(initial_guess=x0, circuit=circuit_string)

    def residuals(params):
        circuit.parameters_ = np.array(params)
        Z_pred = circuit.predict(freq)
        return np.concatenate([
            (z_boot.real - Z_pred.real) * weights,
            (z_boot.imag - Z_pred.imag) * weights,
        ])

    if bounds is not None:
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)
    else:
        lb = 0.0
        ub = np.inf

    try:
        result = least_squares(
            residuals, x0=x0,
            bounds=(lb, ub), method="trf",
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=5000,
        )
        if np.all(np.isfinite(result.x)):
            return result.x
    except Exception:
        pass
    return None
