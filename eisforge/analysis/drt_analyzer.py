"""
DRT Analyzer — Distribution of Relaxation Times (Tikhonov regularized).

Author: Hoda Jafari | July 2026

Physics:
    Z(ω) = R_inf + ∫ γ(ln τ) / (1 + jωτ) d(ln τ)

Discretized on a log-spaced τ grid with N_τ points.  R_inf is solved
jointly with γ(τ) as an explicit column of the design matrix (not
estimated separately from a single high-frequency data point).  R_inf
is excluded from the Tikhonov penalty so it is not biased toward zero.

References
----------
- Ciucci, F. & Chen, C. (2015). Analysis of EIS via DRT.
  Electrochimica Acta, 167, 439-454.
- Saccoccio, M. et al. (2014). Optimal regularization in DRT.
  Electrochimica Acta, 131, 60-70.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DRTResult:
    """Container for DRT analysis results."""

    tau: np.ndarray               # log-spaced time constants (s)
    gamma: np.ndarray             # DRT distribution γ(ln τ) (Ω)
    r_inf: float                  # recovered high-frequency resistance (Ω)
    lambda_opt: float             # optimal Tikhonov parameter
    residuals: np.ndarray         # complex residuals Z_meas - Z_fit
    rms_real: float               # RMS of Re(residuals)
    rms_imag: float               # RMS of Im(residuals)
    curvature: np.ndarray         # L-curve curvature values (for debugging)
    lambda_range: np.ndarray      # λ values tested (for debugging)
    peaks_tau: np.ndarray = field(default_factory=lambda: np.array([]))
    peaks_gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    n_freq: int = 0
    n_tau: int = 0

    @property
    def total_resistance(self) -> float:
        """R_pol = ∫ γ(ln τ) d(ln τ) ≈ Σ γ_k · Δ ln τ."""
        d_ln = np.log(self.tau[1] / self.tau[0]) if len(self.tau) > 1 else 1.0
        return float(np.sum(self.gamma) * d_ln)

    @property
    def r_ct_estimate(self) -> float:
        """Largest peak height as rough R_ct proxy."""
        if len(self.gamma) == 0:
            return 0.0
        return float(np.max(self.gamma))

    def summary(self) -> str:
        lines = [
            f"DRT Analysis -- {self.n_freq} frequencies, {self.n_tau} tau points",
            f"  R_inf          = {self.r_inf:.4f} Ohm",
            f"  Sum gamma(tau) = {self.total_resistance:.4f} Ohm",
            f"  lam_opt        = {self.lambda_opt:.3e}",
            f"  RMS Re(Z)      = {self.rms_real:.5f} Ohm",
            f"  RMS Im(Z)      = {self.rms_imag:.5f} Ohm",
            f"  Peaks detected : {len(self.peaks_tau)}",
        ]
        for i, (t, g) in enumerate(zip(self.peaks_tau, self.peaks_gamma)):
            lines.append(f"    peak {i+1}: tau={t:.3e} s  f={1/t:.1f} Hz  gamma={g:.3f}")
        return "\n".join(lines)


class DRTAnalyzer:
    """Tikhonov-regularized DRT deconvolution with explicit R_inf term."""

    def __init__(
        self,
        n_tau: int = 80,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e2,
        n_lambda: int = 60,
        regularization_order: int = 1,
        tau_extend_factor: float = 1.5,
    ) -> None:
        self.n_tau          = n_tau
        self.lambda_min     = lambda_min
        self.lambda_max     = lambda_max
        self.n_lambda       = n_lambda
        self.reg_order      = regularization_order
        self.tau_extend     = tau_extend_factor

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        freq: np.ndarray,
        z_real: np.ndarray,
        z_imag: np.ndarray,
    ) -> DRTResult:
        """Full DRT analysis: build kernels, L-curve, solve, find peaks."""
        freq   = np.asarray(freq, dtype=float)
        z_real = np.asarray(z_real, dtype=float)
        z_imag = np.asarray(z_imag, dtype=float)

        self._validate_input(freq, z_real, z_imag)

        # Build τ grid (extended beyond measured frequency range)
        tau_vec = self._build_tau_grid(freq)

        # Build design matrix A (with R_inf column)
        A = self._build_kernels(freq, tau_vec)

        # Build data vector b = [Z_re; Z_im]
        b = np.concatenate([z_real, z_imag])

        # L-curve to choose optimal λ
        lambda_opt, lambdas, curvatures = self._l_curve(A, b, tau_vec)

        # Solve at optimal λ
        x = self._solve(A, b, tau_vec, lambda_opt)

        # Extract R_inf from first element, gamma from rest
        r_inf  = float(x[0])
        gamma  = x[1:]

        # Compute fit and residuals
        z_fit       = A @ x
        z_fit_re    = z_fit[:len(freq)]
        z_fit_im    = z_fit[len(freq):]
        res_re      = z_real - z_fit_re
        res_im      = z_imag - z_fit_im
        rms_real    = float(np.sqrt(np.mean(res_re ** 2)))
        rms_imag    = float(np.sqrt(np.mean(res_im ** 2)))

        # Find peaks in γ(ln τ)
        peaks_tau, peaks_gamma = self._find_peaks(tau_vec, gamma)

        return DRTResult(
            tau          = tau_vec,
            gamma        = gamma,
            r_inf        = r_inf,
            lambda_opt   = lambda_opt,
            residuals    = res_re + 1j * res_im,
            rms_real     = rms_real,
            rms_imag     = rms_imag,
            curvature    = curvatures,
            lambda_range = lambdas,
            peaks_tau    = peaks_tau,
            peaks_gamma  = peaks_gamma,
            n_freq       = len(freq),
            n_tau        = len(tau_vec),
        )

    # ── Internal: τ grid ──────────────────────────────────────────────────────

    def _build_tau_grid(self, freq: np.ndarray) -> np.ndarray:
        """Log-spaced τ grid extending beyond the measured frequency range."""
        f_min = freq.min()
        f_max = freq.max()
        tau_min = 1.0 / (2.0 * np.pi * f_max * self.tau_extend)
        tau_max = self.tau_extend / (2.0 * np.pi * f_min)
        if tau_min <= 0 or tau_max <= 0:
            raise ValueError(f"Invalid τ range: [{tau_min:.2e}, {tau_max:.2e}]")
        return np.logspace(np.log10(tau_min), np.log10(tau_max), self.n_tau)

    # ── Internal: kernel / design matrix ──────────────────────────────────────

    def _build_kernels(
        self, freq: np.ndarray, tau_vec: np.ndarray
    ) -> np.ndarray:
        """Build design matrix A with explicit R_inf column.

        The first column (index 0) is the R_inf term: ones for Z_re rows,
        zeros for Z_im rows.  Columns 1..N_tau are the DRT kernel:

            Z_re(omega) = R_inf + SUM gamma_k * K_re(omega, tau_k)
            z_imag_pos  = SUM gamma_k * K_im(omega, tau_k)

        where z_imag_pos is the positive-convention stored value
        (i.e. Z_im_stored = -Im(Z_actual) for capacitive data).

            K_re = 1 / (1 + omega^2 tau^2)
            K_im = omega * tau / (1 + omega^2 tau^2)    (positive, matching stored convention)
        """
        n_freq  = len(freq)
        n_tau   = len(tau_vec)
        omega   = 2.0 * np.pi * freq
        n_total = n_tau + 1  # R_inf + tau elements

        # Real part block
        A_re = np.zeros((n_freq, n_total))
        A_re[:, 0] = 1.0                     # R_inf column
        for k, tau in enumerate(tau_vec):
            denom = 1.0 + (omega * tau) ** 2
            A_re[:, k + 1] = 1.0 / denom

        # Imaginary part block
        A_im = np.zeros((n_freq, n_total))
        A_im[:, 0] = 0.0                     # no R_inf contribution
        for k, tau in enumerate(tau_vec):
            denom = 1.0 + (omega * tau) ** 2
            A_im[:, k + 1] = omega * tau / denom

        return np.vstack([A_re, A_im])

    # ── Internal: Regularization matrix ───────────────────────────────────────

    def _build_regularization(self, n_tau: int) -> np.ndarray:
        """Build Tikhonov regularization matrix L.

        Order 1: first difference (penalizes large γ values).
        Order 2: second difference (penalizes rapid γ variations — smoother).

        R_inf (column 0) is excluded from regularization (L[:, 0] = 0).
        """
        n_total = n_tau + 1
        if self.reg_order == 1:
            # First difference for gamma terms only
            L_gamma = np.zeros((n_tau - 1, n_tau))
            for i in range(n_tau - 1):
                L_gamma[i, i]     = -1.0
                L_gamma[i, i + 1] =  1.0
        else:
            # Second difference (smoother)
            L_gamma = np.zeros((n_tau - 2, n_tau))
            for i in range(n_tau - 2):
                L_gamma[i, i]     =  1.0
                L_gamma[i, i + 1] = -2.0
                L_gamma[i, i + 2] =  1.0

        L = np.zeros((L_gamma.shape[0], n_total))
        L[:, 1:] = L_gamma   # skip column 0 (R_inf)
        return L

    # ── Internal: solve ───────────────────────────────────────────────────────

    def _solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tau_vec: np.ndarray,
        lambda_val: float,
    ) -> np.ndarray:
        """Solve regularized NNLS: min ||A x - b||^2 + lam^2 ||L x||^2,
        subject to x >= 0 (R_inf and gamma all non-negative).

        Implemented as augmented least squares with bounds:
            min ||[A; lam*L] x - [b; 0]||^2  s.t. lb <= x <= ub
        """
        from scipy.optimize import lsq_linear

        L = self._build_regularization(len(tau_vec))
        A_aug = np.vstack([A, lambda_val * L])
        b_aug = np.concatenate([b, np.zeros(L.shape[0])])

        n_total = A.shape[1]
        lb = np.zeros(n_total)
        ub = np.full(n_total, np.inf)

        result = lsq_linear(
            A_aug, b_aug, bounds=(lb, ub),
            method="trf",
        )
        return result.x

    # ── Internal: L-curve ─────────────────────────────────────────────────────

    def _l_curve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tau_vec: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute L-curve and find optimal λ via maximum curvature."""
        lambdas   = np.logspace(
            np.log10(self.lambda_min), np.log10(self.lambda_max), self.n_lambda
        )
        res_norms = np.zeros(self.n_lambda)
        reg_norms = np.zeros(self.n_lambda)

        L = self._build_regularization(len(tau_vec))

        for i, lam in enumerate(lambdas):
            x      = self._solve(A, b, tau_vec, lam)
            res    = A @ x - b
            res_norms[i] = np.log(float(np.sum(res ** 2)))
            reg_norms[i] = np.log(float(np.sum((L @ x) ** 2)))

        # Find corner via maximum curvature
        curvature = self._compute_curvature(res_norms, reg_norms)
        idx_opt   = int(np.argmax(curvature))
        lambda_opt = float(lambdas[idx_opt])

        return lambda_opt, lambdas, curvature

    @staticmethod
    def _compute_curvature(rho: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """L-curve curvature κ(λ) = (ρ' η'' - ρ'' η') / (ρ'² + η'²)^(3/2)."""
        # Smooth first
        drho      = np.gradient(rho)
        deta      = np.gradient(eta)
        d2rho     = np.gradient(drho)
        d2eta     = np.gradient(deta)
        numer     = drho * d2eta - d2rho * deta
        denom     = (drho ** 2 + deta ** 2) ** 1.5
        curvature = np.where(denom > 1e-30, numer / denom, 0.0)
        return curvature

    # ── Internal: peak finding ────────────────────────────────────────────────

    @staticmethod
    def _find_peaks(
        tau_vec: np.ndarray, gamma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find local maxima in γ(ln τ) via simple gradient sign change.

        Returns sorted by peak height (highest first).
        """
        if len(gamma) < 3:
            return np.array([]), np.array([])

        peaks_tau   = []
        peaks_gamma = []

        for i in range(1, len(gamma) - 1):
            if gamma[i] > gamma[i - 1] and gamma[i] > gamma[i + 1]:
                if gamma[i] > gamma.max() * 0.02:  # ignore noise peaks < 2% of max
                    peaks_tau.append(tau_vec[i])
                    peaks_gamma.append(gamma[i])

        if not peaks_tau:
            return np.array([]), np.array([])

        order = np.argsort(peaks_gamma)[::-1]
        return np.array(peaks_tau)[order], np.array(peaks_gamma)[order]

    # ── Internal: validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_input(freq, z_real, z_imag):
        n = len(freq)
        if n < 10:
            raise ValueError(f"Need ≥10 frequency points, got {n}.")
        if len(z_real) != n or len(z_imag) != n:
            raise ValueError("freq, z_real, z_imag must have same length.")
        if not np.all(np.isfinite(z_real)) or not np.all(np.isfinite(z_imag)):
            raise ValueError("z_real / z_imag contain NaN or Inf.")
        if not np.all(freq > 0):
            raise ValueError("All frequencies must be positive.")
