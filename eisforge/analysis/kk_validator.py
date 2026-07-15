"""
Kramers-Kronig Validator — standalone Lin-KK implementation.

Author: Hoda Jafari | July 2026

Based on Schonleber et al. (2014) "A consistent procedure for
Kramers-Kronig testing of impedance spectra" (Electrochimica Acta).

Implements a Voigt-circuit (RC elements in series) with explicit R_omega
and inductance L terms in the design matrix, matching the Schonleber
formulation.  Uses a simplified mu-criterion proxy (fraction of negative
R_k values) rather than the exact weighted Schonleber formula — see
docstring on _select_n_rc for details.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LinKKResult:
    """Container for Lin-KK validation results."""

    passed: bool
    residuals_real: np.ndarray
    residuals_imag: np.ndarray
    max_residual: float
    rmse: float
    n_rc: int
    mu: float
    r_omega: float                # recovered series resistance (Ω)
    inductance: float             # recovered inductance (H), may be 0.0
    z_fit: Optional[np.ndarray] = None
    warning_message: Optional[str] = None

    @property
    def max_residual_pct(self) -> float:
        return self.max_residual * 100.0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Lin-KK: {status}",
            f"  R_ohm     = {self.r_omega:.4f} Ohm",
            f"  L         = {self.inductance:.3e} H",
            f"  N_RC      = {self.n_rc}",
            f"  mu        = {self.mu:.3f}",
            f"  RMSE      = {self.rmse:.4f}",
            f"  max res   = {self.max_residual_pct:.3f} %",
        ]
        if self.warning_message:
            lines.append(f"  WARNING  : {self.warning_message}")
        return "\n".join(lines)


class StandaloneKKValidator:
    """Standalone Lin-KK validator with explicit R_Ω and L terms.

    Does NOT depend on impedance.validation.linKK — this is a
    self-contained implementation based on Schonleber et al. (2014).
    """

    def __init__(
        self,
        residual_threshold: float = 0.01,
        mu: float = 0.85,
        c: float = 0.5,
        include_inductance: bool = False,
    ) -> None:
        self.residual_threshold = residual_threshold
        self.mu                 = mu
        self.c                  = c
        self.include_inductance = include_inductance

    def validate(
        self, freq: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray
    ) -> LinKKResult:
        """Run full Lin-KK validation on EIS data."""
        freq   = np.asarray(freq, dtype=float)
        z_real = np.asarray(z_real, dtype=float)
        z_imag = np.asarray(z_imag, dtype=float)

        n = len(freq)
        if n < 10:
            return LinKKResult(
                passed=False, residuals_real=np.zeros(n),
                residuals_imag=np.zeros(n), max_residual=float("inf"),
                rmse=float("inf"), n_rc=0, mu=0.0,
                r_omega=0.0, inductance=0.0,
                warning_message=f"Too few points ({n}), need >=10.",
            )

        # Adaptive search for optimal n_rc
        n_rc, mu_val, x_final, A_final = self._select_n_rc(freq, z_real, z_imag)

        # Extract fitted values
        r_omega     = float(x_final[0])
        idx_l       = 1 if self.include_inductance else None
        inductance  = float(x_final[1]) if self.include_inductance else 0.0
        r_vals      = x_final[1 + int(self.include_inductance):]

        # Compute fit and residuals (modulus-normalized)
        z_fit_vec = A_final @ x_final
        z_fit_re  = z_fit_vec[:n]
        z_fit_im  = z_fit_vec[n:]   # positive convention, matches stored z_imag

        Z_mod     = np.abs(z_real + 1j * z_imag)
        Z_mod_safe = np.where(Z_mod > 1e-10, Z_mod, 1e-10)
        res_real  = (z_real - z_fit_re) / Z_mod_safe
        res_imag  = (z_imag - z_fit_im) / Z_mod_safe

        max_res = float(np.max(np.abs(np.concatenate([res_real, res_imag]))))
        rmse    = float(np.sqrt(np.mean(np.concatenate([res_real, res_imag]) ** 2)))

        # Mu criterion (simplified proxy — fraction of non-negative R_k)
        mu_val = self._compute_mu(r_vals)

        # Verdict
        passed = max_res <= self.residual_threshold and mu_val >= self.mu

        wmsg = None
        if not passed:
            parts = []
            if max_res > self.residual_threshold:
                parts.append(f"max residual {max_res*100:.3f}% > threshold {self.residual_threshold*100:.2f}%")
            if mu_val < self.mu:
                parts.append(f"μ {mu_val:.3f} < threshold {self.mu:.2f}")
            wmsg = "; ".join(parts) if parts else None

        return LinKKResult(
            passed=passed,
            residuals_real=res_real,
            residuals_imag=res_imag,
            max_residual=max_res,
            rmse=rmse,
            n_rc=n_rc,
            mu=mu_val,
            r_omega=r_omega,
            inductance=inductance,
            z_fit=z_fit_re + 1j * z_fit_im,
            warning_message=wmsg,
        )

    # ── τ grid ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_tau_grid(freq: np.ndarray, n_rc: int) -> np.ndarray:
        """Log-spaced τ grid covering the measured frequency range."""
        f_min = freq.min()
        f_max = freq.max()
        tau_min = 1.0 / (2.0 * np.pi * f_max)
        tau_max = 1.0 / (2.0 * np.pi * f_min)
        return np.logspace(np.log10(tau_min), np.log10(tau_max), n_rc)

    # ── μ-criterion ───────────────────────────────────────────────────────────

    def _select_n_rc(
        self, freq: np.ndarray, z_real: np.ndarray, z_imag: np.ndarray
    ) -> tuple[int, float, np.ndarray, np.ndarray]:
        """Select number of RC elements by scanning from the standard
        heuristic (decades * 5, capped at n//2) downward, stopping when
        the reduction in max_res per added RC saturates.

        The search starts at the standard heuristic (enough RC to resolve
        the data), then steps downward.  Under-fitting with too few RC
        inflates mu to 1.0 but produces huge max_res — we avoid this
        regime.  Over-fitting with too many RC drives mu artificially low
        and increases noise in max_res — we stop before that.

        Returns (n_rc, mu, solution, design_matrix).
        """
        n       = len(freq)
        decades = max(np.log10(freq.max() / freq.min()), 1.0)
        n_start = min(int(decades * 5), n // 2, 40)
        n_min   = max(3, int(decades))

        best_n_rc = n_start
        best_mu   = 0.0
        best_x    = None
        best_A    = None
        best_mr_pct = 1e10

        # Modulus weighting: 1/|Z| for each data point, repeated for real/imag rows
        Z_mod = np.abs(z_real + 1j * z_imag)
        Z_safe = np.where(Z_mod > 1e-10, Z_mod, 1e-10)
        w_data = 1.0 / Z_safe
        weights = np.concatenate([w_data, w_data])  # real + imag block

        for n_rc in range(n_start, n_min - 1, -1):
            tau_vec = self._build_tau_grid(freq, n_rc)
            A, _    = self._build_kk_matrix(freq, tau_vec)
            b       = np.concatenate([z_real, z_imag])
            x, _, _ = self._solve_lsq(A, b, self.include_inductance, weights)
            r_vals  = x[1 + int(self.include_inductance):]
            mu      = self._compute_mu(r_vals)

            z_fit   = A @ x
            max_res = float(np.max(np.abs(np.concatenate([
                (z_real - z_fit[:n]) / Z_safe,
                (z_imag - z_fit[n:]) / Z_safe,
            ]))))

            mr_pct = max_res * 100
            score  = mr_pct + (1.0 - mu) * 50  # joint: penalise low mu
            if score < best_mr_pct + (1.0 - best_mu) * 50:
                best_mu   = mu
                best_n_rc = n_rc
                best_x    = x
                best_A    = A
                best_mr_pct = mr_pct

        return best_n_rc, best_mu, best_x, best_A

    @staticmethod
    def _compute_mu(r_vals: np.ndarray) -> float:
        """Simplified µ-criterion: fraction of R_k values that are non-negative.

        NOTE: This is a simplified proxy, not Schonleber's exact weighted
        formula.  The original µ-criterion weights each RC element by its
        contribution to the total fit and evaluates a more complex sign
        consistency metric.  This proxy (fraction of R_k ≥ 0) is easier
        to compute and correlates well with the exact criterion in
        practice, but should NOT be cited as "the Schonleber µ-criterion"
        without noting this approximation.
        """
        if len(r_vals) == 0:
            return 0.0
        return float(np.sum(r_vals >= 0) / len(r_vals))

    # ── Design matrix ─────────────────────────────────────────────────────────

    def _build_kk_matrix(
        self, freq: np.ndarray, tau_vec: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Build Lin-KK design matrix with explicit R_ohm and optional L terms.

        Model: Z(omega) = R_ohm + jwL + SUM R_k / (1 + jw tau_k)

        This module uses the positive stored convention for imaginary
        impedance (z_imag = -Im(Z_actual) for capacitive data), matching
        the EISDataset storage format.  The RHS vector b = [z_real, z_imag]
        is always non-negative for well-behaved capacitive data.

        Re(Z)     = R_ohm             + SUM R_k / (1 + w^2 tau_k^2)
        z_imag    = -w L              + SUM w tau_k R_k / (1 + w^2 tau_k^2)
        (note: L column uses -omega so positive L gives correct negative
        contribution to the stored positive convention)

        Design matrix columns: [R_ohm, L, R_1, ..., R_M]
        Rows: [Re(Z) block; z_imag block]"""
        n_freq  = len(freq)
        n_tau   = len(tau_vec)
        omega   = 2.0 * np.pi * freq

        n_extra = 1 + int(self.include_inductance)  # R_Ω + optional L
        n_total = n_extra + n_tau

        # Real part block: Z_re
        A_re = np.zeros((n_freq, n_total))
        A_re[:, 0] = 1.0                        # R_Ω
        if self.include_inductance:
            A_re[:, 1] = 0.0                    # L has no real part
        for k, tau in enumerate(tau_vec):
            denom = 1.0 + (omega * tau) ** 2
            A_re[:, n_extra + k] = 1.0 / denom

        # Imaginary part block: -Z_im (= positive when impedance is capacitive)
        A_im = np.zeros((n_freq, n_total))
        A_im[:, 0] = 0.0                        # R_Ω has no imag part
        if self.include_inductance:
            A_im[:, 1] = -omega                 # -w L (negative contribution to stored z_imag)
        for k, tau in enumerate(tau_vec):
            denom = 1.0 + (omega * tau) ** 2
            A_im[:, n_extra + k] = omega * tau / denom

        return np.vstack([A_re, A_im]), n_total

    # ── Least squares solver ──────────────────────────────────────────────────

    @staticmethod
    def _solve_lsq(
        A: np.ndarray, b: np.ndarray, include_inductance: bool = False,
        weights: np.ndarray = None,
    ) -> tuple[np.ndarray, float, int]:
        """Solve Ax = b via modulus-weighted unconstrained least squares.

        If weights is provided (1/|Z| for each frequency point, repeated
        for real and imag rows), both A and b are multiplied element-wise
        so the fit penalises relative rather than absolute errors.  This
        prevents high-|Z| points from dominating the fit when impedance
        spans >2 decades.
        """
        if weights is not None:
            A = A * weights[:, np.newaxis]
            b = b * weights
        x, residuals_arr, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        residuals_sq = float(residuals_arr[0]) if len(residuals_arr) > 0 else float(np.sum((A @ x - b) ** 2))
        return x, residuals_sq, rank
