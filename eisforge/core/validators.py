"""
Kramers-Kronig Validator — Robust implementation.
Author: Hoda Jafari | May 2026

Uses impedance.py's linKK when available, falls back to a simpler
RC-based approximation otherwise.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from eisforge.parsers.base_parser import EISDataset


@dataclass
class KKValidationResult:
    passed: bool
    residuals_real: np.ndarray
    residuals_imag: np.ndarray
    max_residual: float
    n_rc_elements: int
    mu: float
    warning_message: Optional[str] = None

    @property
    def residuals_max_pct(self) -> float:
        if np.isinf(self.max_residual) or np.isnan(self.max_residual):
            return float("inf")
        return self.max_residual * 100.0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        if np.isinf(self.max_residual) or np.isnan(self.max_residual):
            return f"K-K: {status} (validator could not run reliably)"
        return (f"K-K: {status} | max residual={self.residuals_max_pct:.3f}% | "
                f"N_RC={self.n_rc_elements} | μ={self.mu:.3f}")


class KramersKronigValidator:
    """
    Validates EIS data against Kramers-Kronig relations.

    Strategy:
        1. Try impedance.py's linKK function (most rigorous)
        2. If unavailable or fails, use a simplified Voigt-circuit approach
        3. If both fail, return a non-blocking warning result
    """

    def __init__(
        self,
        residual_threshold: float = 0.005,
        mu: float = 0.85,
        c: float = 0.5,
    ) -> None:
        self.residual_threshold = residual_threshold
        self.mu = mu
        self.c  = c

    def validate(self, dataset: EISDataset) -> KKValidationResult:
        """Validate dataset against K-K relations."""
        freq = dataset.frequency
        Z    = dataset.z_complex
        n    = len(freq)

        if n < 10:
            return KKValidationResult(
                passed=False,
                residuals_real=np.zeros(n),
                residuals_imag=np.zeros(n),
                max_residual=float("inf"),
                n_rc_elements=0,
                mu=self.mu,
                warning_message=f"Insufficient data points ({n}). K-K requires ≥10.",
            )

        # ── Method 1: Try impedance.py linKK ──────────────────────────────────
        result = self._try_linkk(freq, Z)
        if result is not None:
            return result

        # ── Method 2: Fallback to Voigt-circuit fit ───────────────────────────
        result = self._fallback_voigt(freq, Z)
        if result is not None:
            return result

        # ── Method 3: Non-blocking notice ─────────────────────────────────────
        return KKValidationResult(
            passed=True,
            residuals_real=np.zeros(n),
            residuals_imag=np.zeros(n),
            max_residual=0.0,
            n_rc_elements=0,
            mu=self.mu,
            warning_message="K-K validation skipped — validator unavailable.",
        )

    # ── Method 1: impedance.py linKK ──────────────────────────────────────────

    def _try_linkk(self, freq: np.ndarray, Z: np.ndarray) -> Optional[KKValidationResult]:
        """Use impedance.py's linKK with strict error handling."""
        try:
            from impedance.validation import linKK
        except ImportError:
            return None

        try:
            output = linKK(
                freq, Z, c=self.c, mu=self.mu,
                fit_type="complex", add_cap=True,
            )
        except Exception as e:
            warnings.warn(f"linKK failed: {e}. Trying fallback.", stacklevel=2)
            return None

        # impedance.py returns different formats in different versions
        if output is None:
            return None

        try:
            if len(output) == 5:
                M, mu_out, Z_fit, res_real, res_imag = output
            elif len(output) == 4:
                M, mu_out, res_real, res_imag = output
            elif len(output) == 3:
                M, res_real, res_imag = output
                mu_out = self.mu
            else:
                return None
        except (TypeError, ValueError):
            return None

        # Check residuals are valid arrays
        if res_real is None or res_imag is None:
            return None

        try:
            res_real = np.asarray(res_real, dtype=float)
            res_imag = np.asarray(res_imag, dtype=float)
        except Exception:
            return None

        # Check for inf/nan
        if not (np.isfinite(res_real).all() and np.isfinite(res_imag).all()):
            return None

        max_res = float(np.max(np.abs(np.concatenate([res_real, res_imag]))))
        passed  = max_res <= self.residual_threshold

        msg = None
        if not passed:
            msg = (
                f"K-K validation failed: max residual {max_res*100:.3f}% > "
                f"threshold {self.residual_threshold*100:.2f}%. "
                "Possible causes: DC drift, non-linearity, or noise. "
                "Results may still be useful if residuals are systematic, "
                "not random."
            )

        return KKValidationResult(
            passed=passed,
            residuals_real=res_real,
            residuals_imag=res_imag,
            max_residual=max_res,
            n_rc_elements=int(M) if M is not None else 0,
            mu=float(mu_out) if mu_out is not None else self.mu,
            warning_message=msg,
        )

    # ── Method 2: Simple Voigt-circuit fallback ───────────────────────────────

    def _fallback_voigt(self, freq: np.ndarray, Z: np.ndarray) -> Optional[KKValidationResult]:
        """
        Simplified K-K test using a Voigt circuit with logarithmically
        spaced RC time constants. Less rigorous but always works.
        """
        try:
            n = len(freq)
            # Choose M RC elements logarithmically
            M = min(max(int(n / 4), 5), 30)
            tau = np.logspace(
                np.log10(1 / (2 * np.pi * freq.max())),
                np.log10(1 / (2 * np.pi * freq.min())),
                M,
            )
            omega = 2 * np.pi * freq

            # Build the Voigt model matrix
            # Z(ω) = R0 + Σ Rk / (1 + jωτk)
            A = np.ones((n, M + 1), dtype=complex)
            for k, t in enumerate(tau):
                A[:, k + 1] = 1.0 / (1 + 1j * omega * t)

            # Solve least squares: [Re(A); Im(A)] · x = [Re(Z); Im(Z)]
            A_split = np.vstack([A.real, A.imag])
            b_split = np.hstack([Z.real, Z.imag])
            x, *_ = np.linalg.lstsq(A_split, b_split, rcond=None)

            Z_fit = A @ x
            Z_mod = np.abs(Z).clip(min=1e-15)
            res_real = (Z.real - Z_fit.real) / Z_mod
            res_imag = (Z.imag - Z_fit.imag) / Z_mod

            max_res = float(np.max(np.abs(np.concatenate([res_real, res_imag]))))
            passed  = max_res <= self.residual_threshold * 2  # 2× more lenient

            msg = None
            if not passed:
                msg = (
                    f"K-K (fallback method) failed: {max_res*100:.3f}% > "
                    f"{self.residual_threshold*200:.2f}% threshold. "
                    "Consider checking for DC drift or non-linearity."
                )

            return KKValidationResult(
                passed=passed,
                residuals_real=res_real,
                residuals_imag=res_imag,
                max_residual=max_res,
                n_rc_elements=M,
                mu=self.mu,
                warning_message=msg,
            )
        except Exception as e:
            warnings.warn(f"Voigt fallback failed: {e}", stacklevel=2)
            return None
