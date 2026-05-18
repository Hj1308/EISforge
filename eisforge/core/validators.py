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
    def residuals_max_pct(self):
        return self.max_residual * 100.0

    def summary(self):
        status = "PASSED" if self.passed else "FAILED"
        return (f"K-K: {status} | max residual={self.residuals_max_pct:.3f}% | "
                f"N_RC={self.n_rc_elements} | μ={self.mu:.3f}")


class KramersKronigValidator:
    def __init__(self, residual_threshold=0.005, mu=0.85, c=0.5):
        self.residual_threshold = residual_threshold
        self.mu = mu
        self.c = c

    def validate(self, dataset: EISDataset) -> KKValidationResult:
        Z = dataset.z_complex
        try:
            from impedance.validation import linKK
            M, mu_out, Z_fit, res_real, res_imag = linKK(
                dataset.frequency, Z, c=self.c, mu=self.mu,
                fit_type="complex", add_cap=True,
            )
        except Exception as exc:
            n = len(dataset.frequency)
            warnings.warn(f"lin-KK failed: {exc}", stacklevel=2)
            return KKValidationResult(
                passed=False,
                residuals_real=np.zeros(n),
                residuals_imag=np.zeros(n),
                max_residual=np.inf,
                n_rc_elements=0,
                mu=self.mu,
                warning_message=str(exc),
            )

        max_res = float(np.max(np.abs(np.concatenate([res_real, res_imag]))))
        passed  = max_res <= self.residual_threshold
        msg = None if passed else (
            f"K-K FAILED: {max_res*100:.3f}% > {self.residual_threshold*100:.2f}%. "
            "Possible drift or non-linearity."
        )
        return KKValidationResult(
            passed=passed,
            residuals_real=res_real,
            residuals_imag=res_imag,
            max_residual=max_res,
            n_rc_elements=M,
            mu=float(mu_out),
            warning_message=msg,
        )
