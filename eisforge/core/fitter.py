from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from impedance.models.circuits import CustomCircuit
from eisforge.parsers.base_parser import EISDataset


@dataclass
class FitResult:
    circuit_string: str
    parameters: dict
    parameter_errors: dict
    chi_squared: float
    z_fit: np.ndarray
    converged: bool
    n_function_evaluations: int = 0
    _circuit_obj: Optional[object] = field(default=None, repr=False)

    def parameter_table(self):
        lines = [f"{'Parameter':<16} {'Value':>16} {'±Error':>16}", "─"*50]
        for name, val in self.parameters.items():
            err = self.parameter_errors.get(name, float("nan"))
            lines.append(f"{name:<16} {val:>16.4e} {err:>16.2e}")
        lines.append(f"\nReduced χ² = {self.chi_squared:.6f}")
        return "\n".join(lines)


class CNLSFitter:
    def __init__(self, circuit_string, initial_guess, bounds=None, weight_by_modulus=True):
        self.circuit_string   = circuit_string
        self.initial_guess    = initial_guess
        self.bounds           = bounds
        self.weight_by_modulus = weight_by_modulus

    def fit(self, dataset: EISDataset) -> FitResult:
        freq   = dataset.frequency
        Z_meas = dataset.z_complex
        weights = 1.0 / np.abs(Z_meas) if self.weight_by_modulus else np.ones(len(freq))

        try:
            circuit = CustomCircuit(
                circuit=self.circuit_string,
                initial_guess=self.initial_guess,
            )
        except Exception as e:
            raise ValueError(f"Invalid circuit '{self.circuit_string}': {e}") from e

        converged = True
        try:
            if self.bounds:
                circuit.fit(freq, Z_meas, weight=weights, bounds=self.bounds)
            else:
                circuit.fit(freq, Z_meas, weight=weights)
        except Exception as e:
            warnings.warn(f"Fitting did not converge: {e}", stacklevel=2)
            converged = False

        param_names    = circuit.get_param_names()
        fitted_params  = circuit.parameters_
        parameters     = dict(zip(param_names, fitted_params))

        try:
            errors = dict(zip(param_names, np.asarray(circuit.conf_, dtype=float)))
        except Exception:
            errors = {n: float("nan") for n in param_names}

        z_fit = circuit.predict(freq)
        n_pts, n_par = len(freq), len(fitted_params)
        dof = max(2 * n_pts - n_par, 1)
        res_r = (Z_meas.real - z_fit.real) * weights
        res_i = (Z_meas.imag - z_fit.imag) * weights
        chi2  = float(np.sum(res_r**2 + res_i**2) / dof)

        return FitResult(
            circuit_string=self.circuit_string,
            parameters=parameters,
            parameter_errors=errors,
            chi_squared=chi2,
            z_fit=z_fit,
            converged=converged,
            _circuit_obj=circuit,
        )
