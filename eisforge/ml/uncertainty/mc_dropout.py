"""
Monte Carlo Dropout — Epistemic Uncertainty for EIS-GPT.
Author: Hoda Jafari | May 2026

Two uncertainty types:
  Aleatoric  (from sigma head)  = noise in the data itself
  Epistemic  (from MC Dropout)  = model lacks training data — Active Learning targets this
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import torch
from eisforge.ml.eis_gpt.transformer import CIRCUIT_NAMES, MAX_PARAMS


@dataclass
class UncertaintyResult:
    """Full uncertainty analysis for one EIS spectrum."""
    epistemic_score      : float
    aleatoric_score      : float
    circuit_probs_mean   : np.ndarray
    circuit_probs_std    : np.ndarray
    param_mu_mean        : np.ndarray
    param_epistemic_std  : np.ndarray
    param_aleatoric_std  : np.ndarray
    predicted_circuit    : str
    confidence           : float
    n_samples            : int
    per_param_uncertainty: dict = field(default_factory=dict)

    def should_query(self, threshold: float = 0.15) -> bool:
        """True when epistemic uncertainty exceeds threshold — model needs more data."""
        return self.epistemic_score > threshold

    def confidence_label(self, threshold: float = 0.15) -> str:
        ratio = self.epistemic_score / threshold
        if ratio < 0.4:   return "High"
        elif ratio < 0.8: return "Medium"
        elif ratio < 1.2: return "Low"
        else:             return "Very Low — query recommended"

    def confidence_pct(self, threshold: float = 0.15) -> int:
        ratio = self.epistemic_score / threshold
        return min(99, max(5, int(100 - ratio * 60)))

    def uncertain_params(self, relative_threshold: float = 0.5) -> list[int]:
        relative_unc = self.param_epistemic_std / (np.abs(self.param_mu_mean) + 1e-8)
        return [i for i, v in enumerate(relative_unc) if v > relative_threshold]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  Uncertainty Analysis — EISForge Active Learning",
            "=" * 60,
            f"  MC samples used       : {self.n_samples}",
            f"  Predicted circuit     : {self.predicted_circuit}",
            f"  Circuit confidence    : {self.confidence:.1%}",
            "-" * 60,
            f"  Epistemic uncertainty : {self.epistemic_score:.4f}",
            f"  Aleatoric uncertainty : {self.aleatoric_score:.4f}",
            f"  Confidence level      : {self.confidence_label()}",
            "-" * 60,
        ]
        for i, (ep, al) in enumerate(zip(self.param_epistemic_std, self.param_aleatoric_std)):
            lines.append(f"  param_{i}  epistemic={ep:.4f}  aleatoric={al:.4f}")
        uncertain = self.uncertain_params()
        if uncertain:
            lines.append(f"  High-uncertainty params: {uncertain}")
        lines.append("=" * 60)
        return "\n".join(lines)


def mc_dropout_predict(
    model,
    freq     : torch.Tensor,
    z_real   : torch.Tensor,
    z_imag   : torch.Tensor,
    n_samples: int = 50,
) -> UncertaintyResult:
    """
    Monte Carlo Dropout inference — keeps dropout ACTIVE for stochastic outputs.

    Run N forward passes with model.train() → measure variance across passes.
    High variance = epistemic uncertainty = model needs more labeled data.

    Parameters
    ----------
    model      : EISForgeModel
    freq       : Tensor (1, N)  frequency in Hz
    z_real     : Tensor (1, N)  Re(Z)
    z_imag     : Tensor (1, N)  -Im(Z)
    n_samples  : int            number of MC passes (default: 50)

    Returns
    -------
    UncertaintyResult with epistemic + aleatoric breakdown
    """
    was_training = model.training

    # Keep dropout ACTIVE — stochastic inference
    model.train()
    circuit_probs_list, param_mu_list = [], []

    with torch.no_grad():
        for _ in range(n_samples):
            out = model.forward(freq, z_real, z_imag)
            circuit_probs_list.append(out["circuit_logprobs"].exp().cpu().numpy())
            param_mu_list.append(out["param_mu"].cpu().numpy())

    # One eval-mode pass for aleatoric sigma
    model.eval()
    with torch.no_grad():
        out_eval = model.forward(freq, z_real, z_imag)
        aleatoric_sigma = out_eval["param_sigma"][0].cpu().numpy()

    model.train(was_training)

    # Stack: (n_samples, batch, ...) → squeeze batch dim
    circuit_probs_arr = np.stack(circuit_probs_list).squeeze(1)  # (n, N_CIRCUITS)
    param_mu_arr      = np.stack(param_mu_list).squeeze(1)       # (n, MAX_PARAMS)

    circuit_probs_mean  = circuit_probs_arr.mean(axis=0)
    circuit_probs_std   = circuit_probs_arr.std(axis=0)
    param_mu_mean       = param_mu_arr.mean(axis=0)
    param_epistemic_std = param_mu_arr.std(axis=0)

    # Overall scores
    epistemic_score = 0.5 * float(circuit_probs_std.mean()) + 0.5 * float(param_epistemic_std.mean())
    aleatoric_score = float(aleatoric_sigma.mean())

    predicted_idx     = int(circuit_probs_mean.argmax())
    predicted_circuit = CIRCUIT_NAMES[predicted_idx]
    confidence        = float(circuit_probs_mean[predicted_idx])

    per_param = {
        f"param_{i}": {
            "epistemic_std" : float(param_epistemic_std[i]),
            "aleatoric_std" : float(aleatoric_sigma[i]),
            "mu_mean"       : float(param_mu_mean[i]),
            "value_estimate": float(np.exp(param_mu_mean[i])),
        }
        for i in range(MAX_PARAMS)
    }

    return UncertaintyResult(
        epistemic_score       = epistemic_score,
        aleatoric_score       = aleatoric_score,
        circuit_probs_mean    = circuit_probs_mean,
        circuit_probs_std     = circuit_probs_std,
        param_mu_mean         = param_mu_mean,
        param_epistemic_std   = param_epistemic_std,
        param_aleatoric_std   = aleatoric_sigma,
        predicted_circuit     = predicted_circuit,
        confidence            = confidence,
        n_samples             = n_samples,
        per_param_uncertainty = per_param,
    )


def batch_epistemic_rank(
    model,
    spectra_list: list[tuple],
    n_samples   : int = 30,
) -> list[tuple[int, float]]:
    """
    Rank unlabeled spectra by epistemic uncertainty (most uncertain first).
    Use this to decide which experiment to label next.
    """
    scores = [
        (idx, mc_dropout_predict(model, f, zr, zi, n_samples=n_samples).epistemic_score)
        for idx, (f, zr, zi) in enumerate(spectra_list)
    ]
    return sorted(scores, key=lambda x: x[1], reverse=True)
