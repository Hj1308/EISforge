"""
Physics-Informed Loss Function — the core innovation of EISForge.

This file is the most important differentiator between EISForge
and all existing ML-based EIS analysis methods.

The problem with conventional ML approaches:
---------------------------------------------
A CNN or Random Forest simply learns to fit the data.
There is no guarantee that the output is physically meaningful.
For example, the model may predict Re(Z) < 0 — which is
physically impossible for a passive electrochemical interface.

Our solution — three physical constraints embedded in the Loss Function:
------------------------------------------------------------------------

Constraint 1: Kramers-Kronig (K-K) relations
    The system must be causal, linear, and stable.
    K-K dictates that Z_real and Z_imag are not independent:
    if Z_imag is known over all frequencies, Z_real is fully determined,
    and vice versa. Violation of K-K implies the system was drifting
    during measurement or exhibits nonlinear behaviour.

Constraint 2: Passivity
    Re(Z(ω)) ≥ 0  for all ω.
    An electrochemical interface cannot generate energy — it can only
    dissipate or store it. Violation is physically impossible unless
    the system is actively driven (e.g. a battery under charge).

Constraint 3: High-Frequency Limit
    As ω → ∞:  Im(Z) → 0  and  Re(Z) → R_solution.
    At sufficiently high frequency, all reactive elements (capacitors,
    inductors) become short circuits or open circuits, leaving only
    the ohmic solution resistance.

Final composite loss:
    L_total = L_recon + λ₁·L_kk + λ₂·L_passivity + λ₃·L_hf

Note on K-K discretisation:
    The full K-K integral runs to ±∞ in frequency. On a finite,
    log-spaced measurement band the constraint is inherently approximate.
    The truncation error depends on the measurement bandwidth and should
    be reported when comparing models quantitatively (see paper §3.2).

Author: Hoda Jafari
Date:   May 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed composite loss function for EIS-GPT.

    Combines a modulus-weighted reconstruction loss with three
    electrochemical physics penalties:
    * Kramers-Kronig consistency
    * Passivity (Re(Z) ≥ 0)
    * High-frequency impedance limit

    Parameters
    ----------
    lambda_kk : float
        Weight for the Kramers-Kronig violation penalty (default: 0.1).
    lambda_passivity : float
        Weight for the passivity violation penalty (default: 0.5).
        Set higher than ``lambda_kk`` because passivity violation is
        a harder physical constraint.
    lambda_hf : float
        Weight for the high-frequency limit penalty (default: 0.05).
    reduction : str
        Aggregation method: ``'mean'`` (default) or ``'sum'``.
    """

    def __init__(
        self,
        lambda_kk: float = 0.1,
        lambda_passivity: float = 0.5,
        lambda_hf: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.lambda_kk = lambda_kk
        self.lambda_passivity = lambda_passivity
        self.lambda_hf = lambda_hf
        self.reduction = reduction

    def forward(
        self,
        z_pred_real: torch.Tensor,
        z_pred_imag: torch.Tensor,
        z_true_real: torch.Tensor,
        z_true_imag: torch.Tensor,
        freq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the total physics-informed loss.

        Parameters
        ----------
        z_pred_real : Tensor, shape (batch, N)
            Predicted real part of impedance Re(Z).
        z_pred_imag : Tensor, shape (batch, N)
            Predicted imaginary part −Im(Z) (sign convention: positive
            for capacitive arcs in the Nyquist plot).
        z_true_real : Tensor, shape (batch, N)
            Ground-truth real part of impedance.
        z_true_imag : Tensor, shape (batch, N)
            Ground-truth imaginary part −Im(Z).
        freq : Tensor, shape (batch, N)
            Frequency array in Hz, ordered from lowest to highest.

        Returns
        -------
        dict[str, Tensor]
            ``'total'``      : weighted composite loss
            ``'recon'``      : modulus-weighted reconstruction (MSE)
            ``'kk'``         : Kramers-Kronig smoothness penalty
            ``'passivity'``  : passivity violation penalty
            ``'hf'``         : high-frequency limit penalty
        """
        # ── Modulus-weighted reconstruction loss (IUPAC convention) ──────────
        # Dividing by |Z| makes the loss scale-invariant, giving equal weight
        # to high-impedance (low-frequency) and low-impedance (high-frequency)
        # regions — following the IUPAC recommendation for CNLS fitting.
        z_mod = torch.sqrt(z_true_real**2 + z_true_imag**2).clamp(min=1e-10)
        recon_real = ((z_pred_real - z_true_real) / z_mod) ** 2
        recon_imag = ((z_pred_imag - z_true_imag) / z_mod) ** 2
        L_recon = self._reduce(recon_real + recon_imag)

        # ── Passivity penalty ─────────────────────────────────────────────────
        # Re(Z) must be ≥ 0. Penalise negative predictions via ReLU(-Z_real).
        L_passivity = self._reduce(F.relu(-z_pred_real) ** 2)

        # ── Kramers-Kronig penalty (approximate, finite-band) ─────────────────
        # Full K-K verification requires integration to ±∞; here we enforce a
        # necessary (not sufficient) condition: Re(Z) should decrease
        # monotonically with frequency for typical Voigt-type spectra, and
        # Im(Z) should vary smoothly.
        # Limitation: this is a soft proxy — it catches gross violations but
        # cannot replace a full linKK validation (see eisforge.core.kk).
        if z_pred_real.size(1) > 1:
            dZr_df = torch.diff(z_pred_real, dim=1)
            dZi_df = torch.diff(z_pred_imag, dim=1)
            # Re(Z) should decrease as frequency increases: dZr/df ≤ 0
            L_kk_real = self._reduce(F.relu(dZr_df) ** 2)
            # Im(Z) should be smooth (small second differences)
            L_kk_imag = self._reduce(dZi_df ** 2) * 0.01
            L_kk = L_kk_real + L_kk_imag
        else:
            L_kk = torch.tensor(0.0, device=z_pred_real.device)

        # ── High-frequency limit penalty ──────────────────────────────────────
        # At the highest measured frequency (last point, assuming ascending
        # frequency order): Im(Z) should approach zero.
        z_imag_hf = z_pred_imag[:, -1]
        L_hf = self._reduce(z_imag_hf ** 2)

        # ── Composite loss ────────────────────────────────────────────────────
        L_total = (
            L_recon
            + self.lambda_kk * L_kk
            + self.lambda_passivity * L_passivity
            + self.lambda_hf * L_hf
        )

        return {
            "total": L_total,
            "recon": L_recon,
            "kk": L_kk,
            "passivity": L_passivity,
            "hf": L_hf,
        }

    def _reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply reduction (mean or sum) to a tensor."""
        if self.reduction == "mean":
            return tensor.mean()
        return tensor.sum()
