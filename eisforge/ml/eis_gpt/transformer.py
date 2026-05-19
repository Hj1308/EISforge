"""
EIS-GPT Transformer — the core prediction engine of EISForge.
Author: Hoda Jafari | May 2026

Architecture:
-------------
                    Raw EIS spectrum
                          ↓
                    EISTokenizer
                          ↓
               Transformer Encoder (6 layers)
               (Self-Attention + FFN + LayerNorm)
                          ↓
                  [CLS] token vector
                   ↙              ↘
         CircuitHead          ParameterHead
     (circuit topology)   (parameter values)
            ↓                      ↓
    Probability over         μ + σ per parameter
      5 circuits             (aleatoric uncertainty)

Why Transformer and not CNN?
-----------------------------
A CNN treats the EIS spectrum like a local image.
A Transformer with Self-Attention can directly learn the
relationship between 100 kHz and 10 mHz — exactly what EIS
physics requires: the Warburg element at low frequency is
correlated with R_ct at mid frequency.

Uncertainty Quantification (two types):
-----------------------------------------
  Aleatoric uncertainty (σ head):
      Irreducible noise in the data itself.
      Model predicts μ AND σ for each parameter.
      High σ → noisy or poorly conditioned spectrum.

  Epistemic uncertainty (MC Dropout):
      Model's lack of knowledge — reducible with more data.
      See eisforge/ml/uncertainty/mc_dropout.py.
      High epistemic → Active Learning should query the user.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eisforge.ml.eis_gpt.tokenizer import EISTokenizer
from eisforge.ml.eis_gpt.physics_loss import PhysicsInformedLoss


# تعداد مدارهای پشتیبانی‌شده
N_CIRCUITS = 5

# حداکثر تعداد پارامتر در یک مدار
MAX_PARAMS = 7

# نام مدارها
CIRCUIT_NAMES = [
    "R0-p(R1,C1)",
    "R0-p(R1,CPE1)",
    "R0-p(R1,CPE1)-W1",
    "R0-p(R1,CPE1)-p(R2,CPE2)",
    "R0-p(R1,CPE1)-Wo1",
]


class EISTransformerEncoder(nn.Module):
    """
    Transformer Encoder برای پردازش sequence طیف EIS.

    Parameters
    ----------
    d_model : int
        بعد embedding (پیش‌فرض: 128).
    n_heads : int
        تعداد attention heads (پیش‌فرض: 8).
        باید d_model را بخش‌پذیر کند.
    n_layers : int
        تعداد لایه‌های Transformer (پیش‌فرض: 6).
    d_ff : int
        بعد Feed-Forward داخل (پیش‌فرض: 512).
    dropout : float
        نرخ Dropout.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",     # GELU بهتر از ReLU برای Transformer
            batch_first=True,      # (batch, seq, feature)
            norm_first=True,       # Pre-LN: آموزش پایدارتر
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (batch, N, d_model)

        Returns
        -------
        Tensor (batch, N, d_model)
        """
        return self.encoder(x)


class CircuitClassifierHead(nn.Module):
    """
    سر پیش‌بینی توپولوژی مدار.

    از [CLS] token (اولین token) برای طبقه‌بندی استفاده می‌کند.
    خروجی: توزیع احتمال روی N_CIRCUITS مدار.
    """

    def __init__(self, d_model: int = 128, n_circuits: int = N_CIRCUITS) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_circuits),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_token : Tensor (batch, d_model)

        Returns
        -------
        Tensor (batch, n_circuits) — log probabilities
        """
        return F.log_softmax(self.classifier(cls_token), dim=-1)


class ParameterRegressionHead(nn.Module):
    """
    سر پیش‌بینی پارامترهای مدار با Uncertainty Quantification.

    برای هر پارامتر دو مقدار پیش‌بینی می‌کند:
        μ  : مقدار میانگین (در log-space)
        σ  : انحراف معیار (عدم‌قطعیت)

    پیش‌بینی در log-space انجام می‌شود چون پارامترها
    چندین مرتبه بزرگی را پوشش می‌دهند.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_params: int = MAX_PARAMS,
    ) -> None:
        super().__init__()
        self.max_params = max_params

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, max_params * 2),   # μ و σ برای هر پارامتر
        )

    def forward(self, cls_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        cls_token : Tensor (batch, d_model)

        Returns
        -------
        mu    : Tensor (batch, max_params) — میانگین در log-space
        sigma : Tensor (batch, max_params) — انحراف معیار (همیشه مثبت)
        """
        out = self.regressor(cls_token)
        mu    = out[:, :self.max_params]
        sigma = F.softplus(out[:, self.max_params:]) + 1e-6   # همیشه > 0
        return mu, sigma


class EISForgeModel(nn.Module):
    """
    مدل کامل EIS-GPT.

    این کلاس همه اجزا را کنار هم قرار می‌دهد:
        EISTokenizer → Transformer → CircuitHead + ParameterHead

    Parameters
    ----------
    d_model : int
        بعد embedding.
    n_heads : int
        تعداد attention heads.
    n_layers : int
        تعداد لایه‌های Transformer.
    d_ff : int
        بعد Feed-Forward.
    dropout : float
        نرخ Dropout.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # [CLS] token — یک vector قابل یادگیری در ابتدای sequence
        # مثل BERT: این token وضعیت کلی طیف را خلاصه می‌کند
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.tokenizer = EISTokenizer(d_model=d_model, dropout=dropout)
        self.encoder   = EISTransformerEncoder(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
        )
        self.circuit_head   = CircuitClassifierHead(d_model=d_model)
        self.parameter_head = ParameterRegressionHead(d_model=d_model)
        self.loss_fn        = PhysicsInformedLoss()

    def forward(
        self,
        freq: torch.Tensor,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass کامل.

        Parameters
        ----------
        freq   : Tensor (batch, N) — فرکانس در Hz
        z_real : Tensor (batch, N) — Re(Z)
        z_imag : Tensor (batch, N) — −Im(Z)

        Returns
        -------
        dict شامل:
            'circuit_logprobs' : (batch, N_CIRCUITS) — log احتمال هر مدار
            'param_mu'         : (batch, MAX_PARAMS) — میانگین پارامترها
            'param_sigma'      : (batch, MAX_PARAMS) — عدم‌قطعیت پارامترها
            'cls_features'     : (batch, d_model)   — ویژگی‌های خام
        """
        batch_size = freq.size(0)

        # ── Tokenize ──────────────────────────────────────────────────────────
        tokens = self.tokenizer(freq, z_real, z_imag)   # (batch, N, d_model)

        # ── اضافه کردن [CLS] token به ابتدای sequence ─────────────────────────
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)          # (batch, N+1, d_model)

        # ── Transformer Encoder ───────────────────────────────────────────────
        encoded = self.encoder(tokens)                    # (batch, N+1, d_model)

        # ── استخراج [CLS] token برای پیش‌بینی ────────────────────────────────
        cls_output = encoded[:, 0, :]                     # (batch, d_model)

        # ── Prediction Heads ──────────────────────────────────────────────────
        circuit_logprobs       = self.circuit_head(cls_output)
        param_mu, param_sigma  = self.parameter_head(cls_output)

        return {
            "circuit_logprobs": circuit_logprobs,
            "param_mu":         param_mu,
            "param_sigma":      param_sigma,
            "cls_features":     cls_output,
        }

    @torch.no_grad()
    def mc_predict(
        self,
        freq     : torch.Tensor,
        z_real   : torch.Tensor,
        z_imag   : torch.Tensor,
        n_samples: int = 50,
    ) -> dict:
        """
        Monte Carlo Dropout prediction with epistemic uncertainty.

        Unlike predict() which calls self.eval() (dropout OFF),
        this method keeps dropout ACTIVE to produce stochastic outputs.
        Running N passes and measuring variance gives epistemic uncertainty.

        Use this method when you need to know HOW CONFIDENT the model is,
        not just WHAT it predicts.

        Parameters
        ----------
        freq, z_real, z_imag : Tensor (1, N)
        n_samples : int
            Number of stochastic forward passes. Default: 50.

        Returns
        -------
        dict with all fields from predict() PLUS:
            'epistemic_score'  : float — overall epistemic uncertainty
            'aleatoric_score'  : float — overall aleatoric uncertainty
            'should_query'     : bool  — True if model recommends human label
            'confidence_pct'   : int   — confidence as percentage (0-100)
            'circuit_probs_std': ndarray — std across MC samples per circuit
            'param_epistemic'  : ndarray — epistemic std per parameter
        """
        from eisforge.ml.uncertainty.mc_dropout import mc_dropout_predict
        result = mc_dropout_predict(self, freq, z_real, z_imag, n_samples=n_samples)

        import numpy as np
        param_values = np.exp(result.param_mu_mean)
        param_errors = param_values * result.param_aleatoric_std

        parameters = {
            f"param_{i}": {
                "value"          : float(param_values[i]),
                "uncertainty_al" : float(param_errors[i]),
                "uncertainty_ep" : float(result.param_epistemic_std[i]),
            }
            for i in range(MAX_PARAMS)
        }

        return {
            "predicted_circuit" : result.predicted_circuit,
            "confidence"        : result.confidence,
            "top3"              : [
                {"circuit": CIRCUIT_NAMES[i], "probability": float(result.circuit_probs_mean[i])}
                for i in result.circuit_probs_mean.argsort()[::-1][:3]
            ],
            "parameters"        : parameters,
            "epistemic_score"   : result.epistemic_score,
            "aleatoric_score"   : result.aleatoric_score,
            "should_query"      : result.should_query(threshold=0.15),
            "confidence_pct"    : result.confidence_pct(threshold=0.15),
            "confidence_label"  : result.confidence_label(threshold=0.15),
            "circuit_probs_std" : result.circuit_probs_std,
            "param_epistemic"   : result.param_epistemic_std,
        }

    @torch.no_grad()
    def predict(
        self,
        freq: torch.Tensor,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
    ) -> dict:
        """
        پیش‌بینی نهایی با تفسیر انسانی.

        Returns
        -------
        dict شامل:
            predicted_circuit : str     — نام مدار پیش‌بینی‌شده
            confidence        : float   — اطمینان (0 تا 1)
            top3              : list    — سه مدار برتر با احتمال
            parameters        : dict    — پارامترها با عدم‌قطعیت
        """
        self.eval()
        outputs = self.forward(freq, z_real, z_imag)

        # مدار پیش‌بینی‌شده
        probs = outputs["circuit_logprobs"].exp()[0]
        top3_idx = probs.argsort(descending=True)[:3]

        predicted_idx    = int(top3_idx[0])
        predicted_circuit = CIRCUIT_NAMES[predicted_idx]
        confidence        = float(probs[predicted_idx])

        top3 = [
            {
                "circuit":     CIRCUIT_NAMES[int(i)],
                "probability": float(probs[i]),
            }
            for i in top3_idx
        ]

        # پارامترها در مقیاس اصلی (exp از log-space)
        mu    = outputs["param_mu"][0].cpu().numpy()
        sigma = outputs["param_sigma"][0].cpu().numpy()

        import numpy as np
        param_values = np.exp(mu)
        param_errors = param_values * sigma   # خطا در مقیاس اصلی

        parameters = {
            f"param_{i}": {
                "value": float(param_values[i]),
                "uncertainty": float(param_errors[i]),
            }
            for i in range(MAX_PARAMS)
        }

        return {
            "predicted_circuit": predicted_circuit,
            "confidence":        confidence,
            "top3":              top3,
            "parameters":        parameters,
        }

    def save(self, path: str) -> None:
        """ذخیره مدل."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load_pretrained(cls, path: str, **kwargs) -> "EISForgeModel":
        """بارگذاری مدل از فایل."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
