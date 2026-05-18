"""
EIS Tokenizer — تبدیل طیف EIS به sequence قابل فهم برای Transformer.

ایده اصلی:
-----------
درست مثل اینکه GPT هر کلمه را به یک vector تبدیل می‌کند،
ما هر نقطه فرکانسی EIS را به یک embedding vector تبدیل می‌کنیم.

هر token شامل:
    [log(f), Z_real_norm, Z_imag_norm, log(|Z|), θ_norm]
    
که یک vector 5 بعدی است. بعد با یک لایه Linear به
d_model بعد (مثلاً 128) تبدیل می‌شود + Positional Encoding.

چرا log(f)?
-----------
فرکانس‌ها از 10 mHz تا 1 MHz هستند — 8 مرتبه بزرگی.
اگر مستقیم استفاده کنیم، مدل به فرکانس‌های بالا bias پیدا می‌کند.
log(f) این مشکل را حل می‌کند.

چرا Positional Encoding?
------------------------
Transformer ذاتاً ترتیب را نمی‌فهمد.
Positional Encoding به مدل می‌گوید هر token کجای sequence است —
یعنی فرکانس پایین (ابتدا) یا فرکانس بالا (انتها).

نویسنده: Hoda Jafari
تاریخ: May 2026
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class EISTokenizer(nn.Module):
    """
    تبدیل طیف EIS خام به sequence از embedding vectors.

    Parameters
    ----------
    d_model : int
        بعد embedding vector (پیش‌فرض: 128).
        باید زوج باشد برای Positional Encoding.
    max_seq_len : int
        حداکثر تعداد نقاط فرکانسی (پیش‌فرض: 128).
    dropout : float
        نرخ Dropout برای جلوگیری از overfitting.

    Input:
        freq   : (batch, N)    — فرکانس در Hz
        z_real : (batch, N)    — بخش حقیقی امپدانس
        z_imag : (batch, N)    — بخش موهومی −Im(Z)

    Output:
        embeddings : (batch, N, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(f"d_model باید زوج باشد، دریافت شد: {d_model}")

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 5 ویژگی خام → d_model بعد
        self.input_projection = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional Encoding ثابت (sinusoidal)
        self.register_buffer(
            "positional_encoding",
            self._build_positional_encoding(max_seq_len, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """
        ساخت Sinusoidal Positional Encoding.

        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        این فرمول از مقاله اصلی Transformer (Vaswani et al. 2017) است.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def _extract_features(
        self,
        freq: torch.Tensor,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        استخراج و نرمال‌سازی ۵ ویژگی از هر نقطه فرکانسی.

        ویژگی‌ها:
            0: log10(f)          — فرکانس لگاریتمی
            1: Z_real_norm       — بخش حقیقی نرمال‌شده
            2: Z_imag_norm       — بخش موهومی نرمال‌شده
            3: log10(|Z|)        — مدول لگاریتمی
            4: θ / 90            — فاز نرمال‌شده به [−2, 2]
        """
        eps = 1e-10

        # log فرکانس
        log_f = torch.log10(freq.clamp(min=eps))

        # مدول و فاز
        z_mod = torch.sqrt(z_real**2 + z_imag**2).clamp(min=eps)
        log_z_mod = torch.log10(z_mod)
        phase = torch.atan2(z_imag, z_real) * (180.0 / np.pi) / 90.0

        # نرمال‌سازی Z_real و Z_imag با مدول
        z_real_norm = z_real / z_mod
        z_imag_norm = z_imag / z_mod

        # ترکیب: (batch, N, 5)
        features = torch.stack(
            [log_f, z_real_norm, z_imag_norm, log_z_mod, phase], dim=-1
        )
        return features

    def forward(
        self,
        freq: torch.Tensor,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass tokenizer.

        Parameters
        ----------
        freq   : Tensor (batch, N)
        z_real : Tensor (batch, N)
        z_imag : Tensor (batch, N)

        Returns
        -------
        Tensor (batch, N, d_model)
        """
        features = self._extract_features(freq, z_real, z_imag)
        x = self.input_projection(features)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        return self.dropout(x)
