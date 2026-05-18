"""
Physics-Informed Loss Function — قلب نوآوری EISForge.

این فایل مهم‌ترین تفاوت EISForge با همه روش‌های ML موجود است.

مشکل روش‌های ML معمولی:
------------------------
یک CNN یا Random Forest فقط یاد می‌گیرد که داده را fit کند.
هیچ تضمینی نیست که خروجی از نظر فیزیکی معنادار باشد.
مثلاً ممکن است Re(Z) < 0 پیش‌بینی کند — که فیزیکاً محال است!

راه‌حل ما — سه قانون فیزیکی در Loss Function:
-----------------------------------------------

قانون ۱: Kramers-Kronig (K-K)
    سیستم باید causal و stable باشد.
    K-K می‌گوید Z_real و Z_imag از هم مستقل نیستند:
    اگر Z_imag را بدانیم، Z_real کاملاً مشخص است و بالعکس.
    نقض K-K → سیستم در حال drift بوده یا غیرخطی است.

قانون ۲: Passivity
    Re(Z(ω)) ≥ 0 برای همه ω
    یعنی اینترفیس الکتروشیمیایی انرژی تولید نمی‌کند.
    نقض → فیزیکاً محال (مگر در سیستم‌های active مثل باتری در charge)

قانون ۳: High-Frequency Limit
    وقتی ω → ∞:  Im(Z) → 0  و  Re(Z) → R_solution
    یعنی در فرکانس خیلی بالا، فقط مقاومت اهمی محلول باقی می‌ماند.

فرمول نهایی:
    L_total = L_recon + λ₁·L_kk + λ₂·L_passivity + λ₃·L_hf

نویسنده: Hoda Jafari
تاریخ: May 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    تابع Loss فیزیک‌محور برای EIS-GPT.

    Parameters
    ----------
    lambda_kk : float
        وزن جریمه نقض Kramers-Kronig (پیش‌فرض: 0.1).
    lambda_passivity : float
        وزن جریمه نقض Passivity (پیش‌فرض: 0.5).
        بالاتر از KK چون نقض آن شدیدتر است.
    lambda_hf : float
        وزن جریمه نقض شرط High-Frequency (پیش‌فرض: 0.05).
    reduction : str
        روش تجمیع loss: 'mean' یا 'sum'.
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
        محاسبه Loss کل.

        Parameters
        ----------
        z_pred_real : Tensor (batch, N)
            بخش حقیقی امپدانس پیش‌بینی‌شده.
        z_pred_imag : Tensor (batch, N)
            بخش موهومی −Im(Z) پیش‌بینی‌شده.
        z_true_real : Tensor (batch, N)
            بخش حقیقی امپدانس واقعی.
        z_true_imag : Tensor (batch, N)
            بخش موهومی −Im(Z) واقعی.
        freq : Tensor (batch, N)
            آرایه فرکانس در Hz.

        Returns
        -------
        dict شامل:
            'total'      : Loss کل
            'recon'      : Loss بازسازی (MSE)
            'kk'         : جریمه K-K
            'passivity'  : جریمه Passivity
            'hf'         : جریمه High-Frequency
        """
        # ── Loss بازسازی (MSE وزن‌دار با مدول) ──────────────────────────────
        z_mod = torch.sqrt(z_true_real**2 + z_true_imag**2).clamp(min=1e-10)
        recon_real = ((z_pred_real - z_true_real) / z_mod) ** 2
        recon_imag = ((z_pred_imag - z_true_imag) / z_mod) ** 2
        L_recon = self._reduce(recon_real + recon_imag)

        # ── جریمه Passivity ──────────────────────────────────────────────────
        # Re(Z) باید ≥ 0 باشد. نقض با ReLU(-Z_real) اندازه‌گیری می‌شود.
        L_passivity = self._reduce(F.relu(-z_pred_real) ** 2)

        # ── جریمه Kramers-Kronig (تقریبی) ───────────────────────────────────
        # بررسی یکنواختی: Z_real باید با افزایش فرکانس کاهش یابد
        # و Z_imag باید یک maximum داشته باشد (قوس نیکویست).
        # تقریب: تغییرات باید smooth باشد — gradient باید کوچک باشد.
        if z_pred_real.size(1) > 1:
            dZr_df = torch.diff(z_pred_real, dim=1)
            dZi_df = torch.diff(z_pred_imag, dim=1)
            # Z_real باید با افزایش فرکانس کاهش یابد: dZr/df ≤ 0
            L_kk_real = self._reduce(F.relu(dZr_df) ** 2)
            # Z_imag باید smooth باشد
            L_kk_imag = self._reduce(dZi_df ** 2) * 0.01
            L_kk = L_kk_real + L_kk_imag
        else:
            L_kk = torch.tensor(0.0, device=z_pred_real.device)

        # ── جریمه High-Frequency ─────────────────────────────────────────────
        # در بالاترین فرکانس (آخرین نقطه): Im(Z) باید کوچک باشد
        z_imag_hf = z_pred_imag[:, -1]   # آخرین نقطه = بالاترین فرکانس
        L_hf = self._reduce(z_imag_hf ** 2)

        # ── Loss کل ──────────────────────────────────────────────────────────
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
        """اعمال reduction (mean یا sum) روی tensor."""
        if self.reduction == "mean":
            return tensor.mean()
        return tensor.sum()
