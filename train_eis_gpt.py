"""
Train EIS-GPT on synthetic AOR spectra — v2 (physics-informed).
===============================================================
v2 changes:
* PhysicsInformedLoss is now ACTUALLY used: predicted parameters are
  decoded into impedance spectra with a differentiable torch circuit
  simulator, and the physics penalties (Kramers-Kronig proxy,
  passivity, high-frequency limit) are applied to the reconstruction.
* --lambda-param / --lambda-physics are CLI-tunable (no magic numbers).
* Periodic checkpoints + --resume support.
* Best-model criterion combines classification accuracy AND
  parameter log10-RMSE (a model that names the right circuit but is
  10x off on R_ct should not win).
* sigma clamped to 1e-2 (one log10-decade floor) for stable NLL.

Prerequisites (one-time fixes in the package):
  Run: python apply_v2_fixes.py
  This patches:
    * aor_dataset_generator.py: circuit 3 uses 2 Warburg params
      (Wo1_R, Wo1_T) — impedance.py's Wo takes exactly 2.
    * transformer.py: MAX_PARAMS = 9 (circuit 3 has 9 parameters).

Usage
-----
Smoke test (~2 min CPU):
  python train_eis_gpt.py --samples-per-circuit 200 --epochs 5

Full training:
  python train_eis_gpt.py --samples-per-circuit 2000 --epochs 50

Resume from checkpoint:
  python train_eis_gpt.py --resume checkpoints/ckpt_ep20.pth ...

Author: Hoda Jafari | 2026
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eisforge.ml.eis_gpt.aor_dataset_generator import AORDatasetGenerator
from eisforge.ml.eis_gpt.transformer import EISForgeModel, MAX_PARAMS

# ──────────────────────────────────────────────────────────────────────
# Differentiable circuit simulator (torch, complex dtype)
# Decodes parameter vectors back into Z(omega) so the physics loss can
# constrain the *predictions*, not just the labels.
# Parameter order matches AOR_CIRCUIT_LIBRARY param_names exactly.
# ──────────────────────────────────────────────────────────────────────

def _cpe(omega, Q, n):
    """Z_CPE = 1 / (Q (j*omega)^n) using polar form (differentiable)."""
    mag = omega.clamp(min=1e-12) ** (-n) / Q
    phase = -n * (math.pi / 2)
    return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))


def _par(*zs):
    """Parallel combination of impedances."""
    y = sum(1.0 / z for z in zs)
    return 1.0 / y


def reconstruct_spectrum(params, labels, freq):
    """
    Decode predicted parameters into complex impedance spectra.

    Parameters
    ----------
    params : (B, MAX_PARAMS) — linear-space parameter values
    labels : (B,)            — circuit class labels 0..4
    freq   : (B, N)          — frequencies in Hz

    Returns
    -------
    Z : (B, N) complex64
    """
    omega = 2.0 * math.pi * freq
    B, N = freq.shape
    Z = torch.zeros(B, N, dtype=torch.complex64, device=freq.device)

    for lab in labels.unique():
        m = labels == lab
        w = omega[m]
        p = params[m]
        c = lambda i: p[:, i].unsqueeze(1)  # (B_sub, 1) broadcastable

        if lab == 0:    # R0-p(R1,CPE1)
            z = c(0) + _par(c(1) + 0j, _cpe(w, c(2), c(3)))

        elif lab == 1:  # R0-p(R1,CPE1)-W1  (semi-infinite Warburg)
            sw = c(4) / torch.sqrt(w.clamp(min=1e-12))
            z = (c(0) + _par(c(1) + 0j, _cpe(w, c(2), c(3)))
                 + torch.complex(sw, -sw))

        elif lab == 2:  # R0-p(R1,CPE1)-p(R2,CPE2)
            z = (c(0) + _par(c(1) + 0j, _cpe(w, c(2), c(3)))
                 + _par(c(4) + 0j, _cpe(w, c(5), c(6))))

        elif lab == 3:  # R0-p(R1,CPE1)-p(R2,CPE2)-Wo1 (open finite Warburg)
            x = torch.sqrt(torch.complex(torch.zeros_like(w), w * c(8)))
            x = x + 1e-6  # avoid 0/0 at omega→0
            zwo = (c(7) + 0j) / (x * torch.tanh(x))
            z = (c(0) + _par(c(1) + 0j, _cpe(w, c(2), c(3)))
                 + _par(c(4) + 0j, _cpe(w, c(5), c(6))) + zwo)

        else:           # 4: R0-p(R1,C1)-p(R2,CPE1)  (ideal capacitor)
            zc = torch.complex(torch.zeros_like(w), -1.0 / (w * c(2)))
            z = (c(0) + _par(c(1) + 0j, zc)
                 + _par(c(3) + 0j, _cpe(w, c(4), c(5))))

        Z[m] = z.to(torch.complex64)

    return Z


# ──────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────

def build_tensors(records):
    """Convert AORSyntheticRecord list into training tensors."""
    n = len(records)
    n_freq = len(records[0].frequency)
    freq, z_real, z_imag = (torch.zeros(n, n_freq) for _ in range(3))
    labels = torch.zeros(n, dtype=torch.long)
    p_target = torch.zeros(n, MAX_PARAMS)   # log10-space targets
    p_mask   = torch.zeros(n, MAX_PARAMS)   # 1 = real param, 0 = padding

    skipped = 0
    for i, r in enumerate(records):
        freq[i]   = torch.from_numpy(r.frequency.astype(np.float32))
        z_real[i] = torch.from_numpy(r.z_real.astype(np.float32))
        z_imag[i] = torch.from_numpy(r.z_imag.astype(np.float32))
        labels[i] = r.circuit_label
        values = list(r.parameters.values())
        if len(values) > MAX_PARAMS:
            skipped += 1
            values = values[:MAX_PARAMS]
        for j, v in enumerate(values):
            p_target[i, j] = float(np.log10(max(v, 1e-12)))
            p_mask[i, j]   = 1.0

    if skipped:
        print(f"WARNING: {skipped} records had >MAX_PARAMS={MAX_PARAMS} "
              f"parameters (truncated). Raise MAX_PARAMS in transformer.py.")
    return TensorDataset(freq, z_real, z_imag, labels, p_target, p_mask)


# ──────────────────────────────────────────────────────────────────────
# Loss = classification NLL + masked heteroscedastic NLL + physics
# ──────────────────────────────────────────────────────────────────────

def compute_loss(model, out, freq, z_real, z_imag, labels,
                 p_target, p_mask, lam_param, lam_physics):
    """
    Three-term loss:
      L = L_cls  +  lam_param * L_param  +  lam_physics * L_physics

    lam_physics should already include warmup scaling from the caller.
    """
    # 1) Circuit classification (log-probabilities from model)
    loss_cls = F.nll_loss(out["circuit_logprobs"], labels)

    # 2) Heteroscedastic Gaussian NLL on log10-parameters
    mu, sigma = out["param_mu"], out["param_sigma"]
    sigma = sigma.clamp(min=1e-2)   # floor = 0.01 log10-decade (~2.3%)
    nll = 0.5 * (((p_target - mu) / sigma) ** 2 + 2 * torch.log(sigma))
    loss_param = (nll * p_mask).sum() / p_mask.sum().clamp(min=1)

    # 3) Physics-informed term via differentiable circuit reconstruction
    #    mu is in log10-space; clamp before exp for numerical safety.
    pred_lin = 10.0 ** mu.clamp(-7.0, 4.0)
    Z_rec    = reconstruct_spectrum(pred_lin, labels, freq)
    zr_rec   = torch.nan_to_num(Z_rec.real,  nan=0.0).clamp(-1e6, 1e6)
    zi_rec   = torch.nan_to_num(-Z_rec.imag, nan=0.0).clamp(-1e6, 1e6)
    phys     = model.loss_fn(zr_rec, zi_rec, z_real, z_imag, freq)
    loss_physics = phys["total"]

    total = loss_cls + lam_param * loss_param + lam_physics * loss_physics
    return total, loss_cls, loss_param, loss_physics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train EIS-GPT (physics-informed transformer) on synthetic AOR spectra.")
    ap.add_argument("--samples-per-circuit", type=int, default=2000)
    ap.add_argument("--epochs",              type=int, default=50)
    ap.add_argument("--batch-size",          type=int, default=64)
    ap.add_argument("--lr",                  type=float, default=3e-4)
    ap.add_argument("--lambda-param",        type=float, default=0.5,
                    help="Weight for parameter regression loss")
    ap.add_argument("--lambda-physics",      type=float, default=0.1,
                    help="Final weight for physics-informed loss")
    ap.add_argument("--physics-warmup",      type=int,   default=5,
                    help="Epochs to linearly ramp lambda-physics from 0")
    ap.add_argument("--seed",                type=int,   default=42)
    ap.add_argument("--checkpoint-every",    type=int,   default=10,
                    help="Save checkpoint every N epochs")
    ap.add_argument("--resume",              type=str,   default=None,
                    help="Path to checkpoint .pth to resume from")
    ap.add_argument("--out",                 type=str,
                    default="eisforge/ml/eis_gpt/weights/eis_gpt_v0.2.pth")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | MAX_PARAMS: {MAX_PARAMS}")

    # 1) Generate synthetic dataset ----------------------------------------
    print(f"Generating dataset ({args.samples_per_circuit} samples/circuit)...")
    gen     = AORDatasetGenerator(
        n_samples_per_circuit=args.samples_per_circuit, seed=args.seed)
    records = gen.generate(verbose=False)
    n_classes = len({r.circuit_label for r in records})
    print(f" -> {len(records)} spectra, {n_classes} circuit classes")
    if n_classes < 5:
        print("WARNING: fewer than 5 classes found. Run apply_v2_fixes.py "
              "to patch the Wo1 parameter count in aor_dataset_generator.py.")

    dataset   = build_tensors(records)
    n_val     = int(0.2 * len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=256)

    # 2) Model + optimiser -------------------------------------------------
    model     = EISForgeModel().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # 3) Optional resume ---------------------------------------------------
    start_epoch, best_score = 1, -1e9
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch  = ck["epoch"] + 1
        best_score   = ck.get("best_score", -1e9)
        print(f"Resumed from {args.resume} (epoch {ck['epoch']})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # 4) Training loop -----------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        # Linear warmup: keep physics loss small until classification stabilises
        lam_phys = args.lambda_physics * min(
            1.0, epoch / max(args.physics_warmup, 1))

        t0, tot, n_seen = time.time(), 0.0, 0
        for freq, zr, zi, lab, pt, pm in train_loader:
            freq, zr, zi = freq.to(device), zr.to(device), zi.to(device)
            lab,  pt, pm = lab.to(device),  pt.to(device), pm.to(device)

            out  = model(freq, zr, zi)
            loss, *_ = compute_loss(model, out, freq, zr, zi, lab, pt, pm,
                                    args.lambda_param, lam_phys)

            if not torch.isfinite(loss):   # skip pathological batch
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):  # inf/nan gradients → skip step
                optimizer.zero_grad()
                continue
            optimizer.step()

            tot    += loss.item() * lab.size(0)
            n_seen += lab.size(0)

        scheduler.step()   # once per EPOCH (CosineAnnealingLR, T_max=epochs)

        # 5) Validation --------------------------------------------------------
        model.eval()
        correct, n_v, vloss, se, n_p = 0, 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for freq, zr, zi, lab, pt, pm in val_loader:
                freq, zr, zi = freq.to(device), zr.to(device), zi.to(device)
                lab,  pt, pm = lab.to(device),  pt.to(device), pm.to(device)
                out  = model(freq, zr, zi)
                loss, *_ = compute_loss(model, out, freq, zr, zi, lab, pt, pm,
                                        args.lambda_param, lam_phys)
                vloss   += loss.item() * lab.size(0)
                correct += (out["circuit_logprobs"].argmax(1) == lab).sum().item()
                se      += (((out["param_mu"] - pt) ** 2) * pm).sum().item()
                n_p     += pm.sum().item()
                n_v     += lab.size(0)

        acc   = correct / n_v
        rmse  = math.sqrt(se / max(n_p, 1))   # log10-decades
        score = acc - 0.1 * rmse              # combined best-model criterion

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {tot/n_seen:.4f} | val {vloss/n_v:.4f} | "
              f"acc {acc*100:5.1f}% | param-RMSE {rmse:.3f} dec | "
              f"lam_phys {lam_phys:.3f} | {time.time()-t0:.1f}s")

        if score > best_score:
            best_score = score
            model.save(str(out_path))

        if epoch % args.checkpoint_every == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_score": best_score,
            }, ckpt_dir / f"ckpt_ep{epoch}.pth")

    print(f"\nDone. Best combined score: {best_score:.4f}")
    print(f"Weights:     {out_path}")
    print(f"Checkpoints: {ckpt_dir}/")


if __name__ == "__main__":
    main()
