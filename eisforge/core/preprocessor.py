"""
EIS Data Preprocessor — Explicit data cleaning methods (ZView-style).
Author: Hoda Jafari | May 2026

Four independent cleaning operations:
    1. remove_inductive_artifacts — remove high-freq points where Z'' < 0
    2. crop_frequencies          — keep only points in [f_min, f_max]
    3. remove_z_jumps            — remove points with sudden jumps in Z' or Z''
    4. drop_specific_frequency   — remove specific frequencies (e.g. 50/60 Hz mains)

Each method returns a NEW EISDataset (immutable). Chain them as needed.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from eisforge.parsers.base_parser import EISDataset

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Static methods for cleaning EIS data before fitting."""

    # ── 1. Remove inductive artifacts ─────────────────────────────────────────

    @staticmethod
    def remove_inductive_artifacts(
        dataset: EISDataset,
        verbose: bool = True,
    ) -> EISDataset:
        """
        Remove high-frequency inductive points (where -Z'' < 0).

        Cable inductance at high frequencies causes Z'' to become positive
        (inductive). These points are instrument artifacts.
        """
        keep = dataset.z_imag >= 0
        n_removed = int(np.sum(~keep))

        if verbose and n_removed > 0:
            logger.info(f"Removed {n_removed} inductive artifact points (Z'' < 0)")

        return EISDataset(
            frequency=dataset.frequency[keep],
            z_real=dataset.z_real[keep],
            z_imag=dataset.z_imag[keep],
            metadata={**dataset.metadata, "preprocessing_inductive": n_removed},
            source_file=dataset.source_file,
        )

    # ── 2. Crop frequencies ───────────────────────────────────────────────────

    @staticmethod
    def crop_frequencies(
        dataset: EISDataset,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        verbose: bool = True,
    ) -> EISDataset:
        """Keep only points where f_min <= frequency <= f_max."""
        keep = np.ones(len(dataset.frequency), dtype=bool)
        if f_min is not None:
            keep &= dataset.frequency >= f_min
        if f_max is not None:
            keep &= dataset.frequency <= f_max

        n_removed = int(np.sum(~keep))
        if verbose and n_removed > 0:
            logger.info(
                f"Cropped {n_removed} points outside "
                f"[{f_min or '-inf'}, {f_max or 'inf'}] Hz"
            )

        return EISDataset(
            frequency=dataset.frequency[keep],
            z_real=dataset.z_real[keep],
            z_imag=dataset.z_imag[keep],
            metadata={
                **dataset.metadata,
                "preprocessing_cropped": n_removed,
                "f_min_applied": f_min,
                "f_max_applied": f_max,
            },
            source_file=dataset.source_file,
        )

    # ── 3. Remove Z jumps (per-axis check) ───────────────────────────────────

    @staticmethod
    def remove_z_jumps(
        dataset: EISDataset,
        threshold_pct: float = 20.0,
        verbose: bool = True,
    ) -> EISDataset:
        """
        Remove points with sudden jumps in Z' OR Z''.

        Per-axis detection: a point is flagged if EITHER Z' or Z'' jumps
        more than threshold_pct from BOTH neighbors. Catches outliers
        that spike on either axis of the Nyquist plot.
        """
        n = len(dataset.frequency)
        if n < 5:
            return dataset

        order = np.argsort(dataset.frequency)[::-1]
        z_re  = dataset.z_real[order]
        z_im  = dataset.z_imag[order]

        keep_sorted = np.ones(n, dtype=bool)
        eps = 1e-9

        # Interior points
        for i in range(1, n - 1):
            re_diff_prev = abs(z_re[i] - z_re[i-1]) / (abs(z_re[i-1]) + eps) * 100
            re_diff_next = abs(z_re[i] - z_re[i+1]) / (abs(z_re[i+1]) + eps) * 100
            im_diff_prev = abs(z_im[i] - z_im[i-1]) / (abs(z_im[i-1]) + eps) * 100
            im_diff_next = abs(z_im[i] - z_im[i+1]) / (abs(z_im[i+1]) + eps) * 100

            re_outlier = (re_diff_prev > threshold_pct and re_diff_next > threshold_pct)
            im_outlier = (im_diff_prev > threshold_pct and im_diff_next > threshold_pct)

            if re_outlier or im_outlier:
                keep_sorted[i] = False

        # Edge points (2× threshold)
        edge_threshold = threshold_pct * 2
        re_diff = abs(z_re[0] - z_re[1]) / (abs(z_re[1]) + eps) * 100
        im_diff = abs(z_im[0] - z_im[1]) / (abs(z_im[1]) + eps) * 100
        if re_diff > edge_threshold or im_diff > edge_threshold:
            keep_sorted[0] = False

        re_diff = abs(z_re[-1] - z_re[-2]) / (abs(z_re[-2]) + eps) * 100
        im_diff = abs(z_im[-1] - z_im[-2]) / (abs(z_im[-2]) + eps) * 100
        if re_diff > edge_threshold or im_diff > edge_threshold:
            keep_sorted[-1] = False

        # Map back to original order
        keep = np.ones(n, dtype=bool)
        for k, original_idx in enumerate(order):
            keep[original_idx] = keep_sorted[k]

        n_removed = int(np.sum(~keep))
        if verbose and n_removed > 0:
            logger.info(
                f"Removed {n_removed} Z-jump outliers (threshold {threshold_pct}%)"
            )

        return EISDataset(
            frequency=dataset.frequency[keep],
            z_real=dataset.z_real[keep],
            z_imag=dataset.z_imag[keep],
            metadata={
                **dataset.metadata,
                "preprocessing_z_jumps": n_removed,
                "z_jump_threshold_pct": threshold_pct,
            },
            source_file=dataset.source_file,
        )

    # ── 4. Drop specific frequency (e.g. 50/60 Hz mains) ─────────────────────

    @staticmethod
    def drop_specific_frequency(
        dataset: EISDataset,
        target_freq: Union[float, list],
        tolerance_hz: float = 0.5,
        verbose: bool = True,
    ) -> EISDataset:
        """
        Remove specific frequencies that are known to be noisy.

        Common targets:
            - 50 Hz : mains frequency in Europe/Asia
            - 60 Hz : mains frequency in North America
            - 100 Hz, 120 Hz : second harmonics of mains
            - Custom frequencies known to have instrument resonance

        Parameters
        ----------
        target_freq : float or list
            Single frequency or list of frequencies (Hz) to remove.
        tolerance_hz : float
            Half-width of the rejection band (default ±0.5 Hz).
            E.g. target_freq=50, tolerance=0.5 removes [49.5, 50.5] Hz.

        Examples
        --------
        >>> # Remove 50 Hz mains noise
        >>> clean = DataPreprocessor.drop_specific_frequency(ds, 50.0)
        >>>
        >>> # Remove multiple frequencies
        >>> clean = DataPreprocessor.drop_specific_frequency(
        ...     ds, target_freq=[50.0, 100.0, 150.0], tolerance_hz=1.0
        ... )
        """
        targets = [target_freq] if isinstance(target_freq, (int, float)) else list(target_freq)

        # Start with keeping all points
        keep = np.ones(len(dataset.frequency), dtype=bool)

        # Remove points within tolerance of each target
        for t in targets:
            keep &= np.abs(dataset.frequency - t) > tolerance_hz

        n_removed = int(np.sum(~keep))

        if verbose and n_removed > 0:
            tlist = ", ".join(f"{t}±{tolerance_hz} Hz" for t in targets)
            logger.info(f"Removed {n_removed} points at frequencies: {tlist}")

        return EISDataset(
            frequency=dataset.frequency[keep],
            z_real=dataset.z_real[keep],
            z_imag=dataset.z_imag[keep],
            metadata={
                **dataset.metadata,
                "preprocessing_dropped_freqs": targets,
                "preprocessing_dropped_count": n_removed,
            },
            source_file=dataset.source_file,
        )

    # ── 5. Full pipeline ──────────────────────────────────────────────────────

    @staticmethod
    def clean_pipeline(
        dataset: EISDataset,
        f_min: Optional[float] = 0.01,
        f_max: Optional[float] = None,
        remove_inductive: bool = True,
        remove_jumps: bool = True,
        jump_threshold_pct: float = 20.0,
        drop_mains: bool = False,
        mains_freq: float = 50.0,
        verbose: bool = True,
    ) -> EISDataset:
        """
        Apply all cleaning steps in sequence.

        Default pipeline:
            1. Remove inductive artifacts (Z'' < 0)
            2. Crop frequencies (f_min=0.01 Hz)
            3. Remove Z-jumps (per-axis, 20%)
            4. Optional: drop mains frequency

        Parameters
        ----------
        drop_mains : bool
            If True, remove mains-frequency noise.
        mains_freq : float
            Mains frequency in Hz (50 for EU/Asia, 60 for US).
        """
        n_original = len(dataset.frequency)
        cleaned    = dataset

        if remove_inductive:
            cleaned = DataPreprocessor.remove_inductive_artifacts(cleaned, verbose)

        if f_min is not None or f_max is not None:
            cleaned = DataPreprocessor.crop_frequencies(cleaned, f_min, f_max, verbose)

        if remove_jumps:
            cleaned = DataPreprocessor.remove_z_jumps(cleaned, jump_threshold_pct, verbose)

        if drop_mains:
            cleaned = DataPreprocessor.drop_specific_frequency(cleaned, mains_freq, verbose=verbose)

        n_final = len(cleaned.frequency)
        if verbose:
            logger.info(
                f"Cleaning pipeline: {n_original} → {n_final} points "
                f"({n_original - n_final} removed total)"
            )

        return cleaned
