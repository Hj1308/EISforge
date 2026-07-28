"""
Ivium IDF Parser — Comprehensive support for CV, LSV, and EIS.
Author: Hoda Jafari | May 2026

Recognized Ivium methods:
    - CyclicVoltammetry → CV
    - LinearSweep / LSV → LSV (same parsing as CV)
    - EIS / FRA / Impedance → EIS

Column auto-detection:
    EIS: frequency column = widest log range + monotonic
         Z_real = positive median; Z_imag = negative median
    CV/LSV: col 0 = potential, col 1 = current (in Amperes)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from eisforge.parsers.base_parser import BaseEISParser, EISDataset

_ENCODINGS = ["latin-1", "cp1252", "utf-8"]

# Method classification
_CV_METHODS  = ("cyclic", "voltammetry", "linearsweep", "linear sweep",
                "lsv", "sweep", "chronoamperometry")
_EIS_METHODS = ("eis", "fra", "impedance", "frequency")


class AutolabIDFParser(BaseEISParser):
    """Parser for Ivium .idf files (CV, LSV, EIS)."""

    def parse(self, filepath: Path | str) -> EISDataset:
        filepath = self._resolve_path(filepath)
        content  = self._read_file(filepath)
        lines    = content.splitlines()

        metadata = self._parse_header(lines, filepath)
        method   = metadata.get("method", "").lower().replace(" ", "")

        data_start, n_points = self._find_data_block(lines)
        if data_start is None:
            raise ValueError(f"No data block in: {filepath.name}")

        raw_data = self._read_data_block(lines, data_start, n_points)
        if raw_data is None or len(raw_data) == 0:
            raise ValueError(f"No numeric data in: {filepath.name}")

        # ── Route by method ───────────────────────────────────────────────────
        if any(m in method for m in _EIS_METHODS):
            return self._parse_eis(raw_data, metadata, filepath)

        if any(m in method for m in _CV_METHODS):
            return self._parse_cv(raw_data, metadata, filepath, method)

        # ── Auto-detect when method is unknown ────────────────────────────────
        if self._looks_like_eis(raw_data):
            return self._parse_eis(raw_data, metadata, filepath)
        return self._parse_cv(raw_data, metadata, filepath, method)

    # ── EIS Parsing ───────────────────────────────────────────────────────────

    def _parse_eis(self, raw_data, metadata, filepath) -> EISDataset:
        if raw_data.shape[1] < 3:
            raise ValueError(f"EIS file needs ≥3 columns. Got {raw_data.shape[1]}.")

        freq_col   = self._find_frequency_column(raw_data)
        other_cols = [c for c in range(raw_data.shape[1]) if c != freq_col][:2]
        col_a, col_b = other_cols

        if np.median(raw_data[:, col_a]) > np.median(raw_data[:, col_b]):
            zre_col, zim_col = col_a, col_b
        else:
            zre_col, zim_col = col_b, col_a

        freq   = raw_data[:, freq_col]
        z_real = raw_data[:, zre_col]
        z_imag = raw_data[:, zim_col]

        # Convert Ivium Im(Z) (negative) → -Im(Z) (positive for capacitive)
        if np.median(z_imag) < 0:
            z_imag = -z_imag

        valid = freq > 0
        freq, z_real, z_imag = freq[valid], z_real[valid], z_imag[valid]

        metadata["data_type"]     = "EIS"
        metadata["detected_cols"] = f"freq=col{freq_col}, Z'=col{zre_col}, Z''=col{zim_col}"
        metadata["n_points"]      = len(freq)

        dataset = EISDataset(
            frequency=np.asarray(freq,   dtype=np.float64),
            z_real=np.asarray(z_real,    dtype=np.float64),
            z_imag=np.asarray(z_imag,    dtype=np.float64),
            metadata=metadata,
            source_file=filepath,
        )
        dataset.validate_shapes()
        return self._sort_by_frequency(dataset)

    # ── CV / LSV Parsing ──────────────────────────────────────────────────────

    def _parse_cv(self, raw_data, metadata, filepath, method) -> EISDataset:
        """
        Parse CV / LSV data.

        Ivium columns:
            col 0: Applied potential (V)
            col 1: Current (A) — converted to mA
            col 2: Measured potential (V) [optional]
        """
        potential = raw_data[:, 0]
        current_a = raw_data[:, 1]
        current_ma = current_a * 1000.0   # A → mA

        # Tag data type for downstream code
        if "linear" in method or "lsv" in method or "sweep" in method:
            data_type = "LSV"
        elif "chrono" in method:
            data_type = "Chronoamperometry"
        else:
            data_type = "CV"

        metadata["data_type"] = data_type
        metadata["n_points"]  = len(potential)
        metadata["current_unit_original"] = "A (auto-converted to mA)"

        return EISDataset(
            frequency=np.arange(len(potential), dtype=float),
            z_real=potential,
            z_imag=current_ma,
            metadata=metadata,
            source_file=filepath,
        )

    # ── Column Auto-Detection ─────────────────────────────────────────────────

    @staticmethod
    def _find_frequency_column(data: np.ndarray) -> int:
        """Find the frequency column = widest log range + most monotonic."""
        best_col, best_score = 0, -1.0
        for col in range(data.shape[1]):
            values = data[:, col]
            if not (values > 0).all():
                continue
            v_max, v_min = values.max(), values.min()
            if v_min < 1e-15:
                continue
            log_range = np.log10(v_max / v_min)
            diffs = np.diff(values)
            mono  = max(np.sum(diffs > 0) / len(diffs),
                        np.sum(diffs < 0) / len(diffs))
            score = log_range * mono
            if score > best_score:
                best_score, best_col = score, col
        return best_col

    @staticmethod
    def _looks_like_eis(data: np.ndarray) -> bool:
        """EIS has at least one column spanning multiple log decades."""
        if data.shape[1] < 3:
            return False
        for col in range(data.shape[1]):
            values = data[:, col]
            if (values > 0).all() and values.min() > 0:
                if np.log10(values.max() / values.min()) > 2:
                    return True
        return False

    # ── Header Parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_header(lines, filepath) -> dict:
        metadata = {
            "source_format": "Ivium IDF",
            "filename": filepath.name,
        }
        for line in lines[:200]:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                try:
                    key, val = line.split("=", 1)
                    key = key.strip().lower().replace(" ", "_")
                    val = val.strip()
                    if key and val:
                        metadata[key] = val
                except ValueError:
                    continue
        return metadata

    # ── Data Block Detection ──────────────────────────────────────────────────

    @staticmethod
    def _find_data_block(lines) -> tuple[Optional[int], Optional[int]]:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower() == "primary_data":
                for j in range(i + 1, min(i + 6, len(lines))):
                    try:
                        n = int(lines[j].strip())
                        if 10 <= n <= 1000000:
                            return j + 1, n
                    except ValueError:
                        continue
            if re.match(r"^\d+\s*$", stripped):
                try:
                    n = int(stripped)
                    if 10 <= n <= 1000000 and i + 1 < len(lines):
                        next_parts = lines[i + 1].strip().split()
                        if len(next_parts) >= 2:
                            float(next_parts[0])
                            float(next_parts[1])
                            return i + 1, n
                except (ValueError, IndexError):
                    continue
        return None, None

    @staticmethod
    def _read_data_block(lines, data_start, n_points) -> Optional[np.ndarray]:
        rows = []
        max_lines = n_points if n_points else len(lines) - data_start
        for line in lines[data_start: data_start + max_lines + 5]:
            if n_points and len(rows) >= n_points:
                break  # multi-object IDF: never read into the next data object
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower() in ("primary_data", "data objects"):
                break
            if re.match(r"^[a-zA-Z_]\w*\s*=", stripped):
                break
            parts = stripped.split()
            try:
                row = [float(p.replace(",", ".")) for p in parts]
                if len(row) >= 2:
                    rows.append(row)
            except ValueError:
                continue
        if not rows:
            return None
        max_cols = max(len(r) for r in rows)
        padded   = [r + [np.nan] * (max_cols - len(r)) for r in rows]
        return np.array(padded, dtype=np.float64)

    def _read_file(self, filepath: Path) -> str:
        for enc in _ENCODINGS:
            try:
                with filepath.open("r", encoding=enc, errors="strict") as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
        with filepath.open("r", encoding="latin-1", errors="replace") as f:
            return f.read()
