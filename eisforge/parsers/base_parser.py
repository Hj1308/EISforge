from __future__ import annotations
import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class EISDataset:
    frequency: np.ndarray
    z_real: np.ndarray
    z_imag: np.ndarray
    metadata: dict = field(default_factory=dict)
    source_file: Optional[Path] = None

    @property
    def z_complex(self):
        return self.z_real - 1j * self.z_imag

    @property
    def z_modulus(self):
        return np.abs(self.z_complex)

    @property
    def phase_deg(self):
        return np.angle(self.z_complex, deg=True)

    @property
    def angular_frequency(self):
        return 2.0 * np.pi * self.frequency

    def to_dataframe(self):
        return pd.DataFrame({
            "frequency": self.frequency,
            "z_real": self.z_real,
            "z_imag": self.z_imag,
            "z_modulus": self.z_modulus,
            "phase_deg": self.phase_deg,
        })

    def validate_shapes(self):
        if not (self.frequency.shape == self.z_real.shape == self.z_imag.shape):
            raise ValueError("Shape mismatch in arrays.")

    def __repr__(self):
        return (f"EISDataset(n={len(self.frequency)}, "
                f"f=[{self.frequency.min():.2e}, {self.frequency.max():.2e}] Hz)")


class BaseEISParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, filepath: Path | str) -> EISDataset: ...

    @staticmethod
    def _resolve_path(filepath: Path | str) -> Path:
        p = Path(filepath).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
        return p

    @staticmethod
    def _sort_by_frequency(dataset: EISDataset) -> EISDataset:
        idx = np.argsort(dataset.frequency)[::-1]
        return EISDataset(
            frequency=dataset.frequency[idx],
            z_real=dataset.z_real[idx],
            z_imag=dataset.z_imag[idx],
            metadata=dataset.metadata.copy(),
            source_file=dataset.source_file,
        )