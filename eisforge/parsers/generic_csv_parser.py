from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from eisforge.parsers.base_parser import BaseEISParser, EISDataset

_FREQ_ALIASES = {"frequency", "freq", "f", "hz"}
_ZREAL_ALIASES = {"z_real", "zreal", "z'", "re(z)", "zre", "real"}
_ZIMAG_ALIASES = {"z_imag", "zimag", "z''", "-im(z)", "im(z)", "zim", "imag"}


class GenericCSVParser(BaseEISParser):
    def __init__(self, separator=None, skip_rows=0, comment_char="#"):
        self.separator = separator
        self.skip_rows = skip_rows
        self.comment_char = comment_char

    def parse(self, filepath: Path | str) -> EISDataset:
        filepath = self._resolve_path(filepath)
        try:
            df = pd.read_csv(
                filepath,
                sep=self.separator if self.separator else r"\s+|,|\t",
                engine="python",
                comment=self.comment_char,
                skiprows=self.skip_rows,
                skip_blank_lines=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}") from e

        df.columns = [str(c).strip().lower() for c in df.columns]

        freq_col = self._find_col(df, _FREQ_ALIASES, "frequency")
        real_col = self._find_col(df, _ZREAL_ALIASES, "z_real")
        imag_col = self._find_col(df, _ZIMAG_ALIASES, "z_imag")

        freq   = df[freq_col].to_numpy(dtype=np.float64)
        z_real = df[real_col].to_numpy(dtype=np.float64)
        z_imag_raw = df[imag_col].to_numpy(dtype=np.float64)
        z_imag = -z_imag_raw if np.median(z_imag_raw) < 0 else z_imag_raw

        dataset = EISDataset(
            frequency=freq, z_real=z_real, z_imag=z_imag,
            metadata={"source_format": "CSV", "filename": filepath.name},
            source_file=filepath,
        )
        dataset.validate_shapes()
        return self._sort_by_frequency(dataset)

    @staticmethod
    def _find_col(df, aliases, name):
        for col in df.columns:
            if col in aliases:
                return col
        raise ValueError(f"Column '{name}' not found. Got: {list(df.columns)}")
