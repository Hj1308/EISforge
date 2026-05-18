from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from eisforge.parsers.base_parser import BaseEISParser, EISDataset


class GamryParser(BaseEISParser):
    _TAG_RE = re.compile(r"^([A-Z]+)\t([^\t]+)(?:\t.*)?$")
    _ZCURVE_MARKER = "ZCURVE"

    def parse(self, filepath: Path | str) -> EISDataset:
        filepath = self._resolve_path(filepath)
        metadata = {"source_format": "Gamry DTA", "filename": filepath.name}
        freq_list, z_real_list, z_imag_list = [], [], []

        with filepath.open("r", encoding="utf-8", errors="replace") as fh:
            lines = list(fh)

        zcurve_idx = self._find_zcurve(lines, metadata)
        if zcurve_idx is None:
            raise ValueError(f"ZCURVE section not found: {filepath}")

        header_line = lines[zcurve_idx + 1].strip()
        col_names = header_line.split("\t")

        try:
            freq_idx  = col_names.index("Freq")
            zreal_idx = col_names.index("Zreal")
            zimag_idx = col_names.index("Zimag")
        except ValueError as e:
            raise ValueError(f"Columns not found: {col_names}") from e

        for line in lines[zcurve_idx + 3:]:
            parts = line.strip().split("\t")
            if len(parts) <= max(freq_idx, zreal_idx, zimag_idx):
                continue
            try:
                freq_list.append(float(parts[freq_idx]))
                z_real_list.append(float(parts[zreal_idx]))
                z_imag_list.append(-float(parts[zimag_idx]))
            except ValueError:
                continue

        if not freq_list:
            raise ValueError(f"No data found: {filepath}")

        dataset = EISDataset(
            frequency=np.array(freq_list, dtype=np.float64),
            z_real=np.array(z_real_list, dtype=np.float64),
            z_imag=np.array(z_imag_list, dtype=np.float64),
            metadata=metadata,
            source_file=filepath,
        )
        dataset.validate_shapes()
        return self._sort_by_frequency(dataset)

    def _find_zcurve(self, lines, metadata):
        for idx, line in enumerate(lines):
            if line.strip().startswith(self._ZCURVE_MARKER):
                return idx
            m = self._TAG_RE.match(line.strip())
            if m:
                metadata[m.group(1).lower()] = m.group(2).strip()
        return None
