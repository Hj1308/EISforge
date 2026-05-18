"""
Autolab IDF Parser — پارسر فایل‌های Metrohm Autolab (.idf).

نویسنده: Hoda Jafari
تاریخ: May 2026

فرمت فایل .idf:
---------------
فایل‌های IDF (Instrument Data File) از نرم‌افزار NOVA
خروجی می‌گیرند. ساختار کلی:

[Header]
Date=...
Technique=EIS
...

[FRA]
Frequency  Zreal  Zimag  Zmod  Phase
...داده‌ها...

نکته: برخی نسخه‌های NOVA فرمت کمی متفاوت دارند.
این پارسر با NOVA 1.x و 2.x سازگار است.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from eisforge.parsers.base_parser import BaseEISParser, EISDataset


class AutolabIDFParser(BaseEISParser):
    """
    پارسر فایل‌های Autolab .idf از نرم‌افزار NOVA.

    سازگار با:
        - NOVA 1.x (.idf)
        - NOVA 2.x (.idf)
        - خروجی FRA32M و FRA2
    """

    # بخش‌هایی که داده EIS در آن‌هاست
    _DATA_SECTIONS = ["[FRA]", "[EIS]", "[EISDATA]", "[DATA]"]

    def parse(self, filepath: Path | str) -> EISDataset:
        """
        پارس فایل .idf و برگشت EISDataset.
        """
        filepath = self._resolve_path(filepath)
        metadata: dict = {
            "source_format": "Autolab IDF",
            "filename": filepath.name,
        }

        with filepath.open("r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        lines = content.splitlines()

        # ── استخراج metadata از header ────────────────────────────────────────
        self._parse_header(lines, metadata)

        # ── پیدا کردن بخش داده ───────────────────────────────────────────────
        data_start = self._find_data_section(lines)

        if data_start is None:
            # تلاش برای پارس مستقیم به عنوان داده عددی
            data_start = self._find_numeric_start(lines)

        if data_start is None:
            raise ValueError(
                f"بخش داده EIS در فایل پیدا نشد: {filepath}\n"
                "مطمئن شوید فایل از نوع EIS/FRA است."
            )

        # ── پارس داده‌های عددی ────────────────────────────────────────────────
        freq_list, zre_list, zim_list = [], [], []

        # تشخیص ستون‌ها از header خط داده
        col_line = lines[data_start].strip().lower()
        freq_col, zre_col, zim_col = self._detect_columns(col_line)

        for line in lines[data_start + 1:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                break
            if stripped.startswith("#") or stripped.startswith(";"):
                continue

            # جداکننده: tab، کاما، یا فاصله
            parts = re.split(r"[\t,;]+|\s{2,}", stripped)
            parts = [p.strip() for p in parts if p.strip()]

            if len(parts) <= max(freq_col, zre_col, zim_col):
                continue

            try:
                freq = float(parts[freq_col].replace(",", "."))
                zre  = float(parts[zre_col].replace(",", "."))
                zim  = float(parts[zim_col].replace(",", "."))

                if freq <= 0:
                    continue

                freq_list.append(freq)
                zre_list.append(zre)
                # Autolab: Zimag معمولاً منفی برای خازنی → نگیت می‌کنیم
                zim_list.append(-zim)

            except (ValueError, IndexError):
                continue

        if not freq_list:
            raise ValueError(f"هیچ داده عددی در فایل پیدا نشد: {filepath}")

        dataset = EISDataset(
            frequency=np.array(freq_list, dtype=np.float64),
            z_real=np.array(zre_list, dtype=np.float64),
            z_imag=np.array(zim_list, dtype=np.float64),
            metadata=metadata,
            source_file=filepath,
        )
        dataset.validate_shapes()
        return self._sort_by_frequency(dataset)

    def _parse_header(self, lines: list[str], metadata: dict) -> None:
        """استخراج metadata از header فایل IDF."""
        header_patterns = {
            r"date\s*[=:]\s*(.+)":        "date",
            r"time\s*[=:]\s*(.+)":        "time",
            r"title\s*[=:]\s*(.+)":       "title",
            r"operator\s*[=:]\s*(.+)":    "operator",
            r"e_dc\s*[=:]\s*(.+)":        "dc_potential_V",
            r"e_ac\s*[=:]\s*(.+)":        "ac_amplitude_V",
            r"f_low\s*[=:]\s*(.+)":       "f_low_Hz",
            r"f_high\s*[=:]\s*(.+)":      "f_high_Hz",
            r"n_points\s*[=:]\s*(.+)":    "n_points",
            r"technique\s*[=:]\s*(.+)":   "technique",
        }
        for line in lines[:50]:  # فقط ۵۰ خط اول
            stripped = line.strip().lower()
            for pattern, key in header_patterns.items():
                m = re.match(pattern, stripped)
                if m:
                    metadata[key] = m.group(1).strip()

    def _find_data_section(self, lines: list[str]) -> int | None:
        """پیدا کردن شروع بخش داده."""
        for i, line in enumerate(lines):
            stripped = line.strip().upper()
            if any(stripped.startswith(sec) for sec in self._DATA_SECTIONS):
                # خط بعدی header ستون‌هاست
                return i + 1
        return None

    @staticmethod
    def _find_numeric_start(lines: list[str]) -> int | None:
        """
        اگر بخش مشخصی پیدا نشد، اولین خطی که
        با عدد شروع می‌شود را پیدا کن.
        """
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    float(parts[0])
                    float(parts[1])
                    float(parts[2])
                    return i - 1  # یک خط قبل را به عنوان header بده
                except ValueError:
                    continue
        return None

    @staticmethod
    def _detect_columns(header_line: str) -> tuple[int, int, int]:
        """
        تشخیص شماره ستون‌های frequency، Zreal، Zimag.

        ستون‌های رایج در Autolab:
            freq | zreal | zimag | zmod | phase | time
        """
        aliases_freq = ["freq", "frequency", "f(hz)", "f"]
        aliases_zre  = ["zreal", "z'", "z_re", "re(z)", "zre", "real"]
        aliases_zim  = ["zimag", "z''", "z_im", "im(z)", "zim", "imag", "-im(z)"]

        parts = re.split(r"[\t,;]+|\s{2,}", header_line)
        parts = [p.strip() for p in parts if p.strip()]

        freq_col, zre_col, zim_col = 0, 1, 2  # پیش‌فرض

        for i, col in enumerate(parts):
            col_clean = col.lower().replace(" ", "").replace("(", "").replace(")", "")
            if any(a in col_clean for a in aliases_freq):
                freq_col = i
            elif any(a in col_clean for a in aliases_zre):
                zre_col = i
            elif any(a in col_clean for a in aliases_zim):
                zim_col = i

        return freq_col, zre_col, zim_col
