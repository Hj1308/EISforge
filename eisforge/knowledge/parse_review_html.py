"""
parse_review_html.py
--------------------
Parser for EISForge v2 HTML extraction review files.

Usage:
    from eisforge.knowledge.parse_review_html import ReviewHTMLParser

    parser = ReviewHTMLParser('eisforge/knowledge/data/reviews/review.html')
    records = parser.parse()
    # records is a list of dicts, one per paper

    # Access all potential values across all papers:
    for rec in records:
        print(rec['title'], '->', rec['potentials'])
"""

from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    raise ImportError(
        "beautifulsoup4 is required for parsing review HTML files. "
        "Install it with: pip install beautifulsoup4"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PotentialEntry:
    value: float
    unit_ref: str
    page: str
    flags: str
    context: str


@dataclass
class CurrentDensityEntry:
    value: float
    unit: str
    page: str
    flags: str
    context: str


@dataclass
class TafelSlopeEntry:
    value: float
    unit: str
    page: str
    flags: str
    context: str


@dataclass
class RctEntry:
    value: float
    unit: str
    page: str
    flags: str
    context: str


@dataclass
class PaperRecord:
    """One paper extracted from a review HTML file."""
    title: str
    doi: Optional[str]
    source_type: str  # e.g. 'crossref', 'pdf-metadata', 'largest-font'
    potentials: List[PotentialEntry] = field(default_factory=list)
    current_densities: List[CurrentDensityEntry] = field(default_factory=list)
    tafel_slopes: List[TafelSlopeEntry] = field(default_factory=list)
    rct_values: List[RctEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'doi': self.doi,
            'source_type': self.source_type,
            'potentials': [
                {'value': p.value, 'unit_ref': p.unit_ref,
                 'page': p.page, 'flags': p.flags, 'context': p.context}
                for p in self.potentials
            ],
            'current_densities': [
                {'value': c.value, 'unit': c.unit,
                 'page': c.page, 'flags': c.flags, 'context': c.context}
                for c in self.current_densities
            ],
            'tafel_slopes': [
                {'value': t.value, 'unit': t.unit,
                 'page': t.page, 'flags': t.flags, 'context': t.context}
                for t in self.tafel_slopes
            ],
            'rct_values': [
                {'value': r.value, 'unit': r.unit,
                 'page': r.page, 'flags': r.flags, 'context': r.context}
                for r in self.rct_values
            ],
        }


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ReviewHTMLParser:
    """
    Parse an EISForge v2 review HTML file.

    Parameters
    ----------
    path : str
        Path to the review HTML file. May be a single file path or a
        glob pattern (e.g. 'reviews/*.html').
    """

    # Regex to detect section headers like "2007 - Ye - Electrooxidation..."
    _SECTION_RE = re.compile(r'^\d{4}\s*-\s*.+')

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> List[PaperRecord]:
        """Parse all matched files and return a flat list of PaperRecord."""
        files = glob.glob(self.path) if '*' in self.path else [self.path]
        records: List[PaperRecord] = []
        for fp in sorted(files):
            records.extend(self._parse_file(fp))
        return records

    def to_dicts(self) -> List[dict]:
        """Convenience: parse and return as plain dicts."""
        return [r.to_dict() for r in self.parse()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: str) -> List[PaperRecord]:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            soup = BeautifulSoup(fh.read(), 'html.parser')

        records: List[PaperRecord] = []

        # Each paper section is wrapped in an <h3> … next <h3> block
        for h3 in soup.find_all('h3'):
            title_tag = h3.find('b')
            title = title_tag.get_text(strip=True) if title_tag else h3.get_text(strip=True)

            # DOI: look for <p> containing 'DOI' text right after <h3>
            doi: Optional[str] = None
            doi_p = h3.find_next('p')
            if doi_p:
                doi_text = doi_p.get_text()
                doi_match = re.search(r'10\.\d{4,}/\S+', doi_text)
                if doi_match:
                    doi = doi_match.group(0).rstrip('.')

            # Source type from <small> or <i> tags inside <h3>
            src_tag = h3.find('small') or h3.find('i')
            source_type = src_tag.get_text(strip=True) if src_tag else 'unknown'

            record = PaperRecord(title=title, doi=doi, source_type=source_type)

            # Walk sibling tags until next <h3>
            sibling = h3.find_next_sibling()
            while sibling and sibling.name != 'h3':
                tag_name = sibling.name

                if tag_name == 'h4':
                    section_label = sibling.get_text(strip=True).lower()
                    table = sibling.find_next_sibling('table')
                    if table:
                        if 'potentials' in section_label:
                            record.potentials = self._parse_numeric_table(
                                table, PotentialEntry,
                                col_keys=['value', 'unit_ref', 'page', 'flags', 'context']
                            )
                        elif 'currentdensities' in section_label or 'current_densities' in section_label:
                            record.current_densities = self._parse_numeric_table(
                                table, CurrentDensityEntry,
                                col_keys=['value', 'unit', 'page', 'flags', 'context']
                            )
                        elif 'tafelslopes' in section_label or 'tafel_slopes' in section_label:
                            record.tafel_slopes = self._parse_numeric_table(
                                table, TafelSlopeEntry,
                                col_keys=['value', 'unit', 'page', 'flags', 'context']
                            )
                        elif 'rct' in section_label:
                            record.rct_values = self._parse_numeric_table(
                                table, RctEntry,
                                col_keys=['value', 'unit', 'page', 'flags', 'context']
                            )

                sibling = sibling.find_next_sibling()

            records.append(record)

        return records

    @staticmethod
    def _parse_numeric_table(table_tag, dataclass_type, col_keys: list):
        """Parse a numeric data table into a list of dataclass instances."""
        rows = table_tag.find_all('tr')
        entries = []
        for row in rows[1:]:  # skip header row
            cells = row.find_all('td')
            if len(cells) < 2:
                continue
            # Skip checkbox cell (first td is always a checkbox input)
            data_cells = [c for c in cells if not c.find('input')]
            if not data_cells:
                data_cells = cells[1:]  # fallback: skip first cell

            # Try to read value (first real data cell)
            try:
                raw_val = data_cells[0].get_text(strip=True)
                numeric_val = float(raw_val)
            except (ValueError, IndexError):
                continue

            kwargs = {'value': numeric_val}
            for i, key in enumerate(col_keys[1:], start=1):
                try:
                    kwargs[key] = data_cells[i].get_text(strip=True)
                except IndexError:
                    kwargs[key] = ''

            try:
                entries.append(dataclass_type(**kwargs))
            except TypeError:
                pass

        return entries


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print('Usage: python parse_review_html.py <path_to_review.html>')
        sys.exit(1)

    html_path = sys.argv[1]
    parser = ReviewHTMLParser(html_path)
    records = parser.to_dicts()
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print(f'\n>>> Total papers parsed: {len(records)}', file=sys.stderr)
