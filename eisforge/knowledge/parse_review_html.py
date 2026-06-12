"""
parse_review_html.py
--------------------
Parser for EISForge v2 HTML extraction review files.

Usage:
    from eisforge.knowledge.parse_review_html import ReviewHTMLParser

    parser = ReviewHTMLParser('eisforge/knowledge/data/reviews/review.html')
    records = parser.parse()

    # Export to JSON
    parser.export_json('output.json')

    # Access all potential values across all papers:
    for rec in records:
        print(rec['title'], '->', rec['potentials'])
"""

from __future__ import annotations

import io
import re
import glob
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    raise ImportError(
        "beautifulsoup4 is required. Install: pip install beautifulsoup4"
    )

# Force UTF-8 output on Windows CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
    source_type: str
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
# DOI extraction helper
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r'10\.\d{4,}/[^\s<>"\']+')


def _extract_doi(text: str) -> Optional[str]:
    """Extract first valid DOI from a text string."""
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    m = _DOI_RE.search(text)
    if m:
        doi = m.group(0).rstrip('.,;)')
        return doi
    return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ReviewHTMLParser:
    """
    Parse an EISForge v2 review HTML file.

    Parameters
    ----------
    path : str
        Path to the review HTML file. Supports glob patterns (e.g. 'reviews/*.html').
    """

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
        """Parse and return as plain dicts."""
        return [r.to_dict() for r in self.parse()]

    def export_json(self, output_path: str) -> None:
        """Parse and save results as a UTF-8 JSON file."""
        data = self.to_dicts()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Exported {len(data)} records to: {output_path}")

    def summary(self) -> None:
        """Print a quick summary table of all parsed papers."""
        records = self.parse()
        print(f"\n{'='*70}")
        print(f"  EISForge Review Parser — {len(records)} papers found")
        print(f"{'='*70}")
        print(f"  {'#':<4} {'Year':<6} {'DOI':<35} {'Potentials':>10} {'Tafel':>6}")
        print(f"  {'-'*65}")
        for i, r in enumerate(records, 1):
            year = r.title[:4] if r.title[:4].isdigit() else '????'
            doi_short = (r.doi[:33] + '..') if r.doi and len(r.doi) > 35 else (r.doi or 'N/A')
            n_pot = len(r.potentials)
            n_tafel = len(r.tafel_slopes)
            print(f"  {i:<4} {year:<6} {doi_short:<35} {n_pot:>10} {n_tafel:>6}")
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: str) -> List[PaperRecord]:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            soup = BeautifulSoup(fh.read(), 'html.parser')

        records: List[PaperRecord] = []

        for h3 in soup.find_all('h3'):
            title_tag = h3.find('b')
            title = title_tag.get_text(strip=True) if title_tag else h3.get_text(strip=True)

            # --- DOI extraction (multi-strategy) ---
            doi: Optional[str] = None

            # Strategy 1: look in the <p> right after <h3>
            doi_p = h3.find_next('p')
            if doi_p:
                doi = _extract_doi(doi_p.get_text())

            # Strategy 2: search in h3 raw HTML itself (some entries embed DOI in <h3>)
            if not doi:
                doi = _extract_doi(str(h3))

            # Strategy 3: scan next few siblings for DOI text
            if not doi:
                sib = h3.find_next_sibling()
                for _ in range(4):
                    if sib is None or sib.name == 'h3':
                        break
                    candidate = _extract_doi(sib.get_text())
                    if candidate:
                        doi = candidate
                        break
                    sib = sib.find_next_sibling()

            # Source type from <small> or <i> tags inside <h3>
            src_tag = h3.find('small') or h3.find('i')
            source_type = src_tag.get_text(strip=True) if src_tag else 'unknown'

            record = PaperRecord(title=title, doi=doi, source_type=source_type)

            # Walk siblings until next <h3>
            sibling = h3.find_next_sibling()
            while sibling and sibling.name != 'h3':
                if sibling.name == 'h4':
                    section_label = sibling.get_text(strip=True).lower().replace(' ', '').replace('_', '')
                    table = sibling.find_next_sibling('table')
                    if table:
                        if 'potentials' in section_label:
                            record.potentials = self._parse_numeric_table(
                                table, PotentialEntry,
                                col_keys=['value', 'unit_ref', 'page', 'flags', 'context']
                            )
                        elif 'currentdensities' in section_label:
                            record.current_densities = self._parse_numeric_table(
                                table, CurrentDensityEntry,
                                col_keys=['value', 'unit', 'page', 'flags', 'context']
                            )
                        elif 'tafelslopes' in section_label:
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
            data_cells = [c for c in cells if not c.find('input')]
            if not data_cells:
                data_cells = cells[1:]

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
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python parse_review_html.py <review.html>              # summary')
        print('  python parse_review_html.py <review.html> output.json  # export JSON')
        sys.exit(1)

    html_path = sys.argv[1]
    parser = ReviewHTMLParser(html_path)

    if len(sys.argv) >= 3:
        parser.export_json(sys.argv[2])
    else:
        parser.summary()
