"""
parse_review_html.py
--------------------
Parser for EISForge v2 HTML extraction review files.

DOI Recovery (4 strategies, in order):
  1. Plain text regex in <p> after <h3>
  2. <a href> links inside <h3> or nearby siblings (doi.org/...)
  3. Scan next 6 siblings for any DOI pattern
  4. CrossRef API lookup by title (only when fetch_missing_dois=True)

Usage:
    from eisforge.knowledge.parse_review_html import ReviewHTMLParser

    # Basic parse (no network)
    parser = ReviewHTMLParser('eisforge/knowledge/data/reviews/review.html')
    records = parser.parse()

    # With CrossRef fallback for missing DOIs (requires internet)
    records = parser.parse(fetch_missing_dois=True)

    # Export to JSON
    parser.export_json('output.json', fetch_missing_dois=True)
"""

from __future__ import annotations

import io
import re
import glob
import json
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
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
    doi_source: str = 'none'   # 'html_text' | 'html_link' | 'sibling' | 'crossref' | 'none'
    potentials: List[PotentialEntry] = field(default_factory=list)
    current_densities: List[CurrentDensityEntry] = field(default_factory=list)
    tafel_slopes: List[TafelSlopeEntry] = field(default_factory=list)
    rct_values: List[RctEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'title': self.title,
            'doi': self.doi,
            'doi_source': self.doi_source,
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
# DOI extraction helpers
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s<>"\',;)\]]+', re.IGNORECASE)
_DOI_URL_RE = re.compile(
    r'(?:https?://(?:dx\.)?doi\.org/|doi:\s*)(10\.\d{4,9}/[^\s<>"\',;)\]]+)',
    re.IGNORECASE
)


def _clean_doi(raw: str) -> str:
    """Strip trailing punctuation that regex may capture."""
    return raw.rstrip('.,;)/]\'"')


def _extract_doi_from_text(text: str) -> Optional[str]:
    """Strategy 1 & 3: Extract DOI from plain text."""
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    # Try doi.org URL pattern first (more precise)
    m = _DOI_URL_RE.search(text)
    if m:
        return _clean_doi(m.group(1))
    # Fallback to bare 10.xxxx/ pattern
    m = _DOI_RE.search(text)
    if m:
        return _clean_doi(m.group(0))
    return None


def _extract_doi_from_links(tag) -> Optional[str]:
    """Strategy 2: Extract DOI from <a href> links inside a BeautifulSoup tag."""
    for a in tag.find_all('a', href=True):
        href = a['href']
        # doi.org links
        m = _DOI_URL_RE.search(href)
        if m:
            return _clean_doi(m.group(1))
        # bare DOI in href
        m = _DOI_RE.search(href)
        if m:
            return _clean_doi(m.group(0))
        # DOI in link text
        m = _extract_doi_from_text(a.get_text())
        if m:
            return m
    return None


def _crossref_lookup(title: str, timeout: int = 6) -> Optional[str]:
    """
    Strategy 4: Query CrossRef REST API to find a DOI by title.
    Returns the best-match DOI or None.
    Rate-limit: add a small delay before calling to respect CrossRef guidelines.
    """
    if not title or len(title.strip()) < 10:
        return None
    try:
        query = urllib.parse.urlencode({'query.title': title, 'rows': '1', 'select': 'DOI,title,score'})
        url = f'https://api.crossref.org/works?{query}'
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'EISForge/0.2.0 (mailto:hoda.jaafari@gmail.com)'}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        items = data.get('message', {}).get('items', [])
        if not items:
            return None
        best = items[0]
        # Only accept if CrossRef score is reasonably confident
        if best.get('score', 0) < 30:
            return None
        doi = best.get('DOI')
        return _clean_doi(doi) if doi else None
    except Exception:
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
        Path to the review HTML file. Supports glob patterns.
    """

    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, fetch_missing_dois: bool = False) -> List[PaperRecord]:
        """
        Parse all matched files and return a flat list of PaperRecord.

        Parameters
        ----------
        fetch_missing_dois : bool
            If True, query CrossRef API for papers whose DOI could not be
            found in the HTML. Requires internet access. Default: False.
        """
        files = glob.glob(self.path) if '*' in self.path else [self.path]
        records: List[PaperRecord] = []
        for fp in sorted(files):
            records.extend(self._parse_file(fp))

        if fetch_missing_dois:
            self._recover_dois_via_crossref(records)

        return records

    def to_dicts(self, fetch_missing_dois: bool = False) -> List[dict]:
        """Parse and return as plain dicts."""
        return [r.to_dict() for r in self.parse(fetch_missing_dois=fetch_missing_dois)]

    def export_json(self, output_path: str, fetch_missing_dois: bool = False) -> None:
        """Parse and save results as a UTF-8 JSON file."""
        data = self.to_dicts(fetch_missing_dois=fetch_missing_dois)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        total = len(data)
        with_doi = sum(1 for d in data if d['doi'])
        print(f"[OK] Exported {total} records → {output_path}")
        print(f"     DOI coverage: {with_doi}/{total} ({100*with_doi//total}%)")

    def summary(self, fetch_missing_dois: bool = False) -> None:
        """Print a quick summary table of all parsed papers."""
        records = self.parse(fetch_missing_dois=fetch_missing_dois)
        with_doi = sum(1 for r in records if r.doi)
        print(f"\n{'='*72}")
        print(f"  EISForge Review Parser — {len(records)} papers")
        print(f"  DOI coverage: {with_doi}/{len(records)} "
              f"({100*with_doi//max(len(records),1)}%)")
        print(f"{'='*72}")
        print(f"  {'#':<4} {'Year':<6} {'Src':<10} {'DOI':<32} {'E':>4} {'T':>4}")
        print(f"  {'-'*68}")
        for i, r in enumerate(records, 1):
            year = r.title[:4] if r.title[:4].isdigit() else '????'
            src  = r.doi_source[:9] if r.doi_source else 'none'
            doi_s = (r.doi[:30] + '..') if r.doi and len(r.doi) > 32 else (r.doi or '—')
            print(f"  {i:<4} {year:<6} {src:<10} {doi_s:<32} {len(r.potentials):>4} {len(r.tafel_slopes):>4}")
        print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # CrossRef recovery
    # ------------------------------------------------------------------

    def _recover_dois_via_crossref(self, records: List[PaperRecord]) -> None:
        """For records with doi=None, attempt CrossRef lookup."""
        missing = [r for r in records if r.doi is None]
        if not missing:
            print("[CrossRef] All DOIs already found — no lookups needed.")
            return
        print(f"[CrossRef] Looking up {len(missing)} missing DOIs...", flush=True)
        recovered = 0
        for i, rec in enumerate(missing, 1):
            # Respect CrossRef rate limit (~1 req/sec for polite pool)
            time.sleep(1.0)
            doi = _crossref_lookup(rec.title)
            if doi:
                rec.doi = doi
                rec.doi_source = 'crossref'
                recovered += 1
                print(f"  [{i}/{len(missing)}] FOUND  {doi[:50]}  ← {rec.title[:50]}")
            else:
                print(f"  [{i}/{len(missing)}] not found  ← {rec.title[:50]}")
        print(f"[CrossRef] Recovered {recovered}/{len(missing)} missing DOIs.")

    # ------------------------------------------------------------------
    # File parser
    # ------------------------------------------------------------------

    def _parse_file(self, filepath: str) -> List[PaperRecord]:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
            soup = BeautifulSoup(fh.read(), 'html.parser')

        records: List[PaperRecord] = []

        for h3 in soup.find_all('h3'):
            title_tag = h3.find('b')
            title = title_tag.get_text(strip=True) if title_tag else h3.get_text(strip=True)

            doi: Optional[str] = None
            doi_source: str = 'none'

            # --- Strategy 1: plain text in <p> right after <h3> ---
            doi_p = h3.find_next('p')
            if doi_p:
                doi = _extract_doi_from_text(doi_p.get_text())
                if doi:
                    doi_source = 'html_text'

            # --- Strategy 2: <a href> links in <h3> itself ---
            if not doi:
                doi = _extract_doi_from_links(h3)
                if doi:
                    doi_source = 'html_link'

            # --- Strategy 2b: <a href> links in the <p> after <h3> ---
            if not doi and doi_p:
                doi = _extract_doi_from_links(doi_p)
                if doi:
                    doi_source = 'html_link'

            # --- Strategy 3: scan next 6 siblings for DOI ---
            if not doi:
                sib = h3.find_next_sibling()
                for _ in range(6):
                    if sib is None or sib.name == 'h3':
                        break
                    # check links first
                    candidate = _extract_doi_from_links(sib)
                    if not candidate:
                        candidate = _extract_doi_from_text(sib.get_text())
                    if candidate:
                        doi = candidate
                        doi_source = 'sibling'
                        break
                    sib = sib.find_next_sibling()

            # Source type from <small> or <i> tags inside <h3>
            src_tag = h3.find('small') or h3.find('i')
            source_type = src_tag.get_text(strip=True) if src_tag else 'unknown'

            record = PaperRecord(
                title=title, doi=doi,
                source_type=source_type, doi_source=doi_source
            )

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
    import argparse

    ap = argparse.ArgumentParser(description='EISForge Review HTML Parser')
    ap.add_argument('html', help='Path to review HTML file (supports glob)')
    ap.add_argument('output', nargs='?', help='Output JSON file (optional)')
    ap.add_argument('--crossref', action='store_true',
                    help='Fetch missing DOIs from CrossRef API (requires internet)')
    args = ap.parse_args()

    parser = ReviewHTMLParser(args.html)

    if args.output:
        parser.export_json(args.output, fetch_missing_dois=args.crossref)
    else:
        parser.summary(fetch_missing_dois=args.crossref)
