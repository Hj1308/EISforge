#!/usr/bin/env python3
"""
EISForge PDF Extraction Pipeline  v2
======================================
Extracts electrochemical performance data from AOR / fuel-cell PDF papers.

Usage
-----
pip install pdfplumber PyYAML tqdm

python extract_papers.py --input "D:\\Articles and Seminar\\Alcohol oxidation" \
                         --out   "D:\\Articles and Seminar\\extracted"

Output
------
  extracted_aor.yaml           AOR records with values + context snippets
  extracted_fuelcell.yaml      Fuel-cell records
  extracted_unclassified.yaml  Unclassified / error papers
  extraction_report.txt        Per-file summary
  review.html                  Visual review: every number beside its source sentence

v2 improvements over v1
------------------------
* Context snippet (±150 chars) + page number stored for every extracted value
* Unit-aware R_ct/R_s: Ω / kΩ / Ω·cm² all normalised, unit kept in record
* Per-entry reference-electrode offset stored (sat.KCl=0.197 V, 3 M NaCl=0.209 V …)
* Context-scoring: values near "our", "this work", "onset", "catalyst" ranked first
* Table extraction via pdfplumber — captures structured data rows
* HTML review report for fast manual curation

ALL values are approximate; review before publication.

Author: Hoda Jafari | EISForge | 2026
"""

from __future__ import annotations

import argparse
import html as html_mod
import re
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    sys.exit("ERROR: pdfplumber not found.  Run:  pip install pdfplumber")

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not found.  Run:  pip install PyYAML")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
# Reference electrode offsets vs RHE  (V, at pH 13 / 0.1 M KOH unless noted)
# Source: BASi Electrochemistry Reference Electrode Guide
# ─────────────────────────────────────────────────────────────────────────────
REFERENCE_OFFSETS = {
    "RHE":          0.000,
    "NHE":          0.000,   # same scale; context determines pH correction
    "SHE":          0.000,
    "Ag/AgCl":      0.197,   # default: saturated KCl
    "Ag/AgCl_satKCl":  0.197,
    "Ag/AgCl_3M_NaCl": 0.209,   # 3 M NaCl — corrected from v1 (was 0.197)
    "Ag/AgCl_3.5M":    0.205,
    "SCE":          0.241,
    "Hg/HgO":       0.098,   # in 0.1 M KOH
    "Hg/HgO_1M_KOH":  0.140,
}

# ─────────────────────────────────────────────────────────────────────────────
# Classification keywords
# ─────────────────────────────────────────────────────────────────────────────
AOR_KEYWORDS = [
    "alcohol oxidation", "methanol oxidation", "ethanol oxidation",
    "2-propanol", "isopropanol", "glycerol oxidation",
    "alcohol electro-oxidation", "anode catalyst", "AOR",
    "formate", "acetate", "acetaldehyde", "acetone",
    "ethylene glycol", "glucose oxidation",
]

FUELCELL_KEYWORDS = [
    "fuel cell", "PEMFC", "AEMFC", "DMFC", "oxygen reduction",
    "ORR", "membrane electrode", "MEA", "cathode catalyst",
    "power density", "polarisation curve", "polarization curve",
    "open circuit voltage", "OCV",
]

# Words that suggest a value belongs to *this* paper's catalyst, not a reference
OWN_WORK_SIGNALS = [
    "our", "this work", "present", "as-prepared", "synthesized", "obtained",
    "achieved", "exhibited", "showed", "we report", "herein",
    "the catalyst", "the electrode", "onset potential", "peak potential",
]

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

RE_POTENTIAL = re.compile(
    r"([\u2212\-]?\d+\.?\d*)\s*V"
    r"(?:\s*(?:vs\.?|versus)?\s*(RHE|NHE|SHE|Ag/AgCl|SCE|Hg/HgO))?",
    re.IGNORECASE,
)

RE_CURRENT = re.compile(
    r"([\u2212\-]?\d+\.?\d*)\s*"
    r"(mA\s*(?:cm[\-\u22122]?|cm\u00b2|mg[\-\u22121]?|g[\-\u22121]?)"
    r"|A\s*(?:g[\-\u22121]?|m[\-\u22122]?))",
    re.IGNORECASE,
)

RE_TAFEL = re.compile(
    r"(\d+\.?\d*)\s*mV\s*(?:dec[\-\u22121]?|per\s*dec|/dec)",
    re.IGNORECASE,
)

RE_IF_IB = re.compile(
    r"(?:I_?f\s*/\s*I_?b|forward[\s/]backward\s*ratio)\s*[=:]?\s*([\d\.]+)",
    re.IGNORECASE,
)

RE_FE = re.compile(
    r"(?:faradaic\s*efficiency|FE)\s*[=:of]*\s*([\d\.]+)\s*%",
    re.IGNORECASE,
)

RE_POWER = re.compile(
    r"([\d\.]+)\s*mW\s*(?:cm[\-\u22122]?|cm\u00b2)",
    re.IGNORECASE,
)

# Unit-aware EIS: captures value AND unit (Ω / kΩ / Ω·cm²)
RE_EIS = re.compile(
    r"(R_?(?:ct|s|ohm|sol|int|charge|series))\s*[=:]\s*([\d\.]+)\s*"
    r"(k?\u03a9(?:\s*cm[\u00b2\-2]?)?|kohm|ohm)",
    re.IGNORECASE,
)

RE_ELECTROLYTE = re.compile(
    r"(\d+\.?\d*)\s*M\s*(KOH|NaOH|H2SO4|HClO4|NaHCO3|PBS|phosphate)",
    re.IGNORECASE,
)

RE_SCAN_RATE = re.compile(
    r"(\d+\.?\d*)\s*mV\s*s[\-\u22121]",
    re.IGNORECASE,
)

# Detect reference electrode type in surrounding text
RE_REF_TYPE = re.compile(
    r"Ag/AgCl\s*\(?\s*(\d+\.?\d*)\s*M\s*(KCl|NaCl)|"
    r"(saturated|sat\.?)\s*(?:KCl|calomel)|"
    r"(SCE|Hg/HgO|RHE|NHE|SHE)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(s: str) -> float | None:
    try:
        return float(str(s).replace("\u2212", "-"))
    except (ValueError, TypeError):
        return None


def _context(full_text: str, pos: int, window: int = 150) -> str:
    """Return ±window characters around position pos."""
    start = max(0, pos - window)
    end   = min(len(full_text), pos + window)
    snippet = full_text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(full_text):
        snippet = snippet + "…"
    return snippet


def _own_work_score(snippet: str) -> int:
    """Count how many 'own-work' signals appear in the context snippet."""
    sl = snippet.lower()
    return sum(sig in sl for sig in OWN_WORK_SIGNALS)


def _detect_reference(text_window: str) -> tuple[str, float]:
    """
    Try to identify the reference electrode from surrounding text.
    Returns (label, offset_V).
    """
    m = RE_REF_TYPE.search(text_window)
    if not m:
        return "unknown", 0.0
    full = m.group(0).lower()
    if "3 m nacl" in full or "3m nacl" in full:
        return "Ag/AgCl_3M_NaCl", REFERENCE_OFFSETS["Ag/AgCl_3M_NaCl"]
    if "3.5" in full:
        return "Ag/AgCl_3.5M", REFERENCE_OFFSETS["Ag/AgCl_3.5M"]
    if "kcl" in full or "saturated" in full or "sat" in full:
        return "Ag/AgCl_satKCl", REFERENCE_OFFSETS["Ag/AgCl_satKCl"]
    if "sce" in full:
        return "SCE", REFERENCE_OFFSETS["SCE"]
    if "hg/hgo" in full:
        return "Hg/HgO", REFERENCE_OFFSETS["Hg/HgO"]
    if "rhe" in full:
        return "RHE", REFERENCE_OFFSETS["RHE"]
    return "unknown", 0.0


def _normalise_eis(value: float, unit_str: str) -> tuple[float, str]:
    """Normalise EIS resistance to Ω. Returns (value_ohm, original_unit)."""
    u = unit_str.lower().replace(" ", "")
    if u.startswith("k"):          # kΩ
        return value * 1000, unit_str
    return value, unit_str


def classify_paper(text: str) -> str:
    tl = text.lower()
    aor_score = sum(kw.lower() in tl for kw in AOR_KEYWORDS)
    fc_score  = sum(kw.lower() in tl for kw in FUELCELL_KEYWORDS)
    if aor_score == 0 and fc_score == 0:
        return "unclassified"
    return "aor" if aor_score >= fc_score else "fuelcell"


def extract_text_and_pages(pdf_path: Path, max_pages: int = 14
                            ) -> list[tuple[int, str]]:
    """
    Returns list of (page_number, page_text) for first max_pages pages.
    Page numbers are 1-indexed.
    """
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], 1):
                t = page.extract_text() or ""
                pages.append((i, t))
    except Exception as exc:
        return [(-1, f"__ERROR__{exc}")]
    return pages


def extract_tables(pdf_path: Path, max_pages: int = 14) -> list[dict]:
    """
    Extract structured tables from PDF pages.
    Returns list of {page, headers, rows} dicts.
    """
    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], 1):
                for tbl in page.extract_tables():
                    if not tbl or len(tbl) < 2:
                        continue
                    headers = [str(c).strip() if c else "" for c in tbl[0]]
                    rows = []
                    for row in tbl[1:]:
                        rows.append([str(c).strip() if c else "" for c in row])
                    tables_found.append({
                        "page": i,
                        "headers": headers,
                        "rows": rows[:20],   # cap rows
                    })
    except Exception:
        pass
    return tables_found


def extract_title_authors(pages: list[tuple[int, str]]) -> tuple[str, str]:
    """
    Improved title/author extraction: skip short lines and journal headers.
    """
    journal_signals = [
        "journal", "elsevier", "wiley", "springer", "acs", "rsc",
        "doi:", "received", "accepted", "published", "copyright",
        "vol.", "volume", "issue", "pages",
    ]
    candidates = []
    for _, text in pages[:2]:
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 10:
                continue
            if any(sig in line.lower() for sig in journal_signals):
                continue
            candidates.append(line)

    title   = candidates[0][:250] if candidates else "Unknown"
    authors = candidates[1][:250] if len(candidates) > 1 else "Unknown"
    return title, authors


# ─────────────────────────────────────────────────────────────────────────────
# Field extractors  (v2: context-scored, with snippet + page)
# ─────────────────────────────────────────────────────────────────────────────

def _make_hit(value, unit: str, snippet: str, page: int,
              score: int, extra: dict | None = None) -> dict:
    hit = {
        "value": value,
        "unit": unit,
        "page": page,
        "own_work_score": score,
        "context_snippet": snippet,
    }
    if extra:
        hit.update(extra)
    return hit


def extract_aor_fields(pages: list[tuple[int, str]]) -> dict:
    full_text = "\n".join(t for _, t in pages)
    page_map  = {}  # char_offset → page number
    offset = 0
    for pno, t in pages:
        for i in range(len(t)):
            page_map[offset + i] = pno
        offset += len(t) + 1  # +1 for \n

    def page_at(pos: int) -> int:
        return page_map.get(pos, 0)

    fields: dict = {}

    # ── Potentials ──────────────────────────────────────────────────────────
    pot_hits = []
    for m in RE_POTENTIAL.finditer(full_text):
        val = _safe_float(m.group(1))
        if val is None or not (-3.0 < val < 3.0):
            continue
        snip  = _context(full_text, m.start())
        score = _own_work_score(snip)
        ref_raw = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        ref_label, offset_v = _detect_reference(snip)
        if ref_raw:
            ref_label = ref_raw
        pot_hits.append(_make_hit(
            value=val, unit="V",
            snippet=snip, page=page_at(m.start()), score=score,
            extra={"reference_electrode": ref_label,
                   "offset_to_RHE_V": REFERENCE_OFFSETS.get(ref_label, None)},
        ))
    if pot_hits:
        pot_hits.sort(key=lambda h: -h["own_work_score"])
        fields["potentials"] = pot_hits[:6]

    # ── Current densities ───────────────────────────────────────────────────
    cur_hits = []
    for m in RE_CURRENT.finditer(full_text):
        val = _safe_float(m.group(1))
        if val is None or val <= 0:
            continue
        snip  = _context(full_text, m.start())
        score = _own_work_score(snip)
        cur_hits.append(_make_hit(
            value=val, unit=m.group(2).strip(),
            snippet=snip, page=page_at(m.start()), score=score,
        ))
    if cur_hits:
        cur_hits.sort(key=lambda h: -h["own_work_score"])
        fields["current_densities"] = cur_hits[:6]

    # ── Tafel slope ──────────────────────────────────────────────────────────
    tafel_hits = []
    for m in RE_TAFEL.finditer(full_text):
        val = _safe_float(m.group(1))
        if val is None or not (20 < val < 300):
            continue
        snip  = _context(full_text, m.start())
        score = _own_work_score(snip)
        tafel_hits.append(_make_hit(
            value=val, unit="mV/dec",
            snippet=snip, page=page_at(m.start()), score=score,
        ))
    if tafel_hits:
        tafel_hits.sort(key=lambda h: -h["own_work_score"])
        fields["tafel_slope"] = tafel_hits[0]
        if len(tafel_hits) > 1:
            fields["tafel_slope_all"] = tafel_hits

    # ── If/Ib ────────────────────────────────────────────────────────────────
    for m in RE_IF_IB.finditer(full_text):
        val = _safe_float(m.group(1))
        if val and 0.1 < val < 20:
            snip = _context(full_text, m.start())
            fields["If_Ib_ratio"] = _make_hit(
                value=val, unit="dimensionless",
                snippet=snip, page=page_at(m.start()),
                score=_own_work_score(snip),
            )
            break

    # ── Faradaic efficiency ──────────────────────────────────────────────────
    for m in RE_FE.finditer(full_text):
        val = _safe_float(m.group(1))
        if val and 0 < val <= 100:
            snip = _context(full_text, m.start())
            fields["faradaic_efficiency"] = _make_hit(
                value=val, unit="%",
                snippet=snip, page=page_at(m.start()),
                score=_own_work_score(snip),
            )
            break

    # ── Electrolyte ──────────────────────────────────────────────────────────
    m = RE_ELECTROLYTE.search(full_text)
    if m:
        fields["electrolyte"] = f"{m.group(1)} M {m.group(2)}"

    # ── Scan rate ────────────────────────────────────────────────────────────
    m = RE_SCAN_RATE.search(full_text)
    if m:
        fields["scan_rate_mV_s"] = _safe_float(m.group(1))

    return fields


def extract_fuelcell_fields(pages: list[tuple[int, str]]) -> dict:
    full_text = "\n".join(t for _, t in pages)
    page_map  = {}
    offset = 0
    for pno, t in pages:
        for i in range(len(t)):
            page_map[offset + i] = pno
        offset += len(t) + 1

    def page_at(pos: int) -> int:
        return page_map.get(pos, 0)

    fields: dict = {}

    # ── Peak power density ───────────────────────────────────────────────────
    power_hits = []
    for m in RE_POWER.finditer(full_text):
        val = _safe_float(m.group(1))
        if val and val > 0:
            snip  = _context(full_text, m.start())
            score = _own_work_score(snip)
            power_hits.append(_make_hit(
                value=val, unit="mW/cm²",
                snippet=snip, page=page_at(m.start()), score=score,
            ))
    if power_hits:
        power_hits.sort(key=lambda h: -h["own_work_score"])
        fields["peak_power_density"] = power_hits[0]

    # ── EIS — unit-aware ─────────────────────────────────────────────────────
    eis_hits: dict[str, list] = defaultdict(list)
    for m in RE_EIS.finditer(full_text):
        key  = m.group(1).lower().replace(" ", "_")
        val  = _safe_float(m.group(2))
        unit = m.group(3)
        if val is None:
            continue
        val_ohm, orig_unit = _normalise_eis(val, unit)
        snip  = _context(full_text, m.start())
        score = _own_work_score(snip)
        eis_hits[key].append(_make_hit(
            value=val_ohm, unit="Ω",
            snippet=snip, page=page_at(m.start()), score=score,
            extra={"original_value": val, "original_unit": orig_unit},
        ))
    if eis_hits:
        fields["eis_parameters"] = {
            k: sorted(v, key=lambda h: -h["own_work_score"])[0]
            for k, v in eis_hits.items()
        }

    # ── Electrolyte / scan rate ──────────────────────────────────────────────
    m = RE_ELECTROLYTE.search(full_text)
    if m:
        fields["electrolyte"] = f"{m.group(1)} M {m.group(2)}"

    m = RE_SCAN_RATE.search(full_text)
    if m:
        fields["scan_rate_mV_s"] = _safe_float(m.group(1))

    # ── Current densities ────────────────────────────────────────────────────
    cur_hits = []
    for m in RE_CURRENT.finditer(full_text):
        val = _safe_float(m.group(1))
        if val and val > 0:
            snip  = _context(full_text, m.start())
            score = _own_work_score(snip)
            cur_hits.append(_make_hit(
                value=val, unit=m.group(2).strip(),
                snippet=snip, page=page_at(m.start()), score=score,
            ))
    if cur_hits:
        cur_hits.sort(key=lambda h: -h["own_work_score"])
        fields["current_densities"] = cur_hits[:4]

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# HTML review report generator
# ─────────────────────────────────────────────────────────────────────────────

def _hit_rows(field_name: str, hit_or_list) -> str:
    """Render one or more hits as HTML table rows."""
    if isinstance(hit_or_list, dict) and "value" in hit_or_list:
        hits = [hit_or_list]
    elif isinstance(hit_or_list, list):
        hits = [h for h in hit_or_list if isinstance(h, dict) and "value" in h]
    else:
        return ""

    rows = []
    for h in hits:
        score_color = "#2a9d8f" if h.get("own_work_score", 0) >= 2 else (
                      "#e9c46a" if h.get("own_work_score", 0) == 1 else "#e76f51")
        snippet_esc = html_mod.escape(h.get("context_snippet", ""))
        ref = h.get("reference_electrode", "")
        extra_info = f" | {ref}" if ref and ref != "unknown" else ""
        rows.append(f"""
        <tr>
          <td style="color:{score_color};font-weight:bold">{html_mod.escape(field_name)}</td>
          <td>{h.get('value','')}&nbsp;{html_mod.escape(h.get('unit',''))}{html_mod.escape(extra_info)}</td>
          <td>p.{h.get('page','?')}</td>
          <td style="font-size:0.82em;color:#555">{snippet_esc}</td>
        </tr>""")
    return "\n".join(rows)


def build_html_report(records: list[dict], label: str) -> str:
    sections = []
    for rec in records:
        fname = html_mod.escape(rec.get("file", ""))
        title = html_mod.escape(rec.get("title_extracted", "")[:120])
        rows_html = ""
        for key, val in rec.items():
            if key in ("file", "title_extracted", "authors_extracted",
                       "extraction_note", "approximate", "electrolyte",
                       "scan_rate_mV_s"):
                continue
            rows_html += _hit_rows(key, val)

        sections.append(f"""
  <details>
    <summary><strong>{fname}</strong> &mdash; <em>{title}</em></summary>
    <p style="font-size:0.85em;color:#666">
      Electrolyte: {html_mod.escape(str(rec.get('electrolyte','?')))} &nbsp;|&nbsp;
      Scan rate: {rec.get('scan_rate_mV_s','?')} mV/s
    </p>
    <table>
      <thead><tr>
        <th>Field</th><th>Value</th><th>Page</th><th>Context (±150 chars)</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </details>""")

    body = "\n".join(sections) if sections else "<p>No records.</p>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EISForge Review — {label}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
          padding: 0 1rem; background:#fafafa; color:#222; }}
  h1   {{ font-size:1.4rem; margin-bottom:.25rem }}
  p.sub {{ color:#666; font-size:.9rem; margin-top:0 }}
  details {{ border:1px solid #ddd; border-radius:6px; margin:.6rem 0;
             background:#fff; padding:.5rem 1rem; }}
  summary  {{ cursor:pointer; font-size:1rem; padding:.3rem 0; }}
  table    {{ border-collapse:collapse; width:100%; margin-top:.5rem; font-size:.88rem; }}
  th       {{ background:#f0f0f0; text-align:left; padding:4px 8px; }}
  td       {{ border-top:1px solid #eee; padding:4px 8px; vertical-align:top; }}
  td:last-child {{ font-family:monospace; word-break:break-word; }}
  .legend  {{ font-size:.82rem; margin:.5rem 0 1rem; }}
  .green   {{ color:#2a9d8f }} .yellow {{ color:#e9c46a }} .red {{ color:#e76f51 }}
</style>
</head>
<body>
<h1>EISForge Extraction Review &mdash; {html_mod.escape(label)}</h1>
<p class="sub">Generated: {datetime.now():%Y-%m-%d %H:%M} &nbsp;|&nbsp;
Records: {len(records)}</p>
<p class="legend">
  Score colour: <span class="green">■ high (own-work signals ≥2)</span>
  &nbsp;<span class="yellow">■ medium (1)</span>
  &nbsp;<span class="red">■ low (0) — may be from a reference</span>
</p>
{body}
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(input_dir: Path, max_pages: int = 14):
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDF files found in {input_dir}")

    aor_records, fc_records, unclassified, report_lines = [], [], [], []
    report_lines += [
        f"EISForge Extraction Report v2 — {datetime.now():%Y-%m-%d %H:%M}",
        f"Input : {input_dir}",
        f"PDFs  : {len(pdf_files)}",
        "-" * 70,
    ]

    iterator = tqdm(pdf_files, desc="Extracting") if HAS_TQDM else pdf_files

    for i, pdf_path in enumerate(iterator, 1):
        pages = extract_text_and_pages(pdf_path, max_pages)

        # Error check
        if pages and pages[0][0] == -1 and pages[0][1].startswith("__ERROR__"):
            report_lines.append(f"[{i:03d}] ERROR        {pdf_path.name}")
            unclassified.append({"file": pdf_path.name,
                                  "error": pages[0][1][9:]})
            continue

        full_text = "\n".join(t for _, t in pages)
        category  = classify_paper(full_text)
        title, authors = extract_title_authors(pages)
        tables = extract_tables(pdf_path, max_pages)

        base = {
            "file":              pdf_path.name,
            "title_extracted":   title,
            "authors_extracted": authors,
            "extraction_note":   "auto-extracted v2 — review manually before use",
            "approximate":       True,
        }
        if tables:
            base["tables_found"] = len(tables)
            base["tables"] = tables[:3]   # keep first 3 tables

        if category == "aor":
            fields = extract_aor_fields(pages)
            aor_records.append({**base, **fields})
            report_lines.append(
                f"[{i:03d}] AOR          ({len(fields)} fields)  {pdf_path.name}")
        elif category == "fuelcell":
            fields = extract_fuelcell_fields(pages)
            fc_records.append({**base, **fields})
            report_lines.append(
                f"[{i:03d}] FUEL CELL    ({len(fields)} fields)  {pdf_path.name}")
        else:
            unclassified.append({**base, "category": "unclassified"})
            report_lines.append(
                f"[{i:03d}] UNCLASSIFIED              {pdf_path.name}")

    report_lines += [
        "-" * 70,
        f"AOR records         : {len(aor_records)}",
        f"Fuel cell records   : {len(fc_records)}",
        f"Unclassified/errors : {len(unclassified)}",
    ]
    return aor_records, fc_records, unclassified, report_lines


def save_yaml(records: list, path: Path, label: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# EISForge auto-extracted {label} dataset  (v2)\n")
        f.write(f"# Generated : {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("# Reference electrode offsets (V vs RHE at pH 13):\n")
        for k, v in REFERENCE_OFFSETS.items():
            f.write(f"#   {k}: {v} V\n")
        f.write("# WARNING: All values approximate — review before publication.\n\n")
        yaml.dump(records, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)
    print(f"  Saved {len(records):3d} records  →  {path}")


def main():
    ap = argparse.ArgumentParser(
        description="EISForge v2: Extract electrochemical data from AOR/fuel-cell PDFs.")
    ap.add_argument("--input", "-i", required=True,
                    help='PDF folder, e.g. "D:\\Articles and Seminar\\Alcohol oxidation"')
    ap.add_argument("--out", "-o", default=".",
                    help="Output directory (default: current folder)")
    ap.add_argument("--max-pages", type=int, default=14,
                    help="Max pages per PDF (default: 14)")
    args = ap.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        sys.exit(f"Input folder not found: {input_dir}")

    print(f"\nEISForge PDF Extractor  v2")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}\n")

    aor, fc, unk, report = process_folder(input_dir, args.max_pages)

    print()
    save_yaml(aor, output_dir / "extracted_aor.yaml",         "AOR")
    save_yaml(fc,  output_dir / "extracted_fuelcell.yaml",    "Fuel Cell")
    save_yaml(unk, output_dir / "extracted_unclassified.yaml", "Unclassified")

    # HTML review reports
    (output_dir / "review_aor.html").write_text(
        build_html_report(aor, "AOR"), encoding="utf-8")
    (output_dir / "review_fuelcell.html").write_text(
        build_html_report(fc,  "Fuel Cell"), encoding="utf-8")

    report_path = output_dir / "extraction_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(f"  HTML review (AOR)     →  {output_dir / 'review_aor.html'}")
    print(f"  HTML review (FC)      →  {output_dir / 'review_fuelcell.html'}")
    print(f"  Report                →  {report_path}")
    print()
    for line in report[-5:]:
        print(" ", line)


if __name__ == "__main__":
    main()
