#!/usr/bin/env python3
"""
EISForge PDF Extraction Pipeline
=================================
Extracts electrochemical performance data from AOR / fuel-cell PDF papers.

Usage
-----
# Install dependencies first (once):
#   pip install pdfplumber PyYAML tqdm

python extract_papers.py --input "D:\\Articles and Seminar\\Alcohol oxidation"

Output files (written next to this script, or use --out to specify folder):
  extracted_aor.yaml           -- alcohol oxidation records
  extracted_fuelcell.yaml      -- fuel cell records
  extracted_unclassified.yaml  -- papers that could not be categorised
  extraction_report.txt        -- per-file summary + warnings

ALL extracted values are approximate and must be reviewed manually
before use in benchmarks or publications.

Author: Hoda Jafari | EISForge | 2026
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
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
# Keywords for automatic paper classification
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

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for numerical data extraction
# ─────────────────────────────────────────────────────────────────────────────

# Potential values: "-0.35 V", "0.45 V vs RHE", "−0.3 V"
RE_POTENTIAL = re.compile(
    r"([\u2212\-]?\d+\.?\d*)\s*V"
    r"(?:\s*(?:vs\.?|versus)?\s*(RHE|NHE|SHE|Ag/AgCl|SCE|Hg/HgO))?",
    re.IGNORECASE,
)

# Current density: "15.8 mA cm-2", "1540 mA mg-1", "120 mA/cm2"
RE_CURRENT = re.compile(
    r"([\u2212\-]?\d+\.?\d*)\s*"
    r"(mA\s*(?:cm[\-\u22122]?|cm\u00b2|mg[\-\u22121]?|g[\-\u22121]?)|A\s*(?:g[\-\u22121]?|m[\-\u22122]?))",
    re.IGNORECASE,
)

# Tafel slope: "68 mV dec-1", "120 mV/dec"
RE_TAFEL = re.compile(
    r"(\d+\.?\d*)\s*mV\s*(?:dec[\-\u22121]?|per\s*dec|/dec)",
    re.IGNORECASE,
)

# If/Ib ratio
RE_IF_IB = re.compile(
    r"(?:I_?f\s*/\s*I_?b|forward[\s/]backward\s*ratio)\s*[=:]?\s*([\d\.]+)",
    re.IGNORECASE,
)

# Faradaic efficiency
RE_FE = re.compile(
    r"(?:faradaic\s*efficiency|FE)\s*[=:of]*\s*([\d\.]+)\s*%",
    re.IGNORECASE,
)

# Peak power density (fuel cell)
RE_POWER = re.compile(
    r"([\d\.]+)\s*mW\s*(?:cm[\-\u22122]?|cm\u00b2)",
    re.IGNORECASE,
)

# EIS parameters
RE_EIS = re.compile(
    r"(R_?(?:ct|s|ohm|sol|int|charge))\s*[=:]\s*([\d\.]+)\s*(?:\u03a9|ohm|Ohm)",
    re.IGNORECASE,
)

# Electrolyte
RE_ELECTROLYTE = re.compile(
    r"(\d+\.?\d*)\s*M\s*(KOH|NaOH|H2SO4|HClO4|NaHCO3|PBS|phosphate)",
    re.IGNORECASE,
)

# Scan rate
RE_SCAN_RATE = re.compile(
    r"(\d+\.?\d*)\s*mV\s*s[\-\u22121]",
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


def classify_paper(text: str) -> str:
    """Return 'aor', 'fuelcell', or 'unclassified'."""
    tl = text.lower()
    aor_score = sum(kw.lower() in tl for kw in AOR_KEYWORDS)
    fc_score  = sum(kw.lower() in tl for kw in FUELCELL_KEYWORDS)
    if aor_score == 0 and fc_score == 0:
        return "unclassified"
    return "aor" if aor_score >= fc_score else "fuelcell"


def extract_text(pdf_path: Path, max_pages: int = 12) -> str:
    """Extract plain text from first max_pages pages of a PDF."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:max_pages]:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
    except Exception as exc:
        return f"__ERROR__{exc}"
    return "\n".join(text_parts)


def extract_title_authors(text: str) -> tuple[str, str]:
    """Best-effort title/author extraction from first lines."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title   = lines[0][:200] if lines else "Unknown"
    authors = lines[1][:200] if len(lines) > 1 else "Unknown"
    return title, authors


def extract_aor_fields(text: str) -> dict:
    """Extract AOR-specific numerical fields."""
    fields: dict = {}

    # Potentials
    potentials = []
    for m in RE_POTENTIAL.finditer(text):
        val = _safe_float(m.group(1))
        ref = m.group(2) if m.lastindex and m.lastindex >= 2 else "unknown"
        if val is not None and -3.0 < val < 3.0:
            potentials.append({"value_V": val, "reference": ref or "unknown"})
    if potentials:
        fields["potentials_extracted"] = potentials[:6]

    # Current densities
    currents = []
    for m in RE_CURRENT.finditer(text):
        val  = _safe_float(m.group(1))
        unit = m.group(2).strip()
        if val is not None and val > 0:
            currents.append({"value": val, "unit": unit})
    if currents:
        fields["current_densities_extracted"] = currents[:6]

    # Tafel slope
    tafel = [_safe_float(m.group(1)) for m in RE_TAFEL.finditer(text)]
    tafel = [v for v in tafel if v and 20 < v < 300]
    if tafel:
        fields["tafel_slope_mV_dec"] = tafel[0]

    # If/Ib
    ifib = [_safe_float(m.group(1)) for m in RE_IF_IB.finditer(text)]
    ifib = [v for v in ifib if v and 0.1 < v < 20]
    if ifib:
        fields["If_Ib_ratio"] = ifib[0]

    # Faradaic efficiency
    fe = [_safe_float(m.group(1)) for m in RE_FE.finditer(text)]
    fe = [v for v in fe if v and 0 < v <= 100]
    if fe:
        fields["faradaic_efficiency_pct"] = fe[0]

    # Electrolyte
    elec = RE_ELECTROLYTE.search(text)
    if elec:
        fields["electrolyte"] = f"{elec.group(1)} M {elec.group(2)}"

    # Scan rate
    sr = RE_SCAN_RATE.search(text)
    if sr:
        fields["scan_rate_mV_s"] = _safe_float(sr.group(1))

    return fields


def extract_fuelcell_fields(text: str) -> dict:
    """Extract fuel-cell-specific numerical fields."""
    fields: dict = {}

    power = [_safe_float(m.group(1)) for m in RE_POWER.finditer(text)]
    power = [v for v in power if v and v > 0]
    if power:
        fields["peak_power_density_mW_cm2"] = max(power)

    eis_params = {}
    for m in RE_EIS.finditer(text):
        key = m.group(1).lower().replace(" ", "_")
        val = _safe_float(m.group(2))
        if val is not None:
            eis_params[key] = val
    if eis_params:
        fields["eis_parameters"] = eis_params

    elec = RE_ELECTROLYTE.search(text)
    if elec:
        fields["electrolyte"] = f"{elec.group(1)} M {elec.group(2)}"

    currents = []
    for m in RE_CURRENT.finditer(text):
        val  = _safe_float(m.group(1))
        unit = m.group(2).strip()
        if val is not None and val > 0:
            currents.append({"value": val, "unit": unit})
    if currents:
        fields["current_densities_extracted"] = currents[:4]

    sr = RE_SCAN_RATE.search(text)
    if sr:
        fields["scan_rate_mV_s"] = _safe_float(sr.group(1))

    return fields


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_folder(input_dir: Path, max_pages: int = 12):
    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No PDF files found in {input_dir}")

    aor_records, fc_records, unclassified, report_lines = [], [], [], []
    report_lines.append(f"EISForge Extraction Report — {datetime.now():%Y-%m-%d %H:%M}")
    report_lines.append(f"Input : {input_dir}")
    report_lines.append(f"PDFs  : {len(pdf_files)}")
    report_lines.append("-" * 70)

    iterator = tqdm(pdf_files, desc="Extracting") if HAS_TQDM else pdf_files

    for i, pdf_path in enumerate(iterator, 1):
        text = extract_text(pdf_path, max_pages)
        if text.startswith("__ERROR__"):
            report_lines.append(f"[{i:03d}] ERROR        {pdf_path.name}")
            unclassified.append({"file": pdf_path.name, "error": text[9:]})
            continue

        category = classify_paper(text)
        title, authors = extract_title_authors(text)
        base = {
            "file": pdf_path.name,
            "title_extracted": title,
            "authors_extracted": authors,
            "extraction_note": "auto-extracted — review manually before use",
            "approximate": True,
        }

        if category == "aor":
            fields = extract_aor_fields(text)
            aor_records.append({**base, **fields})
            report_lines.append(f"[{i:03d}] AOR          ({len(fields)} fields)  {pdf_path.name}")
        elif category == "fuelcell":
            fields = extract_fuelcell_fields(text)
            fc_records.append({**base, **fields})
            report_lines.append(f"[{i:03d}] FUEL CELL    ({len(fields)} fields)  {pdf_path.name}")
        else:
            unclassified.append({**base, "category": "unclassified"})
            report_lines.append(f"[{i:03d}] UNCLASSIFIED              {pdf_path.name}")

    report_lines.append("-" * 70)
    report_lines.append(f"AOR records         : {len(aor_records)}")
    report_lines.append(f"Fuel cell records   : {len(fc_records)}")
    report_lines.append(f"Unclassified/errors : {len(unclassified)}")
    return aor_records, fc_records, unclassified, report_lines


def save_yaml(records: list, path: Path, label: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# EISForge auto-extracted {label} dataset\n")
        f.write(f"# Generated : {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("# WARNING   : All values are auto-extracted and must be\n")
        f.write("#             manually reviewed before use in benchmarks.\n\n")
        yaml.dump(records, f, allow_unicode=True,
                  default_flow_style=False, sort_keys=False)
    print(f"  Saved {len(records):3d} records  →  {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Extract electrochemical data from AOR/fuel-cell PDFs.")
    ap.add_argument("--input", "-i", required=True,
                    help='Folder containing PDF files, e.g. "D:\\Articles and Seminar\\Alcohol oxidation"')
    ap.add_argument("--out", "-o", default=".",
                    help="Output directory (default: current folder)")
    ap.add_argument("--max-pages", type=int, default=12,
                    help="Max pages to read per PDF (default: 12)")
    args = ap.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        sys.exit(f"Input folder not found: {input_dir}")

    print(f"\nEISForge PDF Extractor")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}\n")

    aor, fc, unk, report = process_folder(input_dir, args.max_pages)

    print()
    save_yaml(aor, output_dir / "extracted_aor.yaml",         "AOR")
    save_yaml(fc,  output_dir / "extracted_fuelcell.yaml",    "Fuel Cell")
    save_yaml(unk, output_dir / "extracted_unclassified.yaml", "Unclassified")

    report_path = output_dir / "extraction_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"  Report                →  {report_path}")
    print()
    for line in report[-5:]:
        print(" ", line)


if __name__ == "__main__":
    main()
