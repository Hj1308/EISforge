"""
Litbase PDF / CSV tooling — no LLM API calls.

Workflow:
  A. extract_text_from_pdf(path) -> paper_text.txt       (local, no LLM)
  B. OpenCode reads paper_text.txt, applies extraction    (conversational)
     prompt, writes draft.csv manually.
  C. OpenCode re-reads paper_text.txt and draft.csv,      (conversational)
     applies judge prompt, writes judged.csv manually.
  D. litbase_assemble_draft(judged_csv, pdf_path) ->      (local)
     loads judged.csv, validates against schema, writes
     to litbase.csv (or _STUB_test_output.csv).

Author: Hoda Jafari | July 2026
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from eisforge.litbase.schema import (
    append_draft_rows,
    create_empty_litbase,
    save_litbase,
    validate_row,
)

# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path | str, max_pages: int | None = None) -> str:
    """Extract readable text from a PDF using pymupdf (fitz).

    Returns the concatenated text of all pages, with page markers.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "pymupdf (fitz) is required for PDF text extraction. "
            "Install with: pip install pymupdf"
        )

    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages = []
    total = len(doc)
    limit = max_pages if max_pages else total

    for i in range(min(total, limit)):
        page = doc[i]
        text = page.get_text("text")
        pages.append(f"\n--- PAGE {i + 1} of {total} ---\n{text}")

    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def parse_csv_rows(raw_csv: str) -> list[dict[str, str]]:
    """Parse CSV string (may include markdown fences) into list of dict rows."""
    text = raw_csv.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()

    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [dict(row) for row in reader]
    except Exception:
        try:
            dialect = csv.Sniffer().sniff(text[:1024])
            reader = csv.DictReader(io.StringIO(text), dialect=dialect)
            rows = [dict(row) for row in reader]
        except Exception:
            raise ValueError(
                f"Could not parse CSV. Raw output:\n{text[:500]}..."
            )
    return rows


# ---------------------------------------------------------------------------
# Litbase assembly (Step D)
# ---------------------------------------------------------------------------

def litbase_assemble_draft(
    judged_csv_path: Path | str,
    pdf_source: str,
    output_csv: Path | str,
    extracted_by: str = "opencode-in-context",
    allow_real_write: bool = False,
) -> pd.DataFrame:
    """Read a judged CSV file, validate rows, write to litbase CSV.

    Parameters
    ----------
    allow_real_write: if False, forces output filename to contain
        '_STUB_test_output' as a safety guard against accidental
        writes to the real litbase.csv.
    """
    judged_csv_path = Path(judged_csv_path)
    output_csv = Path(output_csv)

    if not allow_real_write and "_STUB_test_output" not in output_csv.name:
        raise RuntimeError(
            "Safety guard: output filename must contain '_STUB_test_output'.\n"
            "Set allow_real_write=True to write to the real litbase.csv."
        )

    # Parse judged CSV
    raw = judged_csv_path.read_text(encoding="utf-8")
    rows = parse_csv_rows(raw)

    # Validate each row against schema
    print(f"Validating {len(rows)} rows...")
    for i, row in enumerate(rows):
        warnings = validate_row(row)
        if warnings:
            print(f"  Row {i}: {'; '.join(warnings)}")

    # Assemble litbase DataFrame
    df = create_empty_litbase()
    df = append_draft_rows(df, rows, extracted_by=extracted_by, source_pdf=pdf_source)
    save_litbase(df, output_csv)

    # Summary
    verdicts = df["judge_verdict"].fillna("").values
    n_confirmed = sum(1 for v in verdicts if v.startswith("confirmed"))
    n_discrepancy = sum(1 for v in verdicts if v.startswith("discrepancy"))
    n_cannot = sum(1 for v in verdicts if v.startswith("cannot verify"))
    n_empty = sum(1 for v in verdicts if v.strip() == "")

    print(f"Saved {len(df)} rows -> {output_csv}")
    print(f"  confirmed:         {n_confirmed}")
    print(f"  discrepancy found:  {n_discrepancy}")
    print(f"  cannot verify:     {n_cannot}")
    print(f"  no verdict:        {n_empty}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Litbase PDF extraction tooling")
    sub = ap.add_subparsers(dest="command")

    # extract-text
    et = sub.add_parser("extract-text", help="Extract text from PDF")
    et.add_argument("pdf", help="Path to PDF")
    et.add_argument("-o", "--output", default="paper_text.txt", help="Output text file")
    et.add_argument("--max-pages", type=int, default=None)

    # assemble
    asm = sub.add_parser("assemble", help="Assemble judged CSV into litbase format")
    asm.add_argument("judged_csv", help="Path to judged CSV file")
    asm.add_argument("-o", "--output", default="litbase_STUB_test_output.csv")
    asm.add_argument("--pdf-source", default="unknown.pdf")
    asm.add_argument("--allow-real-write", action="store_true")

    args = ap.parse_args()

    if args.command == "extract-text":
        text = extract_text_from_pdf(Path(args.pdf), max_pages=args.max_pages)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Extracted {len(text)} chars -> {args.output}")

    elif args.command == "assemble":
        litbase_assemble_draft(
            judged_csv_path=args.judged_csv,
            pdf_source=args.pdf_source,
            output_csv=Path(args.output),
            allow_real_write=args.allow_real_write,
        )

    else:
        ap.print_help()
