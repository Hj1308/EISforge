"""
eisforge/litbase/schema.py

Defines the litbase schema (literature comparison database for AOR
catalysts) and provides load/save/validate helpers. Deliberately
pandas + stdlib only -- no heavy NLP deps, to stay within Streamlit
Cloud free-tier limits.

Design principle: litbase is a single CSV with a `verified` column.
Rows extracted by an LLM (OpenCode/DeepSeek) always start as
verified=False ("draft"). A row only counts as part of the canonical,
citable database once a human has checked it against the source PDF
and flipped verified=True. Nothing downstream (comparison tables,
figures for the manuscript) should read unverified rows without an
explicit opt-in flag.
"""

from __future__ import annotations

import os
import csv
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Column name -> pandas dtype used when creating an empty DataFrame / after
# loading from CSV (CSV itself has no dtypes, so we coerce on load).
COLUMN_DTYPES: dict[str, str] = {
    "ref_key": "string",
    "doi": "string",
    "catalyst_family": "string",
    "catalyst_name": "string",
    "alcohol": "string",
    "electrolyte_composition": "string",
    "electrolyte_type": "string",
    "electrolyte_concentration_M": "float64",
    "reference_electrode": "string",
    "onset_V": "float64",
    "onset_criterion": "string",
    "tafel_mV_dec": "float64",
    "mass_activity_A_g": "float64",
    "mass_activity_basis": "string",
    "evaluation_potential_basis": "string",
    "page_or_table_ref": "string",
    "verified": "boolean",
    "notes": "string",
    # Bookkeeping columns not part of the extraction prompt output, added
    # automatically by this module:
    "source_pdf": "string",       # filename the row was extracted from
    "extracted_by": "string",     # e.g. "opencode-deepseek", "manual"
    "added_date": "string",       # ISO date string
    "verified_by": "string",      # who flipped verified=True
    "verified_date": "string",    # ISO date string
}

COLUMNS: list[str] = list(COLUMN_DTYPES.keys())

# Columns that MUST be non-empty for a row to be promoted to verified=True.
# (Numeric fields are allowed to be blank -- not every paper reports every
# parameter -- but if a numeric field IS filled in, page_or_table_ref must
# also be filled in. That check is done in validate_row, not here.)
REQUIRED_FOR_VERIFICATION = [
    "ref_key",
    "catalyst_name",
    "alcohol",
    "electrolyte_composition",
]

ELECTROLYTE_TYPES = {"acidic", "alkaline", "neutral"}

CATALYST_FAMILIES = {
    "metal-free carbon",
    "Pt-based",
    "Pt-Ru",
    "Pt-alloy",
    "Pd-based",
    "metal-oxide",
    "other",
}

MASS_ACTIVITY_BASES = {"total catalyst mass", "metal mass only", "geometric area only"}

EVALUATION_POTENTIAL_BASES = {"fixed potential", "CV peak"}

NUMERIC_COLUMNS = [
    "electrolyte_concentration_M",
    "onset_V",
    "tafel_mV_dec",
    "mass_activity_A_g",
]


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def create_empty_litbase() -> pd.DataFrame:
    """Return an empty, correctly-typed litbase DataFrame."""
    df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in COLUMN_DTYPES.items()})
    return df


def load_litbase(path: str) -> pd.DataFrame:
    """
    Load litbase from CSV. If the file doesn't exist yet, returns an empty
    (but correctly-schema'd) DataFrame -- doesn't raise, so callers can do
    load -> append -> save without a separate "does it exist" check.
    """
    if not os.path.exists(path):
        return create_empty_litbase()

    df = pd.read_csv(path, dtype=str)  # read everything as string first

    # Ensure all expected columns exist (forward-compatible: if schema
    # gained a column since this CSV was written, add it as empty rather
    # than crashing).
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Drop any unexpected columns silently is DANGEROUS for a scientific
    # database -- instead, keep them but warn, so nothing is silently lost.
    extra_cols = [c for c in df.columns if c not in COLUMNS]
    if extra_cols:
        print(f"[litbase] WARNING: CSV has unexpected columns, kept as-is: {extra_cols}")

    df = df[COLUMNS + extra_cols]

    # Coerce dtypes
    for col, dtype in COLUMN_DTYPES.items():
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "boolean":
            df[col] = df[col].map({"True": True, "true": True, "TRUE": True,
                                    "False": False, "false": False, "FALSE": False})
            df[col] = df[col].astype("boolean")
        else:
            df[col] = df[col].astype("string")

    return df


def save_litbase(df: pd.DataFrame, path: str) -> None:
    """Save litbase to CSV, quoting all fields (safest for text with commas)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    print(f"[litbase] saved {len(df)} rows -> {path}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_row(row: dict) -> list[str]:
    """
    Validate a single row dict against the schema rules. Returns a list of
    human-readable warning/error strings. Empty list = no problems found.

    This does NOT check factual correctness against the source PDF -- that
    is a human's job. It only checks internal consistency (e.g. "if you
    filled in a numeric value, you must also cite a page").
    """
    issues: list[str] = []

    if row.get("electrolyte_type") and row["electrolyte_type"] not in ELECTROLYTE_TYPES:
        issues.append(
            f"electrolyte_type='{row['electrolyte_type']}' not in {sorted(ELECTROLYTE_TYPES)}"
        )

    if row.get("catalyst_family") and row["catalyst_family"] not in CATALYST_FAMILIES:
        issues.append(
            f"catalyst_family='{row['catalyst_family']}' not in {sorted(CATALYST_FAMILIES)}"
        )

    if row.get("mass_activity_basis") and row["mass_activity_basis"] not in MASS_ACTIVITY_BASES:
        issues.append(
            f"mass_activity_basis='{row['mass_activity_basis']}' not in {sorted(MASS_ACTIVITY_BASES)}"
        )

    if row.get("evaluation_potential_basis") and row["evaluation_potential_basis"] not in EVALUATION_POTENTIAL_BASES:
        issues.append(
            f"evaluation_potential_basis='{row['evaluation_potential_basis']}' "
            f"not in {sorted(EVALUATION_POTENTIAL_BASES)}"
        )

    # Any filled-in numeric value MUST have a page/table reference.
    has_any_numeric = any(row.get(col) not in (None, "", "nan") for col in NUMERIC_COLUMNS)
    if has_any_numeric and not row.get("page_or_table_ref"):
        issues.append(
            "a numeric value is present but page_or_table_ref is empty -- "
            "every number must be traceable to a page/table in the source PDF"
        )

    return issues


def validate_for_promotion(row: dict) -> list[str]:
    """
    Stricter check used before flipping verified=True. Includes
    validate_row() plus the "required fields present" check.
    """
    issues = validate_row(row)
    for col in REQUIRED_FOR_VERIFICATION:
        if not row.get(col):
            issues.append(f"'{col}' is required before a row can be verified")
    return issues


# ---------------------------------------------------------------------------
# Convenience operations
# ---------------------------------------------------------------------------

def append_draft_rows(df: pd.DataFrame, new_rows: list[dict], extracted_by: str,
                       source_pdf: Optional[str] = None) -> pd.DataFrame:
    """
    Append new draft rows (verified=False) to an existing litbase DataFrame.
    Validates each row and prints warnings, but does NOT reject rows with
    warnings -- drafts are allowed to be imperfect; verification is where
    correctness is enforced.
    """
    import datetime
    today = datetime.date.today().isoformat()

    prepared = []
    for i, row in enumerate(new_rows):
        issues = validate_row(row)
        if issues:
            print(f"[litbase] draft row {i} ({row.get('ref_key', '?')}) has warnings:")
            for issue in issues:
                print(f"    - {issue}")

        full_row = {col: row.get(col, pd.NA) for col in COLUMNS}
        full_row["verified"] = False
        full_row["extracted_by"] = extracted_by
        full_row["source_pdf"] = source_pdf or row.get("source_pdf", "")
        full_row["added_date"] = today
        prepared.append(full_row)

    new_df = pd.DataFrame(prepared)
    return pd.concat([df, new_df], ignore_index=True)


def promote_row(df: pd.DataFrame, index: int, verified_by: str) -> pd.DataFrame:
    """
    Flip a single row's verified flag to True, after re-checking it passes
    validate_for_promotion(). Raises ValueError if the row still has
    unresolved issues -- promotion is meant to be a deliberate, checked
    action, not a bulk toggle.
    """
    import datetime

    row = df.loc[index].to_dict()
    issues = validate_for_promotion(row)
    if issues:
        raise ValueError(
            f"Row {index} ({row.get('ref_key', '?')}) cannot be verified, "
            f"unresolved issues: {issues}"
        )

    df.loc[index, "verified"] = True
    df.loc[index, "verified_by"] = verified_by
    df.loc[index, "verified_date"] = datetime.date.today().isoformat()
    return df


def verified_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where verified == True. Use this for any output
    that will be cited in the thesis or manuscript."""
    return df[df["verified"] == True].reset_index(drop=True)  # noqa: E712


if __name__ == "__main__":
    # Smoke test: create empty, add a couple of draft rows, validate,
    # save, reload, confirm round-trip works.
    df = create_empty_litbase()
    print("Empty litbase columns:", list(df.columns))

    draft_rows = [
        {
            "ref_key": "TestPaper2024",
            "catalyst_family": "Pt-based",
            "catalyst_name": "Pt/C",
            "alcohol": "isopropanol",
            "electrolyte_composition": "1 M H2SO4",
            "electrolyte_type": "acidic",
            "electrolyte_concentration_M": "1.0",
            "reference_electrode": "Ag/AgCl",
            "onset_V": "0.05",
            "onset_criterion": "J = 0.1 mA/cm2 threshold",
            "page_or_table_ref": "p. 4, Table 2",
        },
        {
            # Intentionally missing page_or_table_ref despite having a
            # numeric value -- should trigger a warning.
            "ref_key": "BadRowExample",
            "onset_V": "0.10",
        },
    ]

    df = append_draft_rows(df, draft_rows, extracted_by="manual-smoke-test")
    print("\nAfter append:")
    print(df[["ref_key", "onset_V", "page_or_table_ref", "verified"]])

    save_litbase(df, "litbase_smoketest.csv")
    reloaded = load_litbase("litbase_smoketest.csv")
    print("\nReloaded, verified column dtype:", reloaded["verified"].dtype)
    print("Round-trip row count matches:", len(reloaded) == len(df))

    os.remove("litbase_smoketest.csv")
    print("\nSmoke test cleanup done.")
