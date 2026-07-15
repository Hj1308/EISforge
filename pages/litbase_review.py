"""
Litbase Review — review draft rows produced by OpenCode extraction.

Reads draft CSVs from eisforge/litbase/drafts/, displays each row
as a reviewable card, and handles Accept/Reject/Skip actions.
No LLM calls happen here — this is a pure review UI.

Author: Hoda Jafari | July 2026
"""

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from eisforge.litbase.schema import (
    COLUMN_DTYPES,
    append_draft_rows,
    create_empty_litbase,
    load_litbase,
    promote_row,
    save_litbase,
    validate_for_promotion,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DRAFTS_DIR = ROOT / "eisforge" / "litbase" / "drafts"
DATA_DIR = ROOT / "eisforge" / "litbase" / "data"
CANONICAL_CSV = DATA_DIR / "litbase.csv"
REJECTED_LOG = DRAFTS_DIR / "_rejected_log.csv"

DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_draft_file(fpath: Path) -> pd.DataFrame:
    """Load a draft CSV, coerce types."""
    df = pd.read_csv(fpath, dtype=str, keep_default_na=False)
    for col, dtype in COLUMN_DTYPES.items():
        if col in df.columns:
            if dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "boolean":
                df[col] = df[col].map({"True": True, "False": False, "": False}).fillna(False).astype(bool)
            else:
                df[col] = df[col].astype(str)
    return df


def _save_draft_file(df: pd.DataFrame, fpath: Path) -> None:
    """Save a DataFrame back to a draft CSV."""
    df.to_csv(fpath, index=False, quoting=csv.QUOTE_ALL)


def _log_rejection(row: dict, reason: str, source_file: str) -> None:
    """Log a rejected row to the rejection log."""
    log_row = {
        "timestamp": date.today().isoformat(),
        "ref_key": row.get("ref_key", ""),
        "catalyst_name": row.get("catalyst_name", ""),
        "alcohol": row.get("alcohol", ""),
        "judge_verdict": row.get("judge_verdict", ""),
        "rejection_reason": reason,
        "source_draft": source_file,
    }
    log_cols = list(log_row.keys())
    try:
        existing = pd.read_csv(REJECTED_LOG) if REJECTED_LOG.exists() else pd.DataFrame(columns=log_cols)
    except Exception:
        existing = pd.DataFrame(columns=log_cols)
    new_entry = pd.DataFrame([log_row])
    combined = pd.concat([existing, new_entry], ignore_index=True)
    combined.to_csv(REJECTED_LOG, index=False)


def _remove_row_from_draft(fpath: Path, row_index: int) -> None:
    """Remove a single row from a draft CSV and re-save."""
    df = _load_draft_file(fpath)
    df = df.drop(df.index[row_index]).reset_index(drop=True)
    _save_draft_file(df, fpath)


def _accept_row(row: dict, fpath: Path, row_index: int) -> str:
    """Validate, promote, and copy row to canonical litbase.csv. Returns error or empty."""
    issues = validate_for_promotion(row)
    if issues:
        return "; ".join(issues)

    # Load or create canonical litbase
    canonical = load_litbase(str(CANONICAL_CSV)) if CANONICAL_CSV.exists() else create_empty_litbase()

    # Append row
    canonical = append_draft_rows(
        canonical, [row],
        extracted_by=row.get("extracted_by", "opencode"),
        source_pdf=row.get("source_pdf", ""),
    )

    # Promote (the last-added row)
    try:
        canonical = promote_row(canonical, len(canonical) - 1, "Hoda Jafari")
    except ValueError as e:
        return str(e)

    save_litbase(canonical, str(CANONICAL_CSV))
    _remove_row_from_draft(fpath, row_index)
    return ""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "accepted_count" not in st.session_state:
    st.session_state.accepted_count = 0
if "rejected_count" not in st.session_state:
    st.session_state.rejected_count = 0
if "skipped_count" not in st.session_state:
    st.session_state.skipped_count = 0
if "reload_flag" not in st.session_state:
    st.session_state.reload_flag = 0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Litbase Review", page_icon="\U0001f4da", layout="wide")
st.title("\U0001f4da Litbase Review")

# File picker
draft_files = sorted(DRAFTS_DIR.glob("*.csv"))
draft_files = [f for f in draft_files if not f.name.startswith("_")]

if not draft_files:
    st.info("No draft files found in `eisforge/litbase/drafts/`.\n\nPlace draft CSVs (with `judge_verdict` column) in that folder to begin reviewing.")
    st.stop()

file_names = ["All pending drafts (combined)"] + [f.name for f in draft_files]
selected = st.selectbox("Draft file to review:", file_names)

# Load rows
if selected == "All pending drafts (combined)":
    all_rows = []
    for f in draft_files:
        df = _load_draft_file(f)
        df["_source_file"] = f.name
        df["_source_path"] = str(f)
        all_rows.append(df)
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
    else:
        combined = pd.DataFrame()
    rows = combined.to_dict("records")
    source_file = "combined"
else:
    fpath = DRAFTS_DIR / selected
    df = _load_draft_file(fpath)
    df["_source_file"] = selected
    df["_source_path"] = str(fpath)
    rows = df.to_dict("records")
    source_file = selected

if not rows:
    st.info("No rows in selected draft.")
    st.stop()

# Summary counts
pending = len(rows)
c1, c2, c3 = st.columns(3)
c1.metric("Pending", pending)
c2.metric("Accepted this session", st.session_state.accepted_count)
c3.metric("Rejected this session", st.session_state.rejected_count)

st.divider()

# ── Row cards ──
for i, row in enumerate(rows):
    verdict = row.get("judge_verdict", "")
    if verdict.startswith("confirmed"):
        badge_color = "green"
        badge_icon = "\u2705"
    elif verdict.startswith("discrepancy"):
        badge_color = "red"
        badge_icon = "\u274c"
    elif verdict.startswith("cannot verify"):
        badge_color = "orange"
        badge_icon = "\u26a0\ufe0f"
    else:
        badge_color = "gray"
        badge_icon = "\u2753"

    key_label = f"{row.get('catalyst_name','?')} / {row.get('alcohol','?')}"
    with st.expander(f"{badge_icon} {key_label} \u2014 {row.get('onset_V','?')} V \u2014 [{verdict[:40]}]", expanded=(i == 0)):
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**Catalyst:** {row.get('catalyst_name', '—')} ({row.get('catalyst_family', '—')})")
            st.markdown(f"**Alcohol:** {row.get('alcohol', '—')} | **Electrolyte:** {row.get('electrolyte_composition', '—')} ({row.get('electrolyte_type', '—')})")
            st.markdown(f"**Onset:** {row.get('onset_V', '—')} V vs {row.get('reference_electrode', '—')} — {row.get('onset_criterion', '—')}")
            st.markdown(f"**Tafel:** {row.get('tafel_mV_dec', '—')} mV/dec | **Mass activity:** {row.get('mass_activity_A_g', '—')} A/g ({row.get('mass_activity_basis', '—')})")
            st.caption(f"DOI: {row.get('doi', '—')} | Source: {row.get('page_or_table_ref', '—')}")

        with col2:
            st.markdown(f"**:{badge_color}[Judge: {verdict[:80]}]**")
            if row.get("notes"):
                st.caption(f"\U0001f4dd {row['notes']}")

            st.divider()

            # Inline edit fields for validation fixes
            with st.expander("\u270f\ufe0f Edit fields (fix before accepting)"):
                edits = {}
                for col_name in ["catalyst_name", "alcohol", "electrolyte_composition", "onset_criterion",
                                 "page_or_table_ref", "tafel_mV_dec", "mass_activity_A_g", "notes"]:
                    val = row.get(col_name, "")
                    if isinstance(val, float) and pd.isna(val):
                        val = ""
                    edits[col_name] = st.text_input(col_name, value=str(val), key=f"edit_{i}_{col_name}")

            # Actions
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("\u2705 Accept", key=f"accept_{i}", type="primary", use_container_width=True):
                    # Merge edits back into row
                    for col_name, new_val in edits.items():
                        if new_val != str(row.get(col_name, "")):
                            row[col_name] = new_val

                    err = _accept_row(row, Path(row["_source_path"]), i)
                    if err:
                        st.error(f"Cannot accept: {err}")
                    else:
                        st.session_state.accepted_count += 1
                        st.success(f"Accepted: {key_label}")
                        st.rerun()

            with ac2:
                reason_key = f"reject_reason_{i}"
                reason = st.text_input("Reason (optional):", key=reason_key, placeholder="e.g. duplicate, wrong unit")
                if st.button("\u274c Reject", key=f"reject_{i}", use_container_width=True):
                    _log_rejection(row, reason, row.get("_source_file", "unknown"))
                    _remove_row_from_draft(Path(row["_source_path"]), i)
                    st.session_state.rejected_count += 1
                    st.warning(f"Rejected: {key_label}")
                    st.rerun()

            with ac3:
                if st.button("\u23ed\ufe0f Skip", key=f"skip_{i}", use_container_width=True):
                    st.session_state.skipped_count += 1
                    st.info(f"Skipped: {key_label} — stays in drafts")
