"""Private reference data for the real-file tests (patch54).

The author's measurement files are unpublished. Their file names, measurement
conditions and reference values must not appear in this repository (AGENTS.md,
"Private data stays private"), so the real-file tests read them from a JSON file
that lives next to the data, outside the repo:

    $EISFORGE_LOCAL_DATA/reference_values.json

Each test asks for one section of that file. If the variable is not set, the JSON
is missing, or the section is missing, the test is skipped (CI never has the data).
"""
import json
import os
from pathlib import Path

import pytest

LOCAL = os.environ.get("EISFORGE_LOCAL_DATA")
REFS_NAME = "reference_values.json"


def private_refs(section):
    """Return (folder, refs[section]) or skip the calling test."""
    if not LOCAL:
        pytest.skip("EISFORGE_LOCAL_DATA not set (private files)")
    folder = Path(LOCAL)
    path = folder / REFS_NAME
    if not path.exists():
        pytest.skip(f"{REFS_NAME} not found in the private data folder")
    refs = json.loads(path.read_text(encoding="utf-8"))
    if section not in refs:
        pytest.skip(f"section {section!r} not in {REFS_NAME}")
    return folder, refs[section]


def private_file(folder, name):
    """Path of one private file, or skip the calling test if it is missing."""
    p = Path(folder) / name
    if not p.exists():
        pytest.skip("private file not found")
    return p
