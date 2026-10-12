"""
patch42 — cleanup: no deprecated use_container_width anywhere in the Streamlit UI
(replaced by width="stretch", which needs streamlit >= 1.49), and the unused
CVAnalyzer._ONSET_RANGES table is gone.
"""

import re
from pathlib import Path

from eisforge.analysis.cv_analyzer import CVAnalyzer

ROOT = Path(__file__).resolve().parents[1]
UI_FILES = [ROOT / "app.py", *sorted((ROOT / "pages").glob("*.py"))]


def test_no_use_container_width_left():
    left = [f"{p.name}:{n}" for p in UI_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if "use_container_width" in line]
    assert left == [], f'use width="stretch" instead: {left}'


def test_streamlit_minimum_supports_width_stretch():
    for name in ("requirements.txt", "pyproject.toml"):
        m = re.search(r"streamlit>=(\d+)\.(\d+)", (ROOT / name).read_text(encoding="utf-8"))
        assert m, name
        assert (int(m.group(1)), int(m.group(2))) >= (1, 49), name


def test_unused_onset_ranges_removed():
    assert not hasattr(CVAnalyzer, "_ONSET_RANGES")
