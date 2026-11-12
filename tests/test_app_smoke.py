"""ROADMAP 1.2 (patch53): every Streamlit script runs top to bottom without an exception.

pytest never imports app.py, so a duplicate widget ID (StreamlitDuplicateElementId) or a
NameError at module level passed the whole suite silently once (AGENTS.md, "Known gotchas").
AppTest really executes the script body and reports exceptions, so this file turns those
bugs into a red CI run. Pattern: streamlit.testing.v1.AppTest docs (Streamlit >= 1.28).
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["app.py"] + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "pages").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs_without_exception(script):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(ROOT / script), default_timeout=120).run()
    messages = [e.value for e in at.exception]
    assert not messages, f"{script} raised: {messages}"


def test_every_page_is_covered():
    # A new page in pages/ is picked up by the glob above; this guards the glob itself.
    assert "app.py" in SCRIPTS and len(SCRIPTS) >= 2, SCRIPTS
