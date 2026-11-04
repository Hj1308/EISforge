"""
ROADMAP 0.7 (patch52): the sidebar "Catalyst type" has no default.

A carbon user who forgot to change the old "Noble Metal" default silently got
I_f/I_b and CO-poisoning wording. Now the selectbox starts empty (key
"sb_catalyst_type") and CV/LSV analysis waits until a family is chosen.
Runs the real app.py through streamlit.testing.v1.AppTest (see AGENTS.md).
"""

from pathlib import Path

import pytest

from tests.test_cv_blank_tab import _idf

APP = Path(__file__).resolve().parents[1] / "app.py"
CARBON = "Carbon Material (N-doped C, CNT, Graphene)"


def _app():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_no_default_catalyst_type():
    at = _app()
    assert len(at.exception) == 0
    sb = at.selectbox(key="sb_catalyst_type")
    assert sb.value is None
    assert any("Select catalyst type first" in i.value for i in at.info)


def test_cv_waits_for_catalyst_type_then_runs():
    at = _app()
    at.file_uploader(key="cv_up").upload("sample.idf", _idf()).run()
    assert len(at.exception) == 0
    assert "cv_r" not in at.session_state                 # not analysed yet
    assert any("to analyse this CV" in i.value for i in at.info)

    at.selectbox(key="sb_catalyst_type").select(CARBON).run()
    assert len(at.exception) == 0
    assert "cv_r" in at.session_state                     # analysed once a type is set
    assert not any("Select catalyst type first" in i.value for i in at.info)
