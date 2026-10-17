"""patch48 — CVData.compare_to_blank: activity of a metal-free catalyst judged against the
blank, without a peak. Synthetic carbon-like CVs (µA scale): capacitive loop, cathodic
background, current rising toward the anodic vertex, no anodic peak."""

import numpy as np
import pytest

from eisforge.data.cv_data import CVData

E_UP = np.linspace(-0.8, 0.5, 261)
E = np.concatenate([E_UP, E_UP[-2::-1]])
UP = np.arange(len(E)) < len(E_UP)


def _blank(seed=0, noise=2e-6):
    rng = np.random.default_rng(seed)
    cap = np.where(UP, 1.0e-3, -1.0e-3)                  # mA, double-layer loop
    bg = -2.0e-3 * np.exp(-(E + 0.8) / 0.15)             # cathodic background at low E
    rise = 1.5e-3 * np.exp((E - 0.5) / 0.12)             # rising at the vertex, no peak
    return cap + bg + rise + rng.normal(0, noise, len(E))


def _extra():                                            # extra anodic current, forward only
    return np.where(UP, 1.0e-3 * np.exp((E - 0.5) / 0.10), 0.0)


def _cmp(sample_mA, blank_mA, **kw):
    return CVData(E, sample_mA).compare_to_blank(CVData(E, blank_mA), **kw)


def test_extra_anodic_current_is_an_increase():
    r = _cmp(_blank(1) + _extra(), _blank(2))
    assert r.verdict == "increase"
    assert 0.2 < r.e_significant_V < 0.5
    assert r.ratio_max > 1.2 and r.max_net_mA > 0
    assert "no distinct peak" in r.text.lower()


def test_same_electrode_rerun_is_not_significant():
    r = _cmp(_blank(1) * 1.03, _blank(2))                # 3% run-to-run difference
    assert r.verdict == "no significant difference"


def test_lower_current_with_alcohol_is_a_decrease():
    r = _cmp(_blank(1) - 0.6 * _extra(), _blank(2))
    assert r.verdict == "decrease"
    assert "below the blank" in r.text.lower()


def test_noise_threshold_with_zero_relative_floor():
    small = np.where(UP, 2.0e-5 * np.exp((E - 0.5) / 0.10), 0.0)    # 10x the noise at vertex
    assert _cmp(_blank(1) + small, _blank(2), rel_min=0.0).verdict == "increase"
    assert _cmp(_blank(1) + small, _blank(2)).verdict == "no significant difference"


def test_no_overlap_is_undefined():
    r = CVData(E, _blank(1)).compare_to_blank(CVData(E + 2.0, _blank(2)))
    assert r.verdict == "undefined"


@pytest.mark.parametrize("noise", [1e-7, 2e-6, 1e-5])
def test_noise_estimate_tracks_scatter(noise):
    r = _cmp(_blank(1, noise), _blank(2, noise))
    assert r.noise_mA == pytest.approx(noise * np.sqrt(2), rel=0.3)


# ── app: CV tab, carbon catalyst + blank ─────────────────────────────────────
APP = __import__("pathlib").Path(__file__).resolve().parents[1] / "app.py"
CARBON = "Carbon Material (N-doped C, CNT, Graphene)"


def _idf(i_mA) -> bytes:
    e = E[:-1]                                           # closed loop without the repeat point
    rows = "\n".join(f"{a:.6E} {b * 1e-3:.6E} {a:.6E}" for a, b in zip(e, i_mA[:-1]))
    head = ("Method=CyclicVoltammetry\nScanrate=0.05\nN scans=1\nCurrent Range=100uA\n"
            "Apply wrt OCP=false\n")
    return (head + f"primary_data\n3\n{len(e)}\n{rows}\n").encode("latin-1")


def _app(sample_mA, blank_mA):
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    next(s for s in at.selectbox if s.label == "Catalyst type").select(CARBON).run()
    at.file_uploader(key="cv_up").upload("alcohol.idf", _idf(sample_mA))
    at.file_uploader(key="cv_blank_up").upload("blank.idf", _idf(blank_mA)).run()
    assert len(at.exception) == 0, f"app crashed: {at.exception}"
    return at


def test_app_carbon_increase_shown():
    at = _app(_blank(1) + _extra(), _blank(2))
    assert any("Activity vs blank" in s.value and "above the blank" in s.value
               for s in at.success)
    assert "j_max ratio" in [m.label for m in at.metric]
    assert any("usual for metal-free" in i.value for i in at.info)


def test_app_carbon_decrease_shown():
    at = _app(_blank(1) - 0.6 * _extra(), _blank(2))
    assert any("Activity vs blank" in w.value and "BELOW the blank" in w.value
               for w in at.warning)
