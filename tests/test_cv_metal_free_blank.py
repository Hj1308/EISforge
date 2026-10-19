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


# ── patch50: reference catalyst (e.g. carbon black) and charge in a window ───
def test_reference_catalyst_wording():
    r = _cmp(_blank(1) + _extra(), _blank(2), reference="catalyst")
    assert r.verdict == "increase" and "reference catalyst" in r.text
    assert "blank" not in r.text
    r = _cmp(_blank(1) - 0.6 * _extra(), _blank(2), reference="catalyst")
    assert "BELOW the reference catalyst" in r.text


def test_bad_reference_kind_rejected():
    with pytest.raises(ValueError):
        _cmp(_blank(1), _blank(2), reference="carbon")


def test_reference_catalyst_skips_same_electrode_warnings():
    s = CVData(E, _blank(1), meta={"ocp_V": -0.10})
    b = CVData(E, 10 * _blank(2), meta={"ocp_V": -0.30})     # other electrode: 10x, other OCP
    assert len(s.subtract(b).warnings) == 2                   # OCP + blank dominance
    assert s.subtract(b, same_electrode=False).warnings == []


def test_window_charge_matches_trapezoid():
    s_mA, b_mA = _blank(1) + _extra(), _blank(2)
    r = CVData(E, s_mA, scan_rate_mV_s=50.0).compare_to_blank(CVData(E, b_mA),
                                                              e_window=(0.5, 0.2))
    m = UP & (E >= 0.2) & (E <= 0.5)
    assert r.q_sample == pytest.approx(np.trapezoid(s_mA[m], E[m]) / 0.05, rel=1e-6)
    assert r.q_reference == pytest.approx(np.trapezoid(b_mA[m], E[m]) / 0.05, rel=1e-6)
    assert r.q_unit == "mC" and r.ratio_charge > 1


def test_app_reference_catalyst_with_window():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    next(s for s in at.selectbox if s.label == "Catalyst type").select(CARBON)
    at.radio(key="cv_ref_kind").set_value(
        "Reference catalyst (e.g. carbon black, with alcohol)")
    at.checkbox(key="cv_ref_qwin").check().run()
    at.number_input(key="cv_ref_e1").set_value(0.2)
    at.number_input(key="cv_ref_e2").set_value(0.5)
    at.file_uploader(key="cv_up").upload("sample.idf", _idf(_blank(1) + _extra()))
    at.file_uploader(key="cv_blank_up").upload("carbon_black.idf",
                                               _idf(10 * _blank(2))).run()
    assert len(at.exception) == 0, f"app crashed: {at.exception}"
    assert not any("more than twice" in w.value for w in at.warning)
    labels = [m.label for m in at.metric]
    assert "j_max reference" in labels and "Q ratio" in labels
    assert any("Activity vs reference" in x.value for x in list(at.success) + list(at.warning)
               + list(at.info))
