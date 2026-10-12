"""
patch43 — band edges and Mott-Schottky checked against literature values, not only against
the module's own formulas.

* Butler-Ginley: E_CB = chi - 4.50 - Eg/2 is a potential vs NHE (TiO2 anatase: -0.29 V,
  E_VB = +2.91 V; vacuum energies -4.21 / -7.41 eV). TiO2 straddles the water redox levels.
* E_RHE = E_NHE + (RT ln10 / F) pH.
* Mott-Schottky: 1/C^2 = 2/(e eps eps0 A^2 N) (V - Vfb - kT/e) (n-type), so Vfb is the
  V-axis intercept minus kT/e (plus kT/e for p-type).
"""

import re
from pathlib import Path

import numpy as np
import pytest

from eisforge.analysis.band_edge_calculator import BandEdgeCalculator

PAGE = Path(__file__).resolve().parents[1] / "pages" / "band_edge.py"
KT_E_25 = 8.617333262e-5 * 298.15            # 0.02569 V


def _slope(t_c):
    return 8.314462618 * (273.15 + t_c) / 96485.33212 * np.log(10.0)


def test_tio2_literature_values():
    r = BandEdgeCalculator("TiO2", pH=0.0).calculate()
    assert r.Ecb_NHE == pytest.approx(-0.29, abs=1e-6)
    assert r.Evb_NHE == pytest.approx(2.91, abs=1e-6)
    assert r.Ecb_vacuum == pytest.approx(-4.21, abs=1e-6)
    assert r.Evb_vacuum == pytest.approx(-7.41, abs=1e-6)
    assert r.Ecb_NHE < 0.0 < 1.23 < r.Evb_NHE          # straddles H+/H2 and O2/H2O


def test_rhe_sign_and_temperature():
    r7 = BandEdgeCalculator("TiO2", pH=7.0).calculate()
    assert r7.Ecb_RHE == pytest.approx(-0.29 + _slope(25.0) * 7.0, abs=1e-9)   # +0.124 V
    r50 = BandEdgeCalculator("TiO2", pH=7.0, T_celsius=50.0).calculate()
    assert r50.Ecb_RHE == pytest.approx(-0.29 + _slope(50.0) * 7.0, abs=1e-9)
    legacy = BandEdgeCalculator(pH=7.0, T_celsius=50.0).band_edges(chi=5.81, Eg=3.20)
    assert legacy.Ecb_RHE == pytest.approx(r50.Ecb_RHE, abs=1e-12)          # keeps T


def _ms_data(vfb, n_type=True, nd=1e17, eps=10.0):
    a_m2 = 1e-4
    sign = 1.0 if n_type else -1.0
    k = 2.0 / (1.602176634e-19 * 8.8541878128e-12 * eps * a_m2 ** 2 * nd * 1e6)
    if n_type:
        V = np.linspace(vfb + 0.1, vfb + 1.1, 50)
        inv_c2 = k * (V - vfb - KT_E_25)
    else:
        V = np.linspace(vfb - 1.1, vfb - 0.1, 50)
        inv_c2 = sign * k * (V - vfb + KT_E_25)
    return V, 1.0 / np.sqrt(inv_c2)


@pytest.mark.parametrize("n_type", [True, False])
def test_mott_schottky_vfb_includes_kT_over_e(n_type):
    V, C = _ms_data(0.40, n_type=n_type)
    ms = BandEdgeCalculator("custom", electrode_area=1.0).mott_schottky(V, C, epsilon_r=10.0)
    assert ms.sc_type == ("n-type" if n_type else "p-type")
    assert ms.Vfb == pytest.approx(0.40, abs=1e-6)
    assert ms.V_intercept == pytest.approx(0.40 + (KT_E_25 if n_type else -KT_E_25), abs=1e-6)
    assert ms.Nd == pytest.approx(1e17, rel=1e-3)


def test_page_has_rhe_option_and_no_fixed_slope():
    src = PAGE.read_text(encoding="utf-8")
    assert 'key="ms_ref_is_rhe"' in src
    assert not re.search(r"0\.059\d*\s*\*\s*ph_ms", src)


def test_page_band_diagram_numbers():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(PAGE), default_timeout=120).run()
    assert len(at.exception) == 0, at.exception
    at.button[[b.label for b in at.button].index("Calculate Band Edges")].click().run()
    assert len(at.exception) == 0, at.exception
    r = at.session_state["be_result"]
    m = {x.label: x.value for x in at.metric}
    assert float(m["Ecb vs NHE"].split()[0]) == pytest.approx(r.Ecb_NHE, abs=1e-4)
    assert -3.0 < r.Ecb_NHE < 3.0                          # on the electrochemical scale
