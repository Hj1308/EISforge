"""
Regression tests for Level-3 AOR support: pseudo-inductive loops and
negative differential resistance (NDR/HNDR).

Scientific basis (project knowledge base §4.2 + AOR EIS literature):
the low-frequency 4th-quadrant inductive loop is the definitive kinetic
fingerprint of adsorbed-intermediate coverage relaxation; past the current
peak the relaxation resistance turns negative (2nd-quadrant arc, precursor
of the Hopf oscillatory regime documented for MeOH and 2-propanol).

Covers:
  A — circuit catalog: AOR_PSEUDOINDUCTIVE / AOR_NDR models + lookup hints
  B — CNLSFitter: allow_negative_r mode (bounds, recovery of L and R<0)
  C — PhysicsInformedLoss: mode="electrocatalysis" (passivity HF-only,
      no Re-monotonicity penalty on inductive spectra)
  D — AORDatasetGenerator: labels 5/6 produce inductive / NDR spectra
"""

import numpy as np
import pytest

from impedance.models.circuits import CustomCircuit

from eisforge.catalogs.circuit_models import CircuitCatalog, lookup_circuit
from eisforge.core.fitter import CNLSFitter
from eisforge.parsers.base_parser import EISDataset
from eisforge.ml.eis_gpt.aor_dataset_generator import (
    AORDatasetGenerator,
    AOR_CIRCUIT_LIBRARY,
    AOR_FREQUENCIES,
)

torch = pytest.importorskip("torch")
from eisforge.ml.eis_gpt.physics_loss import PhysicsInformedLoss  # noqa: E402


FREQ = np.logspace(-2, 5, 60)  # ascending, 10 mHz -> 100 kHz
AOR_CIRCUIT = "R0-p(CPE1,R1-p(R2,L2))"


def simulate(params, freq=FREQ, noise=0.0, seed=0):
    """Simulate the nested AOR faradaic circuit; returns complex Z."""
    c = CustomCircuit(circuit=AOR_CIRCUIT, initial_guess=list(params))
    Z = c.predict(freq)
    if noise:
        rng = np.random.default_rng(seed)
        Z = Z + rng.normal(0, noise * np.abs(Z)) \
              + 1j * rng.normal(0, noise * np.abs(Z))
    return Z


# params: [R0, CPE_Q, CPE_n, R1, R2, L2]
INDUCTIVE_TRUE = [8.0, 2e-4, 0.90, 60.0, 40.0, 30.0]     # 4th-quadrant loop
# Visible NDR needs |R2| > R1 so the DC faradaic resistance R1+R2 < 0 and the
# low-frequency arc enters the 2nd quadrant. |R2| < R1 is the HIDDEN-NDR
# (HNDR) case: the negative branch exists but Re(Z) stays positive.
NDR_TRUE       = [8.0, 2e-4, 0.90, 60.0, -90.0, 30.0]    # 2nd-quadrant arc


# ═══ A — Circuit catalog ═════════════════════════════════════════════════════

class TestCatalog:
    def test_inductive_models_exist_and_parse(self):
        m = CircuitCatalog.AOR_PSEUDOINDUCTIVE
        assert "L2" in m.notation
        assert m.elements == ("R0", "p(CPE1,R1-p(R2,L2))")

    def test_lookup_inductive_hint(self):
        m = lookup_circuit("noble_metal", "post-onset", "acidic",
                           inductive_loop=True)
        assert m is CircuitCatalog.AOR_PSEUDOINDUCTIVE

    def test_lookup_ndr_hint_wins(self):
        m = lookup_circuit("carbon_material", "post-onset", "alkaline",
                           inductive_loop=True, negative_resistance=True)
        assert m is CircuitCatalog.AOR_NDR
        assert "allow_negative_r=True" in m.rationale

    def test_legacy_lookup_unchanged(self):
        m = lookup_circuit("noble_metal", "onset", "acidic")
        assert m is CircuitCatalog.TWO_RC_WARBURG


# ═══ B — CNLS fitter ═════════════════════════════════════════════════════════

class TestFitterInductive:
    def test_inductive_spectrum_has_fourth_quadrant(self):
        Z = simulate(INDUCTIVE_TRUE)
        # 4th quadrant in Nyquist convention: -Im(Z) < 0  ⇔  Im(Z) > 0
        assert np.any(Z.imag > 0), "synthetic loop must dip below the axis"

    def test_fit_recovers_inductance(self):
        Z = simulate(INDUCTIVE_TRUE, noise=0.005)
        # EISDataset stores z_imag in the −Im(Z) (Nyquist) convention
        ds = EISDataset(frequency=FREQ, z_real=Z.real, z_imag=-Z.imag,
                        metadata={})
        # nested R1-(R2||L2) branches are strongly parameter-correlated;
        # start within ~30% (typical practice: seed from catalog p0 or a
        # previous potential step)
        p0 = [v * f for v, f in zip(INDUCTIVE_TRUE,
                                    [0.8, 1.3, 0.95, 1.3, 0.7, 1.4])]
        fitter = CNLSFitter(AOR_CIRCUIT, p0, allow_negative_r=True,
                            remove_outliers=False)
        res = fitter.fit(ds)
        assert res.converged
        fitted = list(res.parameters.values())
        assert fitted[3] == pytest.approx(INDUCTIVE_TRUE[3], rel=0.15)  # R1
        assert fitted[4] == pytest.approx(INDUCTIVE_TRUE[4], rel=0.20)  # R2
        assert fitted[5] == pytest.approx(INDUCTIVE_TRUE[5], rel=0.25)  # L2

    def test_fit_recovers_negative_resistance(self):
        Z = simulate(NDR_TRUE, noise=0.005)
        assert np.any(Z.real < 0), "NDR spectrum must enter the 2nd quadrant"
        # EISDataset stores z_imag in the −Im(Z) (Nyquist) convention
        ds = EISDataset(frequency=FREQ, z_real=Z.real, z_imag=-Z.imag,
                        metadata={})
        p0 = [v * f for v, f in zip(NDR_TRUE,
                                    [0.8, 1.3, 0.95, 1.3, 0.7, 1.4])]
        fitter = CNLSFitter(AOR_CIRCUIT, p0, allow_negative_r=True,
                            remove_outliers=False)
        res = fitter.fit(ds)
        assert res.converged
        r2 = list(res.parameters.values())[4]
        assert r2 < 0, "fitted coverage-relaxation resistance must stay negative"
        assert r2 == pytest.approx(NDR_TRUE[4], rel=0.25)

    def test_default_mode_cannot_go_negative(self):
        """Without allow_negative_r the legacy non-negative bounds hold —
        documents that the flag is required for NDR fitting."""
        Z = simulate(NDR_TRUE, noise=0.005)
        # EISDataset stores z_imag in the −Im(Z) (Nyquist) convention
        ds = EISDataset(frequency=FREQ, z_real=Z.real, z_imag=-Z.imag,
                        metadata={})
        p0 = [8.0, 2e-4, 0.90, 60.0, 10.0, 30.0]
        fitter = CNLSFitter(AOR_CIRCUIT, p0, remove_outliers=False)
        res = fitter.fit(ds)
        r2 = list(res.parameters.values())[4]
        assert r2 >= 0

    def test_series_rs_stays_nonnegative_in_electrocatalysis_mode(self):
        lb, ub = CNLSFitter(AOR_CIRCUIT, [1] * 6, allow_negative_r=True) \
            ._build_electrocatalysis_bounds(
                ["R0", "CPE1_0", "CPE1_1", "R1", "R2", "L2"])
        assert lb[0] == 0.0                      # series R_s >= 0
        assert lb[3] == -np.inf and lb[4] == -np.inf   # faradaic R's free
        assert lb[5] > 0                          # L > 0
        assert ub[2] == 1.0                       # CPE exponent <= 1


# ═══ C — Physics-informed loss ═══════════════════════════════════════════════

def _to_batch(Z):
    zr = torch.tensor(Z.real, dtype=torch.float64).unsqueeze(0)
    zi = torch.tensor(-Z.imag, dtype=torch.float64).unsqueeze(0)  # −Im convention
    f = torch.tensor(FREQ, dtype=torch.float64).unsqueeze(0)
    return zr, zi, f


class TestPhysicsLossModes:
    def test_general_mode_penalises_valid_ndr(self):
        """Documents the OLD behaviour: a physically valid NDR spectrum is
        (wrongly, for AOR) penalised in general mode."""
        Z = simulate(NDR_TRUE)
        zr, zi, f = _to_batch(Z)
        out = PhysicsInformedLoss(mode="general")(zr, zi, zr, zi, f)
        assert out["passivity"].item() > 0

    def test_electrocatalysis_mode_accepts_ndr(self):
        Z = simulate(NDR_TRUE)
        zr, zi, f = _to_batch(Z)
        out = PhysicsInformedLoss(mode="electrocatalysis")(zr, zi, zr, zi, f)
        assert out["passivity"].item() == pytest.approx(0.0, abs=1e-12)

    def test_electrocatalysis_mode_still_guards_hf_solution_resistance(self):
        Z = simulate(NDR_TRUE).copy()
        Z_bad = Z.copy()
        Z_bad.real[-5:] = -5.0      # unphysical: negative R_s at high f
        zr, zi, f = _to_batch(Z_bad)
        out = PhysicsInformedLoss(mode="electrocatalysis")(zr, zi, zr, zi, f)
        assert out["passivity"].item() > 0

    def test_electrocatalysis_kk_does_not_punish_inductive_curl(self):
        """The Re-monotonicity proxy fires on inductive loops in general
        mode; electrocatalysis mode must be markedly gentler."""
        Z = simulate(INDUCTIVE_TRUE)
        zr, zi, f = _to_batch(Z)
        kk_gen = PhysicsInformedLoss(mode="general")(zr, zi, zr, zi, f)["kk"].item()
        kk_cat = PhysicsInformedLoss(mode="electrocatalysis")(zr, zi, zr, zi, f)["kk"].item()
        assert kk_cat < kk_gen

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            PhysicsInformedLoss(mode="battery-ish")


# ═══ D — Dataset generator ═══════════════════════════════════════════════════

class TestGeneratorInductive:
    @pytest.fixture(scope="class")
    def records(self):
        gen = AORDatasetGenerator(n_samples_per_circuit=25, seed=7)
        return gen.generate(verbose=False)

    def test_library_contains_new_labels(self):
        labels = {c["label"] for c in AOR_CIRCUIT_LIBRARY}
        assert {5, 6} <= labels

    def test_labels_5_and_6_generated(self, records):
        labels = {r.circuit_label for r in records}
        assert 5 in labels and 6 in labels

    def test_pseudoinductive_records_show_fourth_quadrant(self, records):
        recs5 = [r for r in records if r.circuit_label == 5]
        # z_imag is stored as −Im(Z); a 4th-quadrant loop → z_imag < 0 at low f
        frac = np.mean([np.any(r.z_imag < 0) for r in recs5])
        assert frac > 0.5, "most pseudo-inductive samples must show the loop"

    def test_ndr_records_have_negative_lowf_real_but_positive_hf(self, records):
        recs6 = [r for r in records if r.circuit_label == 6]
        assert recs6
        for r in recs6:
            assert r.z_real[-1] > 0, "R_s (highest f) must stay positive"
        crossing = np.array([np.any(r.z_real < 0) for r in recs6])
        # visible NDR (|R2| > R1: Re crosses 0) AND hidden NDR (HNDR,
        # |R2| < R1: Re stays positive) are both physical — the training
        # set must contain both subpopulations.
        assert crossing.mean() > 0.15, "need visible-NDR samples (Re<0)"
        assert crossing.mean() < 1.00 or len(recs6) < 5, \
            "hidden-NDR samples (Re>0 throughout) should also appear"

    def test_ndr_sampled_r2_is_negative(self, records):
        recs6 = [r for r in records if r.circuit_label == 6]
        assert all(r.parameters["R2"] < 0 for r in recs6)

    def test_legacy_circuits_still_generated(self, records):
        labels = {r.circuit_label for r in records}
        assert {0, 1, 2, 3, 4} <= labels
