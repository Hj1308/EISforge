"""
Active Learner — Intelligent Data Acquisition for EIS-GPT.
Author: Hoda Jafari | May 2026

The model tells you WHICH experiment to label next, based on where it is most uncertain.
For B4C: model will flag high uncertainty on first encounter → guides you to collect
exactly the data needed to improve B4C predictions.

Workflow:
  1. Upload EIS spectrum
  2. evaluate() → runs MC Dropout
  3. Confident? → show prediction directly
  4. Uncertain? → flag it, ask for CNLS fit
  5. add_label() → save to pool
  6. When pool is large enough → retrain model
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from eisforge.ml.uncertainty.mc_dropout import UncertaintyResult, mc_dropout_predict

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    """Result of evaluating one spectrum through the Active Learning system."""
    uncertainty        : UncertaintyResult
    predicted_circuit  : str
    confidence_pct     : int
    confidence_label   : str
    should_query       : bool
    epistemic_score    : float
    aleatoric_score    : float
    threshold_used     : float

    @property
    def status_color(self) -> str:
        if not self.should_query: return "green"
        elif self.epistemic_score < self.threshold_used * 1.5: return "yellow"
        else: return "red"

    @property
    def ui_message(self) -> str:
        if not self.should_query:
            return (f"EIS-GPT is confident — {self.predicted_circuit} "
                    f"({self.confidence_pct}% confidence)")
        elif self.epistemic_score < self.threshold_used * 1.5:
            return ("EIS-GPT is moderately uncertain. "
                    "Consider verifying with CNLS fit.")
        else:
            return ("EIS-GPT has not seen enough spectra like this one. "
                    "Please run CNLS fit — your label will improve the model.")

    @property
    def param_uncertainty_bars(self) -> list[dict]:
        bars = []
        for key, val in self.uncertainty.per_param_uncertainty.items():
            ep, al = val["epistemic_std"], val["aleatoric_std"]
            bars.append({
                "param"    : key,
                "value"    : val["value_estimate"],
                "epistemic": ep,
                "aleatoric": al,
                "total"    : ep + al,
                "high_unc" : (ep + al) > 0.5,
            })
        return bars


@dataclass
class LabeledSample:
    """One labeled EIS spectrum stored in the training pool."""
    spectrum_id        : str
    timestamp          : float
    catalyst_type      : str
    electrolyte        : str
    circuit_name       : str
    parameters         : dict
    epistemic_at_query : float
    source             : str = "user"
    notes              : str = ""


class ActiveLearner:
    """
    Active Learning orchestrator for EIS-GPT.

    Parameters
    ----------
    model : EISForgeModel
    epistemic_threshold : float
        Uncertainty above which model queries the user. Default: 0.15.
    n_mc_samples : int
        MC Dropout passes per inference. Default: 50.
    min_pool_for_retrain : int
        New labels needed before suggesting retraining. Default: 10.
    save_pool_path : str or None
        Auto-save labeled pool to this JSON file.
    """

    def __init__(
        self,
        model,
        epistemic_threshold  : float = 0.15,
        n_mc_samples         : int   = 50,
        min_pool_for_retrain : int   = 10,
        save_pool_path       : Optional[str] = None,
    ) -> None:
        self.model                = model
        self.epistemic_threshold  = epistemic_threshold
        self.n_mc_samples         = n_mc_samples
        self.min_pool_for_retrain = min_pool_for_retrain
        self.save_pool_path       = save_pool_path
        self.labeled_pool         : list[LabeledSample] = []
        self.query_history        : list[dict] = []
        self.eval_history         : list[dict] = []
        self._new_since_retrain   = 0

    def evaluate(self, freq, z_real, z_imag) -> Evaluation:
        """
        Main entry point — evaluate a new EIS spectrum.
        Returns prediction + confidence + whether to query the user.
        """
        unc = mc_dropout_predict(self.model, freq, z_real, z_imag,
                                 n_samples=self.n_mc_samples)
        should_query = unc.should_query(self.epistemic_threshold)

        self.eval_history.append({
            "timestamp": time.time(),
            "epistemic": unc.epistemic_score,
            "aleatoric": unc.aleatoric_score,
            "circuit"  : unc.predicted_circuit,
            "queried"  : should_query,
        })

        if should_query:
            logger.info(f"Query triggered: epistemic={unc.epistemic_score:.3f} "
                        f"> threshold={self.epistemic_threshold:.3f}")

        return Evaluation(
            uncertainty       = unc,
            predicted_circuit = unc.predicted_circuit,
            confidence_pct    = unc.confidence_pct(self.epistemic_threshold),
            confidence_label  = unc.confidence_label(self.epistemic_threshold),
            should_query      = should_query,
            epistemic_score   = unc.epistemic_score,
            aleatoric_score   = unc.aleatoric_score,
            threshold_used    = self.epistemic_threshold,
        )

    def add_label(
        self,
        freq              : torch.Tensor,
        z_real            : torch.Tensor,
        z_imag            : torch.Tensor,
        circuit_name      : str,
        parameters        : dict,
        epistemic_at_query: float = 0.0,
        catalyst_type     : str = "unknown",
        electrolyte       : str = "unknown",
        notes             : str = "",
    ) -> LabeledSample:
        """
        Add a user-labeled spectrum to the training pool.

        Call this after the user runs CNLS fit and provides
        circuit name + parameter values.
        """
        sample = LabeledSample(
            spectrum_id        = f"spectrum_{int(time.time())}",
            timestamp          = time.time(),
            catalyst_type      = catalyst_type,
            electrolyte        = electrolyte,
            circuit_name       = circuit_name,
            parameters         = parameters,
            epistemic_at_query = epistemic_at_query,
            source             = "user",
            notes              = notes,
        )
        self.labeled_pool.append(sample)
        self._new_since_retrain += 1
        self.query_history.append({
            "timestamp" : sample.timestamp,
            "circuit"   : circuit_name,
            "epistemic" : epistemic_at_query,
            "catalyst"  : catalyst_type,
        })
        logger.info(f"Label added: {circuit_name} | pool size={len(self.labeled_pool)}")
        if self.save_pool_path:
            self._save_pool()
        return sample

    def add_from_literature(self, circuit_name, parameters, source_ref,
                             catalyst="unknown", electrolyte="unknown") -> None:
        """Bootstrap the pool with labeled samples from literature."""
        self.labeled_pool.append(LabeledSample(
            spectrum_id=f"lit_{int(time.time())}", timestamp=time.time(),
            catalyst_type=catalyst, electrolyte=electrolyte,
            circuit_name=circuit_name, parameters=parameters,
            epistemic_at_query=0.0, source="literature", notes=source_ref,
        ))
        self._new_since_retrain += 1

    @property
    def ready_to_retrain(self) -> bool:
        return self._new_since_retrain >= self.min_pool_for_retrain

    @property
    def pool_size(self) -> int:
        return len(self.labeled_pool)

    def mark_retrained(self) -> None:
        self._new_since_retrain = 0
        logger.info(f"Marked as retrained. Pool size: {self.pool_size}")

    def stats(self) -> dict:
        if not self.eval_history:
            return {"total_evaluations": 0, "total_queries": 0, "pool_size": 0}
        ep_vals  = [e["epistemic"] for e in self.eval_history]
        n_q      = sum(1 for e in self.eval_history if e["queried"])
        circuits  = {}
        catalysts = {}
        for s in self.labeled_pool:
            circuits[s.circuit_name] = circuits.get(s.circuit_name, 0) + 1
            catalysts[s.catalyst_type] = catalysts.get(s.catalyst_type, 0) + 1
        return {
            "total_evaluations"   : len(self.eval_history),
            "total_queries"       : n_q,
            "query_rate_pct"      : int(n_q / len(self.eval_history) * 100),
            "pool_size"           : self.pool_size,
            "new_since_retrain"   : self._new_since_retrain,
            "ready_to_retrain"    : self.ready_to_retrain,
            "mean_epistemic"      : float(np.mean(ep_vals)),
            "circuits_in_pool"    : circuits,
            "catalysts_in_pool"   : catalysts,
            "threshold"           : self.epistemic_threshold,
        }

    def export_training_tensors(self) -> Optional[dict]:
        """Export labeled pool as PyTorch tensors for retraining."""
        from eisforge.ml.eis_gpt.transformer import CIRCUIT_NAMES, MAX_PARAMS
        if not self.labeled_pool:
            return None
        circuit_labels, param_values = [], []
        for sample in self.labeled_pool:
            idx = CIRCUIT_NAMES.index(sample.circuit_name) if sample.circuit_name in CIRCUIT_NAMES else 0
            circuit_labels.append(idx)
            vals = list(sample.parameters.values())[:MAX_PARAMS]
            while len(vals) < MAX_PARAMS:
                vals.append(1.0)
            param_values.append([np.log(max(abs(v), 1e-10)) for v in vals])
        return {
            "circuit_labels": torch.LongTensor(circuit_labels),
            "param_values"  : torch.FloatTensor(param_values),
            "n_samples"     : len(self.labeled_pool),
        }

    def _save_pool(self) -> None:
        if not self.save_pool_path:
            return
        Path(self.save_pool_path).write_text(json.dumps({
            "pool"          : [asdict(s) for s in self.labeled_pool],
            "query_history" : self.query_history,
            "threshold"     : self.epistemic_threshold,
            "saved_at"      : time.time(),
        }, indent=2))

    def load_pool(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self.labeled_pool  = [LabeledSample(**s) for s in data["pool"]]
        self.query_history = data.get("query_history", [])
        self._new_since_retrain = 0
        logger.info(f"Loaded {len(self.labeled_pool)} samples from {path}")

    def summary(self) -> str:
        s = self.stats()
        lines = [
            "=" * 55,
            "  Active Learning Status — EISForge",
            "=" * 55,
            f"  Threshold             : {s['threshold']:.3f}",
            f"  Total evaluations     : {s['total_evaluations']}",
            f"  Query rate            : {s.get('query_rate_pct', 0)}%",
            f"  Labeled pool size     : {s['pool_size']}",
            f"  New since retrain     : {s['new_since_retrain']}",
            f"  Ready to retrain      : {'Yes' if s['ready_to_retrain'] else 'No'}",
            f"  Mean epistemic unc.   : {s.get('mean_epistemic', 0):.4f}",
            "=" * 55,
        ]
        return "\n".join(lines)
