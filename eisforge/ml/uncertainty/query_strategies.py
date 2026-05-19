"""
Query Strategies — How to Choose the Next Spectrum to Label.
Author: Hoda Jafari | May 2026

Given a pool of unlabeled EIS spectra, these strategies decide
which one to ask the user to label next. Different strategies
optimize for different goals.

Available strategies:
---------------------
  1. UncertaintySampling   — pick the most uncertain spectrum (simplest, works well)
  2. MaxEntropyStrategy    — pick the spectrum with highest output entropy
  3. BALDStrategy          — Bayesian Active Learning by Disagreement (most rigorous)
  4. DiversitySampling     — pick spectra that cover different regions of data space

For EISForge the recommended strategy is:
    UncertaintySampling for everyday use
    BALDStrategy        for careful scientific data collection (B4C, new catalysts)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch

from eisforge.ml.uncertainty.mc_dropout import mc_dropout_predict


class QueryStrategy(ABC):
    """Abstract base class for Active Learning query strategies."""

    @abstractmethod
    def score(
        self,
        model,
        freq   : torch.Tensor,
        z_real : torch.Tensor,
        z_imag : torch.Tensor,
        n_mc   : int = 30,
    ) -> float:
        """
        Compute a score for one spectrum.
        Higher score = higher priority for labeling.
        """

    def rank(
        self,
        model,
        spectra_list: list[tuple],
        n_mc        : int = 30,
    ) -> list[tuple[int, float]]:
        """
        Rank a list of unlabeled spectra.

        Parameters
        ----------
        spectra_list : list of (freq, z_real, z_imag) tuples.

        Returns
        -------
        list of (index, score) sorted descending — highest priority first.
        """
        scores = []
        for idx, (freq, z_real, z_imag) in enumerate(spectra_list):
            s = self.score(model, freq, z_real, z_imag, n_mc=n_mc)
            scores.append((idx, s))
        return sorted(scores, key=lambda x: x[1], reverse=True)


class UncertaintySampling(QueryStrategy):
    """
    Simplest and most effective strategy for most cases.

    Selects the spectrum where the model's epistemic uncertainty is highest.
    Directly uses MC Dropout variance as the score.

    When to use:
    - General use across all catalyst types
    - When you want a simple, interpretable strategy
    - When computation time matters

    Reference: Lewis & Gale (1994). A Sequential Algorithm for Training
    Text Classifiers. SIGIR.
    """

    def score(self, model, freq, z_real, z_imag, n_mc=30) -> float:
        result = mc_dropout_predict(model, freq, z_real, z_imag, n_samples=n_mc)
        return result.epistemic_score


class MaxEntropyStrategy(QueryStrategy):
    """
    Selects the spectrum with highest entropy in the circuit prediction.

    Entropy measures how "spread out" the probability distribution is:
    - Low entropy  → model is confident (one circuit has high probability)
    - High entropy → model is confused (all circuits equally likely)

    When to use:
    - When you care mainly about circuit topology classification
    - When you have many new catalyst types

    H(y|x) = -Σ p(c|x) log p(c|x)
    """

    def score(self, model, freq, z_real, z_imag, n_mc=30) -> float:
        result = mc_dropout_predict(model, freq, z_real, z_imag, n_samples=n_mc)
        probs  = result.circuit_probs_mean + 1e-10   # avoid log(0)
        entropy = -float(np.sum(probs * np.log(probs)))
        return entropy


class BALDStrategy(QueryStrategy):
    """
    Bayesian Active Learning by Disagreement (BALD).

    The most principled Active Learning strategy for Bayesian models.

    BALD = Total entropy - Expected entropy under the posterior
         = H[y|x] - E_θ[H[y|x,θ]]

    Intuition:
    - Total entropy (H[y|x]): how uncertain is the MEAN prediction?
    - Expected entropy (E[H[y|x,θ]]): how uncertain are INDIVIDUAL MC samples?
    - BALD = difference: how much does the model DISAGREE WITH ITSELF?

    High BALD = the model's different "versions" (via dropout) strongly
    disagree → this is a genuinely informative region for learning.

    When to use:
    - For B4C and new metal-free catalysts (novel regime)
    - When you want to maximize information gain with minimum labels
    - When labeling is expensive (requires careful CNLS fitting)

    Reference: Houlsby et al. (2011). Bayesian Active Learning for
    Classification and Preference Learning. ArXiv.
    """

    def score(self, model, freq, z_real, z_imag, n_mc=50) -> float:
        # Need more MC samples for reliable BALD estimation
        n_mc = max(n_mc, 50)
        result = mc_dropout_predict(model, freq, z_real, z_imag, n_samples=n_mc)

        # ── Total entropy of mean prediction ──────────────────────────────────
        probs_mean = result.circuit_probs_mean + 1e-10
        H_mean = -float(np.sum(probs_mean * np.log(probs_mean)))

        # ── Expected entropy across MC samples ────────────────────────────────
        # We approximate this from the std across MC samples
        # High std → samples disagree → high expected entropy
        probs_std = result.circuit_probs_std
        H_expected_approx = float(np.sum(probs_std))

        # BALD = Total entropy - Expected entropy
        bald = H_mean - H_expected_approx
        return max(0.0, bald)   # BALD is always >= 0


class DiversitySampling(QueryStrategy):
    """
    Selects spectra that are most different from already-labeled spectra.

    Prevents the labeled pool from being dominated by similar spectra.
    Ensures the model learns from diverse examples.

    When to use:
    - After you already have some labeled data
    - When you notice the model keeps asking for similar spectra
    - Combine with UncertaintySampling for best results

    Uses CLS token embeddings as a proxy for spectrum similarity.
    Spectra with features far from labeled pool = diverse = high score.
    """

    def __init__(self, labeled_embeddings: Optional[list] = None) -> None:
        """
        Parameters
        ----------
        labeled_embeddings : list of np.ndarray (d_model,)
            CLS token embeddings of already-labeled spectra.
            If None, returns 0 for all (no diversity pressure yet).
        """
        self.labeled_embeddings = labeled_embeddings or []

    def add_labeled_embedding(self, embedding: np.ndarray) -> None:
        """Call this when a new spectrum is labeled."""
        self.labeled_embeddings.append(embedding)

    def score(self, model, freq, z_real, z_imag, n_mc=1) -> float:
        """
        Score = minimum distance from any labeled spectrum in embedding space.
        High score = far from labeled data = diverse = valuable.
        """
        if not self.labeled_embeddings:
            return 0.0   # no diversity pressure without labeled data

        model.eval()
        with torch.no_grad():
            out = model.forward(freq, z_real, z_imag)
            embedding = out["cls_features"][0].cpu().numpy()   # (d_model,)

        # Minimum distance to any labeled spectrum
        min_dist = float("inf")
        for labeled_emb in self.labeled_embeddings:
            dist = float(np.linalg.norm(embedding - labeled_emb))
            min_dist = min(min_dist, dist)

        return min_dist


class HybridStrategy(QueryStrategy):
    """
    Combines Uncertainty + Diversity for balanced Active Learning.

    score = alpha * uncertainty + (1 - alpha) * diversity

    This prevents the pathological case where the model keeps querying
    the same type of spectrum over and over (uncertainty-only problem).

    When to use:
    - General purpose — best default for production use
    - When you want both informativeness and coverage

    Parameters
    ----------
    alpha : float
        Weight for uncertainty (0 = pure diversity, 1 = pure uncertainty).
        Default: 0.7 (favor uncertainty).
    """

    def __init__(
        self,
        alpha    : float = 0.7,
        diversity: Optional[DiversitySampling] = None,
    ) -> None:
        self.alpha     = alpha
        self.unc       = UncertaintySampling()
        self.diversity = diversity or DiversitySampling()

    def score(self, model, freq, z_real, z_imag, n_mc=30) -> float:
        unc_score = self.unc.score(model, freq, z_real, z_imag, n_mc=n_mc)
        div_score = self.diversity.score(model, freq, z_real, z_imag, n_mc=1)

        # Normalize diversity to same scale as uncertainty
        div_normalized = min(div_score / 100.0, 1.0)

        return self.alpha * unc_score + (1 - self.alpha) * div_normalized


# ── Strategy factory ──────────────────────────────────────────────────────────

STRATEGIES = {
    "uncertainty" : UncertaintySampling,
    "max_entropy" : MaxEntropyStrategy,
    "bald"        : BALDStrategy,
    "diversity"   : DiversitySampling,
    "hybrid"      : HybridStrategy,
}


def get_strategy(name: str = "uncertainty", **kwargs) -> QueryStrategy:
    """
    Get a query strategy by name.

    Parameters
    ----------
    name : str
        One of: 'uncertainty', 'max_entropy', 'bald', 'diversity', 'hybrid'.

    Returns
    -------
    QueryStrategy instance.

    Example
    -------
        strategy = get_strategy('bald')
        scores = strategy.rank(model, spectra_list)
    """
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name](**kwargs)


# ── Recommendation logic for EISForge ─────────────────────────────────────────

def recommend_strategy(catalyst_type: str, pool_size: int) -> str:
    """
    Recommend the best query strategy for the given situation.

    Parameters
    ----------
    catalyst_type : str
        One of: 'noble_metal', 'alloy', 'metal_oxide', 'metal_free'.
    pool_size : int
        Number of already-labeled samples in the pool.

    Returns
    -------
    str — strategy name.
    """
    if catalyst_type == "metal_free":
        # B4C, N-doped C: novel regime, use BALD for max information gain
        return "bald"
    elif pool_size < 20:
        # Not enough labeled data yet — pure uncertainty is best
        return "uncertainty"
    elif pool_size < 100:
        # Have some data — start mixing in diversity
        return "hybrid"
    else:
        # Large pool — BALD for rigorous selection
        return "bald"
