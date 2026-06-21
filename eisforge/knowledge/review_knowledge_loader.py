"""
review_knowledge_loader.py
--------------------------
Bridge between parsed review.html JSON output and LiteratureEngine.

Author : EISForge
Date   : June 2026

Usage:
    from eisforge.knowledge.review_knowledge_loader import ReviewKnowledgeLoader

    loader = ReviewKnowledgeLoader('output_review.json')
    result = loader.query(keywords=['methanol', 'Pt', 'acidic'])
    print(result.report())
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReviewQueryResult:
    """Query result from the real literature database (review.html)."""

    system_found: bool
    matched_papers: list = field(default_factory=list)
    n_papers: int = 0

    # Potential statistics
    potential_mean: Optional[float] = None
    potential_std:  Optional[float] = None
    potential_min:  Optional[float] = None
    potential_max:  Optional[float] = None
    potential_unit: str = "V"

    # Tafel slope statistics
    tafel_mean: Optional[float] = None
    tafel_std:  Optional[float] = None
    tafel_unit: str = "mV/dec"

    # Rct statistics
    rct_mean: Optional[float] = None
    rct_std:  Optional[float] = None
    rct_unit: str = "Ohm"

    # EIS fitting initial guess
    initial_guess: dict = field(default_factory=dict)

    references: list = field(default_factory=list)
    warnings:   list = field(default_factory=list)
    confidence: str  = "low"

    def report(self) -> str:
        """Return a full English text report of the query result."""
        conf_label = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(
            self.confidence, self.confidence.upper()
        )
        lines = [
            "=" * 62,
            f"  Literature Search  —  {self.n_papers} paper(s) matched",
            f"  Confidence : {conf_label}",
            "=" * 62,
        ]

        if not self.system_found:
            lines.append("  No matching papers found. Try different keywords.")
            lines.append("=" * 62)
            return "\n".join(lines)

        # Potentials
        if self.potential_mean is not None:
            lines += [
                "",
                "  Observed Potentials (from literature):",
                f"    Mean   : {self.potential_mean:+.3f} {self.potential_unit}",
                f"    Std    : +/- {self.potential_std:.3f} {self.potential_unit}" if self.potential_std else "",
                f"    Range  : [{self.potential_min:+.3f} ,  {self.potential_max:+.3f}] {self.potential_unit}",
            ]

        # Tafel slope
        if self.tafel_mean is not None:
            lines += [
                "",
                "  Tafel Slope (from literature):",
                f"    Mean   : {self.tafel_mean:.1f} {self.tafel_unit}",
                f"    Std    : +/- {self.tafel_std:.1f} {self.tafel_unit}" if self.tafel_std else "",
            ]

        # Rct
        if self.rct_mean is not None:
            lines += [
                "",
                "  Charge Transfer Resistance Rct (from literature):",
                f"    Mean   : {self.rct_mean:.2e} {self.rct_unit}",
                f"    Std    : +/- {self.rct_std:.2e} {self.rct_unit}" if self.rct_std else "",
            ]

        # Initial guess
        if self.initial_guess:
            lines += ["", "  EIS Initial Guess (derived from literature statistics):"]
            for k, v in self.initial_guess.items():
                lines.append(f"    {k:<12} = {v:.3e}")

        # Warnings
        if self.warnings:
            lines += ["", "  Warnings:"]
            for w in self.warnings:
                lines.append(f"    * {w}")

        # References
        if self.references:
            lines += ["", f"  References ({len(self.references)} paper(s)):"]
            for r in self.references[:10]:
                lines.append(f"    - {r}")
            if len(self.references) > 10:
                lines.append(f"    ... and {len(self.references) - 10} more paper(s)")

        lines.append("=" * 62)
        return "\n".join(line for line in lines if line is not None)


# ---------------------------------------------------------------------------
# ReviewKnowledgeLoader
# ---------------------------------------------------------------------------

class ReviewKnowledgeLoader:
    """
    Load and search the JSON output produced by parse_review_html.py.

    Parameters
    ----------
    json_path : str | Path
        Path to output_review.json.
    """

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self._records: list[dict] = self._load()
        print(f"[ReviewKnowledgeLoader] Loaded {len(self._records)} papers from {self.json_path.name}")

    def _load(self) -> list[dict]:
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {self.json_path}\n"
                "Run first: python -m eisforge.knowledge.parse_review_html review.html output_review.json"
            )
        with open(self.json_path, encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        keywords: list[str],
        min_papers: int = 3,
    ) -> ReviewQueryResult:
        """
        Search papers by keywords in their titles and compute statistics.

        Parameters
        ----------
        keywords : list[str]
            Keywords to search for (e.g. ['methanol', 'Pt', 'acidic']).
        min_papers : int
            Minimum paper count required for HIGH confidence.

        Returns
        -------
        ReviewQueryResult
        """
        matched = self._search(keywords)

        if not matched:
            return ReviewQueryResult(system_found=False, n_papers=0)

        potentials:   list[float] = []
        tafel_slopes: list[float] = []
        rct_values:   list[float] = []
        refs:         list[str]   = []

        for paper in matched:
            for p in paper.get("potentials", []):
                try:
                    potentials.append(float(p["value"]))
                except (ValueError, KeyError):
                    pass

            for t in paper.get("tafel_slopes", []):
                try:
                    v = abs(float(t["value"]))
                    if 20 < v < 500:          # filter physically unrealistic values
                        tafel_slopes.append(v)
                except (ValueError, KeyError):
                    pass

            for r in paper.get("rct_values", []):
                try:
                    v = float(r["value"])
                    if v > 0:
                        rct_values.append(v)
                except (ValueError, KeyError):
                    pass

            title = paper.get("title", "")
            doi   = paper.get("doi", "")
            refs.append(
                f"{title[:55]}... | DOI: {doi}" if doi else title[:70]
            )

        pot_stats   = self._stats(potentials)
        tafel_stats = self._stats(tafel_slopes)
        rct_stats   = self._stats(rct_values)

        initial_guess: dict = {}
        if rct_stats["mean"] is not None:
            initial_guess["R_ct"]    = rct_stats["mean"]
        if pot_stats["mean"] is not None:
            initial_guess["E_onset"] = pot_stats["mean"]

        n = len(matched)
        if n >= min_papers * 3 and (potentials or tafel_slopes):
            confidence = "high"
        elif n >= min_papers:
            confidence = "medium"
        else:
            confidence = "low"

        warnings: list[str] = []
        if tafel_stats["mean"] and tafel_stats["mean"] > 120:
            warnings.append(
                f"High Tafel slope ({tafel_stats['mean']:.0f} mV/dec) — "
                "possibly a two-step mechanism or CO poisoning effect."
            )
        if (
            rct_stats["std"] and rct_stats["mean"] and
            rct_stats["std"] / rct_stats["mean"] > 1.5
        ):
            warnings.append(
                "High Rct scatter in literature — "
                "likely different experimental conditions or catalyst loadings."
            )

        return ReviewQueryResult(
            system_found=True,
            matched_papers=matched,
            n_papers=n,
            potential_mean=pot_stats["mean"],
            potential_std=pot_stats["std"],
            potential_min=pot_stats["min"],
            potential_max=pot_stats["max"],
            tafel_mean=tafel_stats["mean"],
            tafel_std=tafel_stats["std"],
            rct_mean=rct_stats["mean"],
            rct_std=rct_stats["std"],
            initial_guess=initial_guess,
            references=refs,
            warnings=warnings,
            confidence=confidence,
        )

    def list_dois(self) -> list[str]:
        """Return all DOIs present in the database."""
        return [r["doi"] for r in self._records if r.get("doi")]

    def get_paper_by_doi(self, doi: str) -> Optional[dict]:
        """Find a paper by its DOI."""
        for r in self._records:
            if r.get("doi") == doi:
                return r
        return None

    def stats_summary(self) -> None:
        """Print a summary of database contents."""
        total_pots  = sum(len(r.get("potentials",    [])) for r in self._records)
        total_tafel = sum(len(r.get("tafel_slopes",  [])) for r in self._records)
        total_rct   = sum(len(r.get("rct_values",    [])) for r in self._records)
        with_doi    = sum(1 for r in self._records if r.get("doi"))

        print("=" * 50)
        print("  Literature Database — Summary")
        print("=" * 50)
        print(f"  Total papers     : {len(self._records)}")
        print(f"  Papers with DOI  : {with_doi}  ({100 * with_doi // len(self._records)}%)")
        print(f"  Potential data   : {total_pots} entries")
        print(f"  Tafel slope data : {total_tafel} entries")
        print(f"  Rct data         : {total_rct} entries")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search(self, keywords: list[str]) -> list[dict]:
        """Match papers whose title contains at least one keyword."""
        kws = [k.lower() for k in keywords]
        return [
            rec for rec in self._records
            if any(kw in rec.get("title", "").lower() for kw in kws)
        ]

    @staticmethod
    def _stats(values: list) -> dict:
        """Compute basic statistics."""
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "std":  statistics.stdev(values) if len(values) > 1 else 0.0,
            "min":  min(values),
            "max":  max(values),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage : python review_knowledge_loader.py <output_review.json> <keyword1> [keyword2] ...")
        print("Example: python review_knowledge_loader.py output_review.json methanol Pt acidic")
        sys.exit(1)

    loader = ReviewKnowledgeLoader(sys.argv[1])
    loader.stats_summary()

    kws = sys.argv[2:]
    print(f"\nSearching for: {kws}\n")
    result = loader.query(kws)
    print(result.report())
