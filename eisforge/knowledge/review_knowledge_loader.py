"""
review_knowledge_loader.py
--------------------------
Bridge between parsed review.html JSON output and LiteratureEngine.

نویسنده: EISForge
تاریخ: June 2026

کاربرد:
-------
وقتی کاربر یک سیستم الکتروشیمیایی را مشخص می‌کند،
این ماژول از output_review.json (خروجی parse_review_html)
مقالات مرتبط را پیدا می‌کند و:
  ۱. بازه پتانسیل، تافل سلوپ و Rct را از ادبیات واقعی استخراج می‌کند
  ۲. حدس اولیه پارامترهای EIS را می‌دهد
  ۳. مراجع مرتبط را لیست می‌کند
  ۴. این اطلاعات را در قالب LiteratureEngine قابل استفاده می‌کند

مثال استفاده:
    from eisforge.knowledge.review_knowledge_loader import ReviewKnowledgeLoader

    loader = ReviewKnowledgeLoader('output_review.json')
    result = loader.query(keywords=['methanol', 'Pt', 'acidic'])
    result.report()
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass (سازگار با LiteratureGuess)
# ---------------------------------------------------------------------------

@dataclass
class ReviewQueryResult:
    """
    نتیجه جستجو در ادبیات واقعی (review.html).
    """
    system_found: bool
    matched_papers: list = field(default_factory=list)  # list of dicts
    n_papers: int = 0

    # آمار پتانسیل‌ها
    potential_mean: Optional[float] = None
    potential_std: Optional[float] = None
    potential_min: Optional[float] = None
    potential_max: Optional[float] = None
    potential_unit: str = "V"

    # آمار Tafel slope
    tafel_mean: Optional[float] = None
    tafel_std: Optional[float] = None
    tafel_unit: str = "mV/dec"

    # آمار Rct
    rct_mean: Optional[float] = None
    rct_std: Optional[float] = None
    rct_unit: str = "Ω"

    # حدس اولیه برای EIS fitting
    initial_guess: dict = field(default_factory=dict)

    references: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    confidence: str = "low"

    def report(self) -> str:
        """گزارش کامل نتیجه جستجو."""
        lines = [
            "═" * 62,
            f"  📚 جستجو در ادبیات واقعی — {self.n_papers} مقاله یافت شد",
            f"  اطمینان: {self._confidence_fa()}",
            "═" * 62,
        ]

        if not self.system_found:
            lines.append("  ⚠️ هیچ مقاله مرتبطی پیدا نشد. کلیدواژه‌ها را تغییر دهید.")
            lines.append("═" * 62)
            return "\n".join(lines)

        # پتانسیل‌ها
        if self.potential_mean is not None:
            lines += [
                "",
                "  ⚡ پتانسیل‌های مشاهده‌شده در ادبیات:",
                f"     میانگین : {self.potential_mean:+.3f} {self.potential_unit}",
                f"     انحراف  : ± {self.potential_std:.3f} {self.potential_unit}" if self.potential_std else "",
                f"     بازه    : [{self.potential_min:+.3f} , {self.potential_max:+.3f}] {self.potential_unit}",
            ]

        # Tafel slope
        if self.tafel_mean is not None:
            lines += [
                "",
                "  📐 Tafel Slope از ادبیات:",
                f"     میانگین : {self.tafel_mean:.1f} {self.tafel_unit}",
                f"     انحراف  : ± {self.tafel_std:.1f} {self.tafel_unit}" if self.tafel_std else "",
            ]

        # Rct
        if self.rct_mean is not None:
            lines += [
                "",
                "  🔋 مقاومت انتقال بار (Rct) از ادبیات:",
                f"     میانگین : {self.rct_mean:.2e} {self.rct_unit}",
                f"     انحراف  : ± {self.rct_std:.2e} {self.rct_unit}" if self.rct_std else "",
            ]

        # حدس اولیه
        if self.initial_guess:
            lines += ["", "  🎯 حدس اولیه پارامترهای EIS (از آمار ادبیات):"]
            for k, v in self.initial_guess.items():
                lines.append(f"     {k:<12} = {v:.3e}")

        # هشدارها
        if self.warnings:
            lines += ["", "  ⚠️  هشدارها:"]
            for w in self.warnings:
                lines.append(f"     • {w}")

        # مراجع
        if self.references:
            lines += ["", f"  📖 مراجع ({len(self.references)} مقاله):"]
            for r in self.references[:10]:  # نمایش حداکثر 10 مرجع
                lines.append(f"     • {r}")
            if len(self.references) > 10:
                lines.append(f"     ... و {len(self.references) - 10} مقاله دیگر")

        lines.append("═" * 62)
        return "\n".join(line for line in lines if line is not None)

    def _confidence_fa(self) -> str:
        return {"high": "✅ بالا", "medium": "⚠️ متوسط", "low": "❓ پایین"}.get(
            self.confidence, self.confidence
        )


# ---------------------------------------------------------------------------
# ReviewKnowledgeLoader
# ---------------------------------------------------------------------------

class ReviewKnowledgeLoader:
    """
    بارگذاری و جستجو در خروجی JSON مرور ادبیات (review.html).

    Parameters
    ----------
    json_path : str | Path
        مسیر فایل output_review.json که توسط parse_review_html.py تولید شده.
    """

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        self._records: list[dict] = self._load()
        print(f"[ReviewKnowledgeLoader] بارگذاری شد: {len(self._records)} مقاله")

    def _load(self) -> list[dict]:
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"فایل JSON پیدا نشد: {self.json_path}\n"
                "ابتدا اجرا کنید: python -m eisforge.knowledge.parse_review_html review.html output_review.json"
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
        جستجو بر اساس کلیدواژه‌ها در عنوان مقالات.

        Parameters
        ----------
        keywords : list[str]
            کلیدواژه‌ها (مثلاً ['methanol', 'Pt', 'acidic'])
        min_papers : int
            حداقل تعداد مقاله برای اطمینان بالا.

        Returns
        -------
        ReviewQueryResult
        """
        matched = self._search(keywords)

        if not matched:
            return ReviewQueryResult(system_found=False, n_papers=0)

        # جمع‌آوری داده‌های عددی
        potentials = []
        tafel_slopes = []
        rct_values = []
        refs = []

        for paper in matched:
            # پتانسیل‌ها
            for p in paper.get("potentials", []):
                try:
                    potentials.append(float(p["value"]))
                except (ValueError, KeyError):
                    pass

            # Tafel slopes
            for t in paper.get("tafel_slopes", []):
                try:
                    v = abs(float(t["value"]))  # absolute value
                    if 20 < v < 500:  # فیلتر مقادیر غیرمنطقی
                        tafel_slopes.append(v)
                except (ValueError, KeyError):
                    pass

            # Rct
            for r in paper.get("rct_values", []):
                try:
                    v = float(r["value"])
                    if v > 0:
                        rct_values.append(v)
                except (ValueError, KeyError):
                    pass

            # مرجع
            title = paper.get("title", "")
            doi = paper.get("doi", "")
            if doi:
                refs.append(f"{title[:55]}... | DOI: {doi}")
            else:
                refs.append(title[:70])

        # آمار
        pot_stats = self._stats(potentials)
        tafel_stats = self._stats(tafel_slopes)
        rct_stats = self._stats(rct_values)

        # حدس اولیه برای EIS
        initial_guess = {}
        if rct_stats["mean"] is not None:
            initial_guess["R_ct"] = rct_stats["mean"]
        if pot_stats["mean"] is not None:
            initial_guess["E_onset"] = pot_stats["mean"]

        # سطح اطمینان
        n = len(matched)
        if n >= min_papers * 3 and (potentials or tafel_slopes):
            confidence = "high"
        elif n >= min_papers:
            confidence = "medium"
        else:
            confidence = "low"

        # هشدارها
        warnings = []
        if tafel_stats["mean"] and tafel_stats["mean"] > 120:
            warnings.append(
                f"Tafel slope بالا ({tafel_stats['mean']:.0f} mV/dec) — "
                "احتمالاً مکانیسم دو مرحله‌ای یا تاثیر CO poisoning."
            )
        if rct_stats["std"] and rct_stats["mean"] and rct_stats["std"] / rct_stats["mean"] > 1.5:
            warnings.append(
                "پراکندگی بالای Rct در ادبیات — "
                "احتمالاً شرایط آزمایشگاهی یا کاتالیست‌های متفاوت."
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
        """لیست همه DOIهای موجود در دیتابیس."""
        return [r["doi"] for r in self._records if r.get("doi")]

    def get_paper_by_doi(self, doi: str) -> Optional[dict]:
        """پیدا کردن مقاله با DOI."""
        for r in self._records:
            if r.get("doi") == doi:
                return r
        return None

    def stats_summary(self) -> None:
        """آمار کلی دیتابیس."""
        total_pots = sum(len(r.get("potentials", [])) for r in self._records)
        total_tafel = sum(len(r.get("tafel_slopes", [])) for r in self._records)
        total_rct = sum(len(r.get("rct_values", [])) for r in self._records)
        with_doi = sum(1 for r in self._records if r.get("doi"))

        print("═" * 50)
        print(f"  📊 آمار دیتابیس مرور ادبیات")
        print("═" * 50)
        print(f"  مقالات کل       : {len(self._records)}")
        print(f"  دارای DOI        : {with_doi} ({100*with_doi//len(self._records)}%)")
        print(f"  پتانسیل‌ها       : {total_pots} داده")
        print(f"  Tafel Slope      : {total_tafel} داده")
        print(f"  Rct              : {total_rct} داده")
        print("═" * 50)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search(self, keywords: list[str]) -> list[dict]:
        """جستجوی مقالات بر اساس کلیدواژه‌ها در عنوان."""
        kws = [k.lower() for k in keywords]
        matched = []
        for record in self._records:
            title_lower = record.get("title", "").lower()
            if any(kw in title_lower for kw in kws):
                matched.append(record)
        return matched

    @staticmethod
    def _stats(values: list) -> dict:
        """محاسبه آمار پایه."""
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python review_knowledge_loader.py <output_review.json> <keyword1> [keyword2] ...")
        print("Example: python review_knowledge_loader.py output_review.json methanol Pt acidic")
        sys.exit(1)

    loader = ReviewKnowledgeLoader(sys.argv[1])
    loader.stats_summary()

    keywords = sys.argv[2:]
    print(f"\nجستجو برای: {keywords}")
    result = loader.query(keywords)
    print(result.report())
