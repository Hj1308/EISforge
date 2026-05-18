"""
Literature Engine — موتور استنتاج از ادبیات علمی.

نویسنده: Hoda Jafari
تاریخ: May 2026

هدف:
----
قبل از اینکه کاربر هیچ داده‌ای اندازه‌گیری کند،
این ماژول بر اساس اطلاعات سیستم الکتروشیمیایی:

  ۱. بهترین حدس اولیه پارامترهای EIS را از ادبیات می‌دهد
  ۲. مدار معادل احتمالی را پیشنهاد می‌دهد
  ۳. بازه انتظار E_onset و I_f/I_b را می‌گوید
  ۴. هشدارهای مهم می‌دهد (مثل خطر مسمومیت CO)
  ۵. مراجع علمی را لیست می‌کند

این مثل یک همکار ارشد است که مقالات مرتبط را خوانده
و قبل از آزمایش به شما راهنمایی می‌کند.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# مسیر دیتابیس
_DB_PATH = Path(__file__).parent / "data" / "electrochemistry_knowledge.json"


@dataclass
class LiteratureGuess:
    """
    حدس اولیه بر اساس ادبیات علمی.

    Attributes
    ----------
    system_found : bool
        آیا سیستم در دیتابیس پیدا شد؟
    system_name : str
        نام سیستم یافت‌شده.
    recommended_circuit : str
        مدار معادل پیشنهادی از ادبیات.
    alternative_circuits : list[str]
        مدارهای جایگزین.
    initial_guess : dict[str, float]
        حدس اولیه پارامترها (میانگین بازه ادبیات).
    parameter_ranges : dict
        بازه‌های پارامترها از ادبیات.
    cv_expectations : dict
        انتظارات CV: E_onset، I_f/I_b.
    warnings : list[str]
        هشدارهای مهم.
    tips : list[str]
        نکات عملی اندازه‌گیری.
    references : list[str]
        مراجع علمی.
    confidence : str
        اطمینان به حدس: 'high'، 'medium'، 'low'.
    """

    system_found: bool
    system_name: str = ""
    recommended_circuit: str = "R0-p(R1,CPE1)"
    alternative_circuits: list = field(default_factory=list)
    initial_guess: dict = field(default_factory=dict)
    parameter_ranges: dict = field(default_factory=dict)
    cv_expectations: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    tips: list = field(default_factory=list)
    references: list = field(default_factory=list)
    confidence: str = "low"

    def report(self) -> str:
        """گزارش کامل حدس از ادبیات."""
        if not self.system_found:
            return (
                "⚠️  سیستم در دیتابیس ادبیات پیدا نشد.\n"
                "از initial guess پیش‌فرض استفاده کنید."
            )

        lines = [
            "═" * 62,
            f"  📚 راهنمای ادبیات — {self.system_name}",
            f"  اطمینان: {self._confidence_persian()}",
            "═" * 62,
            "",
            f"  🔌 مدار پیشنهادی: {self.recommended_circuit}",
        ]

        if self.alternative_circuits:
            lines.append(f"  🔌 مدارهای جایگزین: {', '.join(self.alternative_circuits)}")

        lines += ["", "  📊 حدس اولیه پارامترها (از ادبیات):"]
        for name, val in self.initial_guess.items():
            lo = self.parameter_ranges.get(name, {}).get("min", "?")
            hi = self.parameter_ranges.get(name, {}).get("max", "?")
            unit = self.parameter_ranges.get(name, {}).get("unit", "")
            lines.append(f"     {name:<15} = {val:.3e} {unit}  [{lo} — {hi}]")

        if self.cv_expectations:
            lines += ["", "  📈 انتظارات CV از ادبیات:"]
            for k, v in self.cv_expectations.items():
                lines.append(f"     {k}: {v}")

        if self.warnings:
            lines += ["", "  ⚠️  هشدارها:"]
            for w in self.warnings:
                lines.append(f"     • {w}")

        if self.tips:
            lines += ["", "  💡 نکات عملی:"]
            for t in self.tips:
                lines.append(f"     • {t}")

        if self.references:
            lines += ["", "  📖 مراجع:"]
            for r in self.references:
                lines.append(f"     [{r}]")

        lines.append("═" * 62)
        return "\n".join(lines)

    def _confidence_persian(self) -> str:
        return {"high": "✅ بالا", "medium": "⚠️ متوسط", "low": "❓ پایین"}.get(
            self.confidence, self.confidence
        )


class LiteratureEngine:
    """
    موتور جستجو و استنتاج از دیتابیس ادبیات الکتروشیمی.

    Parameters
    ----------
    db_path : str | Path | None
        مسیر فایل JSON دیتابیس.
        اگر None باشد، از مسیر پیش‌فرض استفاده می‌کند.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        path = Path(db_path) if db_path else _DB_PATH
        self._db = self._load_db(path)
        logger.info("دیتابیس ادبیات بارگذاری شد: %d سیستم", len(self._db.get("systems", {})))

    @staticmethod
    def _load_db(path: Path) -> dict:
        """بارگذاری دیتابیس از JSON."""
        if not path.exists():
            logger.warning("فایل دیتابیس پیدا نشد: %s", path)
            return {"systems": {}, "circuit_descriptions": {}}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def query(
        self,
        system_type: str,
        catalyst: str = "",
        electrolyte: str = "",
        alcohol: str = "",
        potential: Optional[float] = None,
    ) -> LiteratureGuess:
        """
        جستجوی دیتابیس و تولید حدس اولیه.

        Parameters
        ----------
        system_type : str
            نوع سیستم: 'AOR'، 'Battery'، 'Corrosion'، 'FuelCell'، 'Biosensor'.
        catalyst : str
            نام کاتالیست: 'Pt'، 'Pd'، 'PtRu'، 'PtSn'، ...
        electrolyte : str
            نوع الکترولیت: 'acidic'، 'alkaline'، 'NaCl'، 'PBS'.
        alcohol : str
            نوع الکل برای AOR: 'methanol'، 'ethanol'، ...
        potential : float | None
            پتانسیل اندازه‌گیری EIS (V) — برای تفسیر ناحیه CV.

        Returns
        -------
        LiteratureGuess
        """
        # ── جستجوی سیستم در دیتابیس ──────────────────────────────────────────
        system_key = self._find_system_key(system_type, catalyst)

        if system_key is None:
            return LiteratureGuess(
                system_found=False,
                warnings=[
                    f"سیستم '{system_type} / {catalyst}' در دیتابیس نیست. "
                    "می‌توانید آن را اضافه کنید."
                ],
            )

        system_data = self._db["systems"][system_key]
        electrolyte_key = self._classify_electrolyte(electrolyte)
        elec_data = system_data.get("electrolytes", {}).get(
            electrolyte_key,
            list(system_data.get("electrolytes", {}).values())[0]
            if system_data.get("electrolytes") else {}
        )

        # ── استخراج پارامترهای EIS ────────────────────────────────────────────
        eis_params = elec_data.get("EIS_parameters", {})
        initial_guess, parameter_ranges = self._compute_initial_guess(eis_params)

        # ── انتظارات CV ───────────────────────────────────────────────────────
        cv_params = elec_data.get("CV_parameters", {})
        cv_expectations = self._format_cv_expectations(cv_params, alcohol, potential)

        # ── مدار پیشنهادی ─────────────────────────────────────────────────────
        recommended_circuit = elec_data.get(
            "recommended_circuit",
            system_data.get("recommended_circuit", "R0-p(R1,CPE1)")
        )
        alternative_circuits = elec_data.get("alternative_circuits", [])

        # ── هشدارها و نکات ────────────────────────────────────────────────────
        warnings, tips = self._generate_warnings_tips(
            system_data, elec_data, electrolyte_key, potential, cv_params, alcohol
        )

        # ── اطمینان ───────────────────────────────────────────────────────────
        confidence = self._estimate_confidence(system_data, elec_data, alcohol)

        # ── مراجع ─────────────────────────────────────────────────────────────
        references = (
            elec_data.get("references", []) or
            system_data.get("references", [])
        )

        return LiteratureGuess(
            system_found=True,
            system_name=f"{system_data.get('catalyst', system_type)} / {electrolyte}",
            recommended_circuit=recommended_circuit,
            alternative_circuits=alternative_circuits,
            initial_guess=initial_guess,
            parameter_ranges=parameter_ranges,
            cv_expectations=cv_expectations,
            warnings=warnings,
            tips=tips,
            references=references,
            confidence=confidence,
        )

    def _find_system_key(self, system_type: str, catalyst: str) -> Optional[str]:
        """پیدا کردن کلید سیستم در دیتابیس."""
        systems = self._db.get("systems", {})

        # جستجوی دقیق
        for key, data in systems.items():
            if (
                data.get("system_type", "").upper() == system_type.upper() and
                catalyst.upper() in data.get("catalyst", "").upper()
            ):
                return key

        # جستجوی عمومی با نوع سیستم
        for key, data in systems.items():
            if data.get("system_type", "").upper() == system_type.upper():
                return key

        return None

    @staticmethod
    def _classify_electrolyte(electrolyte: str) -> str:
        """تشخیص نوع الکترولیت."""
        e = electrolyte.lower()
        if any(x in e for x in ["koh", "naoh", "alkaline", "base", "بازی"]):
            return "alkaline"
        if any(x in e for x in ["h2so4", "hclo4", "hno3", "acid", "اسیدی"]):
            return "acidic"
        if "nacl" in e or "cl" in e:
            return "NaCl"
        if "pbs" in e or "phosphate" in e:
            return "PBS"
        if "organic" in e or "liPF6" in e.lower():
            return "organic_carbonate"
        return "acidic"  # پیش‌فرض

    @staticmethod
    def _compute_initial_guess(
        eis_params: dict,
    ) -> tuple[dict[str, float], dict]:
        """
        محاسبه حدس اولیه از بازه‌های ادبیات.

        از میانگین هندسی (geometric mean) استفاده می‌کنیم
        چون پارامترها در log-space توزیع دارند.
        """
        import math
        initial_guess = {}
        parameter_ranges = {}

        param_map = {
            "R_solution": "R0",
            "R_ohmic":    "R0",
            "R_ct":       "R1",
            "CPE_Q":      "CPE1_Q",
            "CPE_n":      "CPE1_n",
            "CPE_dl":     "CPE1_Q",
            "Warburg_sigma": "W1_sigma",
            "R_oxide":    "R2",
            "C_oxide":    "C1",
            "R_SEI":      "R1",
            "C_SEI":      "C1",
            "R_film":     "R2",
            "C_film":     "C1",
        }

        for param_key, param_data in eis_params.items():
            lo  = param_data.get("min", 1e-6)
            hi  = param_data.get("max", 1.0)
            mid = math.exp((math.log(max(lo, 1e-15)) + math.log(max(hi, 1e-15))) / 2)

            guess_name = param_map.get(param_key, param_key)
            initial_guess[guess_name] = mid
            parameter_ranges[guess_name] = {
                "min":  lo,
                "max":  hi,
                "unit": param_data.get("unit", ""),
                "note": param_data.get("note", ""),
            }

        return initial_guess, parameter_ranges

    @staticmethod
    def _format_cv_expectations(
        cv_params: dict,
        alcohol: str,
        potential: Optional[float],
    ) -> dict:
        """فرمت‌بندی انتظارات CV."""
        expectations = {}
        alcohol_key = alcohol.lower().replace(" ", "_")

        for key, val in cv_params.items():
            if isinstance(val, dict):
                lo = val.get("min", "?")
                hi = val.get("max", "?")
                unit = val.get("unit", "")
                note = val.get("note", "")
                if alcohol_key in key or "methanol" not in key or alcohol_key == "methanol":
                    label = key.replace("_", " ").replace(alcohol_key, "").strip()
                    expectations[label] = f"{lo} — {hi} {unit}  {note}".strip()
            else:
                expectations[key] = str(val)

        return expectations

    @staticmethod
    def _generate_warnings_tips(
        system_data: dict,
        elec_data: dict,
        electrolyte_key: str,
        potential: Optional[float],
        cv_params: dict,
        alcohol: str,
    ) -> tuple[list, list]:
        """تولید هشدارها و نکات عملی."""
        warnings = []
        tips = []

        co_risk = (
            elec_data.get("CO_poisoning_risk") or
            system_data.get("CO_poisoning_risk", "")
        )
        if co_risk in ("HIGH", "VERY HIGH"):
            warnings.append(
                f"⚠️ خطر مسمومیت CO: {co_risk}. "
                "اگر دو قوس در Nyquist دیدید، از مدار دو قوسی استفاده کنید."
            )
            tips.append("EIS را در زمان‌های مختلف بگیرید تا drift مسمومیت را بررسی کنید.")

        if electrolyte_key == "acidic":
            tips.append("K-K validation را حتماً انجام دهید — داده اسیدی اغلب drift دارد.")

        if alcohol in ("ethanol", "ethylene_glycol", "glycerol"):
            tips.append(
                f"برای {alcohol}: فرکانس پایین را تا 10 mHz بگیرید "
                "— شکستن پیوند C-C نیاز به فرکانس پایین دارد."
            )

        if potential is not None:
            onset_key = f"E_onset_{alcohol.lower().replace(' ', '_')}"
            cv_onset = cv_params.get(onset_key, {})
            if cv_onset:
                lo = cv_onset.get("min", 0)
                hi = cv_onset.get("max", 1)
                if potential < lo:
                    warnings.append(
                        f"پتانسیل EIS ({potential:.3f} V) کمتر از E_onset "
                        f"({lo:.2f}—{hi:.2f} V) است. "
                        "R_ct خیلی بزرگ خواهد بود — واکنش شروع نشده."
                    )
                elif potential > hi:
                    tips.append(
                        f"پتانسیل EIS ({potential:.3f} V) بالاتر از E_onset است — "
                        "شرایط خوب برای مطالعه کینتیک واکنش."
                    )

        return warnings, tips

    @staticmethod
    def _estimate_confidence(
        system_data: dict,
        elec_data: dict,
        alcohol: str,
    ) -> str:
        """تخمین سطح اطمینان به حدس."""
        score = 0
        if elec_data.get("EIS_parameters"):
            score += 2
        if elec_data.get("CV_parameters"):
            score += 1
        if elec_data.get("references"):
            score += 1
        if elec_data.get("recommended_circuit"):
            score += 1

        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        return "low"

    def list_systems(self) -> list[dict]:
        """لیست همه سیستم‌های موجود در دیتابیس."""
        result = []
        for key, data in self._db.get("systems", {}).items():
            result.append({
                "key": key,
                "type": data.get("system_type", ""),
                "catalyst": data.get("catalyst", data.get("chemistry", data.get("material", ""))),
                "electrolytes": list(data.get("electrolytes", {}).keys()),
            })
        return result

    def get_circuit_description(self, circuit_string: str) -> Optional[dict]:
        """توضیح مدار معادل از دیتابیس."""
        return self._db.get("circuit_descriptions", {}).get(circuit_string)

    def add_system(self, key: str, system_data: dict, save: bool = False) -> None:
        """
        اضافه کردن سیستم جدید به دیتابیس.

        Parameters
        ----------
        key : str
            کلید یکتا (مثلاً 'AOR_PdAu').
        system_data : dict
            داده سیستم.
        save : bool
            اگر True، در فایل JSON ذخیره شود.
        """
        self._db["systems"][key] = system_data
        logger.info("سیستم جدید اضافه شد: %s", key)
        if save and _DB_PATH.exists():
            with open(_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(self._db, f, ensure_ascii=False, indent=2)
