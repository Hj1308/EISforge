"""
Experimental Metadata Manager — مدیریت پارامترهای آزمایشگاهی.

نویسنده: Hoda Jafari | May 2026

پارامترهایی که ثبت می‌شوند:
    - سرعت scan (mV/s)
    - شدت جریان و واحد (A، mA، μA، nA)
    - غلظت الکل (M یا mM)
    - غلظت الکترولیت (M)
    - دمای واکنش (°C)
    - مساحت الکترود (cm²)
    - ECSA و loading
    - پتانسیل مرجع

این اطلاعات:
    1. در تحلیل استفاده می‌شوند (تبدیل واحد، نرمال‌سازی)
    2. در گزارش نهایی چاپ می‌شوند
    3. برای مقایسه بین آزمایشگاه‌ها ضروری هستند
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)


# ── تعریف واحدهای جریان ────────────────────────────────────────────────────
CURRENT_UNITS = {
    "A":   1.0,          # آمپر — ضریب تبدیل به mA
    "mA":  1.0,          # میلی‌آمپر (پیش‌فرض EISForge)
    "μA":  1e-3,         # میکروآمپر → ÷1000 برای mA
    "nA":  1e-6,         # نانوآمپر → ÷1,000,000 برای mA
    "pA":  1e-9,         # پیکوآمپر
}

# ── تعریف واحدهای غلظت ────────────────────────────────────────────────────
CONCENTRATION_UNITS = {
    "M":   1.0,          # مولار
    "mM":  1e-3,         # میلی‌مولار
    "μM":  1e-6,         # میکرومولار
    "ppm": None,         # نیاز به وزن مولکولی دارد
}

# ── الکترودهای مرجع رایج ──────────────────────────────────────────────────
REFERENCE_ELECTRODES = {
    "RHE":              0.000,   # پیش‌فرض — نیاز به تبدیل نیست
    "NHE/SHE":          0.000,   # در pH=0 همان RHE است
    "Ag/AgCl (sat.)":   0.197,   # در KCl اشباع
    "Ag/AgCl (3M KCl)": 0.210,   # در KCl 3M
    "SCE":              0.241,   # کالومل اشباع
    "Hg/HgO (1M KOH)":  0.098,   # برای محیط بازی
    "Hg/HgSO4":         0.640,   # برای محیط اسیدی
    "MSE":              0.640,   # Mercury Sulfate Electrode
}


@dataclass
class ExperimentalConditions:
    """
    شرایط کامل آزمایشگاهی برای یک اندازه‌گیری الکتروشیمیایی.

    تمام این پارامترها در گزارش نهایی ظاهر می‌شوند و
    برای مقایسه صحیح با ادبیات ضروری هستند.
    """

    # ── اطلاعات کلی ────────────────────────────────────────────────────────
    experiment_id:   str = ""
    date:            str = ""
    operator:        str = ""
    lab:             str = ""
    notes:           str = ""

    # ── کاتالیست ───────────────────────────────────────────────────────────
    catalyst:        str   = ""        # مثلاً "PtRu/C (1:1)"
    catalyst_source: str   = ""        # مثلاً "Sigma-Aldrich 40wt%"
    electrode_type:  str   = ""        # GCE، RDE، carbon paper، ...

    # ── مساحت الکترود ──────────────────────────────────────────────────────
    geometric_area:  float = 1.0       # cm² — مساحت هندسی
    ecsa:            float = 0.0       # cm²_metal — از H_upd یا CO stripping
    ecsa_method:     str   = ""        # "H_upd"، "CO stripping"، "BET"
    catalyst_loading:float = 0.0       # mg/cm²

    # ── سرعت scan ──────────────────────────────────────────────────────────
    scan_rate_cv:    float = 50.0      # mV/s برای CV
    scan_rate_lsv:   float = 5.0       # mV/s برای LSV
    scan_rate_eis:   float = 0.0       # N/A برای EIS

    # ── شدت جریان و واحد ──────────────────────────────────────────────────
    current_unit:    str   = "mA"      # A، mA، μA، nA
    current_range:   str   = ""        # مثلاً "±200 mA" — از پتانسیواستات

    # ── پتانسیل مرجع ───────────────────────────────────────────────────────
    reference_electrode:   str   = "RHE"   # نوع الکترود مرجع
    e_ref_to_rhe:          float = 0.0     # V — ضریب تبدیل به RHE
    # محاسبه خودکار e_ref_to_rhe از نوع الکترود مرجع

    # ── الکترولیت ──────────────────────────────────────────────────────────
    electrolyte_type:   str   = ""         # "H2SO4"، "KOH"، "HClO4"، ...
    electrolyte_conc:   float = 0.5        # مولار
    electrolyte_unit:   str   = "M"
    electrolyte_ph:     float = -1.0       # اگر منفی باشد = نامشخص

    # ── سوبسترا (الکل) ────────────────────────────────────────────────────
    substrate:          str   = ""         # "ethanol"، "methanol"، ...
    substrate_conc:     float = 0.0        # غلظت
    substrate_unit:     str   = "M"        # M یا mM
    substrate_conc_m:   float = 0.0        # غلظت در مولار (محاسبه خودکار)

    # ── دمای واکنش ─────────────────────────────────────────────────────────
    temperature_c:      float = 25.0       # درجه سانتیگراد
    temperature_k:      float = 298.15     # کلوین (محاسبه خودکار)

    # ── پارامترهای EIS ─────────────────────────────────────────────────────
    eis_potential:      float = 0.0        # V vs مرجع
    eis_amplitude_mv:   float = 10.0       # mV — دامنه AC
    eis_freq_high:      float = 1e5        # Hz
    eis_freq_low:       float = 0.01       # Hz
    eis_points_per_dec: int   = 10

    def __post_init__(self):
        """محاسبات خودکار بعد از مقداردهی."""
        # تبدیل دما
        self.temperature_k = self.temperature_c + 273.15

        # تبدیل غلظت سوبسترا به مولار
        factor = CONCENTRATION_UNITS.get(self.substrate_unit, 1.0)
        if factor:
            self.substrate_conc_m = self.substrate_conc * factor

        # تبدیل الکترود مرجع به RHE
        if self.e_ref_to_rhe == 0.0 and self.reference_electrode in REFERENCE_ELECTRODES:
            self.e_ref_to_rhe = REFERENCE_ELECTRODES[self.reference_electrode]

        # pH خودکار
        if self.electrolyte_ph < 0:
            if "koh" in self.electrolyte_type.lower() or "naoh" in self.electrolyte_type.lower():
                self.electrolyte_ph = 14 + np.log10(self.electrolyte_conc)
            elif any(x in self.electrolyte_type.lower() for x in ["h2so4","hclo4","hno3"]):
                n_h = 2 if "h2so4" in self.electrolyte_type.lower() else 1
                self.electrolyte_ph = -np.log10(n_h * self.electrolyte_conc)

    def convert_current_to_ma(self, current: float) -> float:
        """تبدیل جریان از واحد ورودی به mA."""
        factor = CURRENT_UNITS.get(self.current_unit, 1.0)
        return current * factor

    def convert_current_array_to_ma(self, current_array) -> "np.ndarray":
        """تبدیل آرایه جریان به mA."""
        factor = CURRENT_UNITS.get(self.current_unit, 1.0)
        return np.asarray(current_array) * factor

    def thermal_voltage(self) -> float:
        """ولتاژ حرارتی V_T = kT/e = RT/F در دمای آزمایش."""
        from eisforge.utils.constants import GAS_CONSTANT, FARADAY_CONSTANT
        return GAS_CONSTANT * self.temperature_k / FARADAY_CONSTANT

    def tafel_slope_theoretical(self, alpha: float = 0.5) -> float:
        """
        Tafel slope نظری در دمای آزمایش.

        b = 2.303·RT / (α·n·F)   [V/dec]

        Parameters
        ----------
        alpha : float
            ضریب انتقال (transfer coefficient)، معمولاً 0.5.
        """
        vt = self.thermal_voltage()
        return 2.303 * vt / alpha * 1000  # mV/dec

    def summary(self) -> str:
        """خلاصه شرایط آزمایشگاهی."""
        lines = [
            "═" * 60,
            "  🧪 شرایط آزمایشگاهی — EISForge",
            "═" * 60,
            f"  کاتالیست     : {self.catalyst or 'نامشخص'}",
            f"  الکترولیت    : {self.electrolyte_type} {self.electrolyte_conc} {self.electrolyte_unit}  (pH≈{self.electrolyte_ph:.1f})",
            f"  سوبسترا       : {self.substrate} {self.substrate_conc} {self.substrate_unit}  ({self.substrate_conc_m:.4f} M)",
            f"  دما           : {self.temperature_c} °C  ({self.temperature_k:.1f} K)",
            "─" * 60,
            f"  سرعت scan CV : {self.scan_rate_cv} mV/s",
            f"  سرعت scan LSV: {self.scan_rate_lsv} mV/s",
            f"  واحد جریان   : {self.current_unit}",
            f"  الکترود مرجع : {self.reference_electrode}  (E_RHE offset = {self.e_ref_to_rhe:.3f} V)",
            "─" * 60,
            f"  مساحت هندسی  : {self.geometric_area} cm²",
            f"  ECSA          : {self.ecsa:.4f} cm²  ({self.ecsa_method})",
            f"  Loading        : {self.catalyst_loading} mg/cm²",
            "─" * 60,
            f"  Tafel نظری   : {self.tafel_slope_theoretical():.1f} mV/dec  (α=0.5، {self.temperature_c}°C)",
            "═" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        """ذخیره شرایط آزمایش در JSON."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("شرایط آزمایش ذخیره شد: %s", path)

    @classmethod
    def load(cls, path: str) -> "ExperimentalConditions":
        """بارگذاری از JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_report_header(self) -> str:
        """
        هدر گزارش برای مقاله — فرمت استاندارد ادبیات.

        مثال خروجی:
            Measurements were performed at 25°C in 0.5M H2SO4
            containing 1.0M ethanol, using Pt/C as catalyst
            (loading: 0.2 mg/cm²) at a scan rate of 50 mV/s.
        """
        return (
            f"Measurements were performed at {self.temperature_c}°C "
            f"in {self.electrolyte_conc}{self.electrolyte_unit} {self.electrolyte_type} "
            f"{'containing ' + str(self.substrate_conc) + self.substrate_unit + ' ' + self.substrate if self.substrate else ''}, "
            f"using {self.catalyst or 'catalyst'} "
            f"{'(loading: ' + str(self.catalyst_loading) + ' mg/cm²)' if self.catalyst_loading > 0 else ''} "
            f"at a scan rate of {self.scan_rate_cv} mV/s "
            f"(CV) and {self.scan_rate_lsv} mV/s (LSV). "
            f"All potentials are referenced to {self.reference_electrode}."
        )
