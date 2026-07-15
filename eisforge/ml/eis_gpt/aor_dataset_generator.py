"""
تولیدکننده دیتاست مصنوعی EIS مخصوص AOR روی کاتالیست‌ها.

نویسنده: Hoda Jafari
تاریخ: May 2026

زمینه علمی:
-----------
AOR (Alcohol Oxidation Reaction) روی کاتالیست‌های فلزی
(Pt، Pd، PtRu، PtSn و ...) یکی از پیچیده‌ترین سیستم‌های
الکتروشیمیایی است. طیف EIS در این سیستم شامل:

۱. R_s  : مقاومت محلول (الکترولیت + تماس)
۲. R_ct : مقاومت انتقال بار روی سطح کاتالیست
۳. CPE  : دو لایه الکتریکی غیرایده‌آل (ناهمواری سطح)
۴. Z_W  : Warburg — انتشار الکل به سطح کاتالیست
۵. R_ads: مقاومت جذب واسطه‌های مسموم‌کننده (CO_ads روی Pt)
۶. C_ads: ظرفیت مربوط به پوشش واسطه‌های جذب‌شده

مکانیزم‌های مختلف در AOR طیف‌های متفاوتی تولید می‌کنند:
- کاتالیست تازه/فعال      → یک قوس ساده
- مسمومیت CO             → دو قوس (قوس دوم در فرکانس پایین)
- انتشار محدودکننده       → دنباله Warburg در فرکانس پایین
- فیلم اکسید روی سطح     → قوس اضافی در فرکانس بالا
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit

logger = logging.getLogger(__name__)

# ── فرکانس‌های استاندارد برای AOR ────────────────────────────────────────────
# از 10 mHz تا 100 kHz — محدوده‌ای که همه پدیده‌های AOR را می‌پوشاند
AOR_FREQUENCIES = np.logspace(-2, 5, 60)  # 10 mHz → 100 kHz

# ── کتابخانه مدارهای معادل خاص AOR ──────────────────────────────────────────
AOR_CIRCUIT_LIBRARY = [
    {
        # ── مدار ۰: کاتالیست تازه و فعال ────────────────────────────────────
        # وقتی کاتالیست تمیز است و هیچ مسمومیتی ندارد
        # فقط یک قوس نیمه‌دایره در Nyquist
        "string": "R0-p(R1,CPE1)",
        "label": 0,
        "name": "AOR - Fresh Catalyst",
        "description": "کاتالیست تازه: فقط انتقال بار + دو لایه",
        "electrochemistry": "مکانیزم ساده Langmuir-Hinshelwood، بدون مسمومیت",
        "param_names": ["R0", "R1", "CPE1_Q", "CPE1_n"],
        "param_ranges": {
            # R_s: مقاومت محلول الکترولیت (H2SO4 یا KOH)
            "R0":      (0.5,   20.0),    # Ω — بسته به غلظت الکترولیت
            # R_ct: انتقال بار — برای کاتالیست خوب کوچک است
            "R1":      (5.0,   500.0),   # Ω — Pt فعال: ~10Ω، Pd: ~50Ω
            # Q: پیش‌ضریب CPE — مربوط به ظرفیت دو لایه و سطح واقعی
            "CPE1_Q":  (1e-5,  1e-2),    # F·s^(n-1) — سطح بالا → Q بزرگ
            # n: توان CPE — 1=خازن ایده‌آل، کمتر=ناهمواری سطح
            "CPE1_n":  (0.75,  0.98),    # برای کاتالیست‌های Pt/C: ~0.85-0.95
        },
    },
    {
        # ── مدار ۱: انتشار محدودکننده ────────────────────────────────────────
        # وقتی غلظت الکل پایین است یا جریان بالاست
        # دنباله Warburg در فرکانس پایین نیکویست
        "string": "R0-p(R1,CPE1)-W1",
        "label": 1,
        "name": "AOR - Diffusion Limited",
        "description": "انتشار الکل محدودکننده: Warburg در فرکانس پایین",
        "electrochemistry": "انتشار الکل به سطح کاتالیست سرعت واکنش را محدود می‌کند",
        "param_names": ["R0", "R1", "CPE1_Q", "CPE1_n", "W1_sigma"],
        "param_ranges": {
            "R0":       (0.5,   20.0),
            "R1":       (10.0,  800.0),
            "CPE1_Q":   (1e-5,  1e-2),
            "CPE1_n":   (0.75,  0.98),
            # σ: ضریب Warburg — بزرگ‌تر = انتشار کندتر
            "W1_sigma": (5.0,   500.0),  # Ω·s^(-0.5)
        },
    },
    {
        # ── مدار ۲: مسمومیت CO ───────────────────────────────────────────────
        # مهم‌ترین مشکل AOR روی Pt خالص!
        # CO_ads روی سطح Pt جذب می‌شود و سایت‌های فعال را می‌بندد
        # دو قوس در Nyquist:
        #   قوس اول (فرکانس بالا): انتقال بار اصلی الکل
        #   قوس دوم (فرکانس پایین): دینامیک جذب/دفع CO
        "string": "R0-p(R1,CPE1)-p(R2,CPE2)",
        "label": 2,
        "name": "AOR - CO Poisoning",
        "description": "مسمومیت CO: دو قوس — انتقال بار + دینامیک CO_ads",
        "electrochemistry": (
            "قوس اول: اکسیداسیون الکل (R_ct1 + CPE_dl)\n"
            "قوس دوم: تجمع و اکسیداسیون CO_ads (R_CO + CPE_CO)\n"
            "R_CO بزرگ‌تر = مسمومیت شدیدتر"
        ),
        "param_names": [
            "R0", "R1", "CPE1_Q", "CPE1_n",
            "R2", "CPE2_Q", "CPE2_n"
        ],
        "param_ranges": {
            "R0":      (0.5,   20.0),
            "R1":      (10.0,  500.0),   # R_ct الکل
            "CPE1_Q":  (1e-5,  1e-2),
            "CPE1_n":  (0.75,  0.98),
            # R2 معمولاً بزرگ‌تر از R1 است — CO جذب محکم
            "R2":      (50.0,  5000.0),  # R_CO — مسمومیت شدید → بزرگ
            "CPE2_Q":  (1e-6,  1e-3),    # ظرفیت کوچک‌تر (سایت‌های CO)
            "CPE2_n":  (0.65,  0.95),
        },
    },
    {
        # ── مدار ۳: مکانیزم کامل AOR با انتشار محدود ────────────────────────
        # پیچیده‌ترین حالت: مسمومیت + انتشار همزمان
        # مثلاً: Pt/C در غلظت اتانول پایین با زمان طولانی
        "string": "R0-p(R1,CPE1)-p(R2,CPE2)-Wo1",
        "label": 3,
        "name": "AOR - Full Mechanism",
        "description": "مکانیزم کامل: انتقال بار + CO + انتشار محدود",
        "electrochemistry": "هر سه پدیده همزمان فعال هستند",
        "param_names": [
            "R0", "R1", "CPE1_Q", "CPE1_n",
            "R2", "CPE2_Q", "CPE2_n",
            "Wo1_R", "Wo1_T"
        ],
        "param_ranges": {
            "R0":      (0.5,   20.0),
            "R1":      (10.0,  500.0),
            "CPE1_Q":  (1e-5,  1e-2),
            "CPE1_n":  (0.75,  0.98),
            "R2":      (50.0,  3000.0),
            "CPE2_Q":  (1e-6,  1e-3),
            "CPE2_n":  (0.65,  0.95),
            # Warburg باز: انتشار در لایه نازک الکترولیت
            "Wo1_R":   (10.0,  300.0),   # مقاومت انتشار
            "Wo1_T":   (0.01,  10.0),    # ثابت زمانی انتشار
        },
    },
    {
        # ── مدار ۴: فیلم اکسید روی سطح کاتالیست ────────────────────────────
        # مخصوص کاتالیست‌های Pd در محیط قلیایی
        # لایه Pd-OH/PdO روی سطح تشکیل می‌شود
        # قوس اضافی در فرکانس بالا (قبل از R_ct)
        "string": "R0-p(R1,C1)-p(R2,CPE1)",
        "label": 4,
        "name": "AOR - Oxide Film (Pd/alkaline)",
        "description": "فیلم اکسید Pd-OH: مخصوص کاتالیست Pd در KOH",
        "electrochemistry": (
            "قوس اول (فرکانس بالا): فیلم اکسید Pd-OH (R_ox + C_ox)\n"
            "قوس دوم (فرکانس میانی): انتقال بار اکسیداسیون الکل"
        ),
        "param_names": [
            "R0", "R1", "C1",
            "R2", "CPE1_Q", "CPE1_n"
        ],
        "param_ranges": {
            "R0":      (0.5,   15.0),
            # R_ox: مقاومت فیلم اکسید — معمولاً کوچک
            "R1":      (1.0,   50.0),    # Ω — لایه اکسید نازک
            # C_ox: ظرفیت فیلم اکسید — بزرگ‌تر از CPE_dl
            "C1":      (1e-4,  1e-1),    # F — فیلم اکسید ضخیم → بزرگ
            "R2":      (20.0,  2000.0),  # R_ct اصلی
            "CPE1_Q":  (1e-5,  5e-3),
            "CPE1_n":  (0.80,  0.98),
        },
    },
    {
        # ── مدار ۵: حلقه شبه‌القایی — امضای اصلی AOR ────────────────────────
        # مهم‌ترین اثرانگشت سینتیکی AOR در نیکویست:
        # حلقه القایی فرکانس‌پایین در ربع چهارم (−Z'' منفی)
        # ناشی از آسایش کند پوشش واسطه جذب‌شده (CO_ads, استالدهید_ads,
        # ایزوپروپوکساید→استون) در رقابت با تفکیک آب.
        # توپولوژی: شاخه فارادیک R1-(R2||L2) داخل CPE دولایه.
        "string": "R0-p(CPE1,R1-p(R2,L2))",
        "label": 5,
        "name": "AOR - Pseudo-inductive Loop",
        "description": "حلقه القایی فرکانس‌پایین: آسایش پوشش واسطه جذب‌شده",
        "electrochemistry": (
            "R1: انتقال بار لحظه‌ای اکسیداسیون الکل\n"
            "R2||L2: دینامیک آسایش پوشش θ واسطه (CO_ads و مشابه)\n"
            "قوس فرکانس‌پایین به ربع چهارم می‌پیچد — fingerprint قطعی AOR"
        ),
        "param_names": ["R0", "CPE1_Q", "CPE1_n", "R1", "R2", "L2"],
        "param_ranges": {
            "R0":      (0.5,   20.0),
            "CPE1_Q":  (1e-5,  1e-2),
            "CPE1_n":  (0.75,  0.98),
            "R1":      (10.0,  1000.0),   # R_ct
            "R2":      (10.0,  2000.0),   # مقاومت آسایش پوشش
            # L: شبه‌القا — در AOR مقادیر بزرگ (تا هزاران هانری) گزارش شده
            "L2":      (0.5,   5000.0),   # H
        },
        "allow_negative_real": False,
    },
    {
        # ── مدار ۶: NDR/HNDR — مقاومت تفاضلی منفی ──────────────────────────
        # پس از پیک جریان، R آسایش پوشش منفی می‌شود
        # (Hidden Negative Differential Resistance — پیش‌درآمد رژیم نوسانی
        # هاپف که برای ۲-پروپانول و متانول مستند است).
        # قوس فرکانس‌پایین وارد ربع دوم می‌شود: Re(Z) < 0 مجاز و فیزیکی!
        "string": "R0-p(CPE1,R1-p(R2,L2))",
        "label": 6,
        "name": "AOR - Negative Differential Resistance (NDR)",
        "description": "NDR/HNDR پس از پیک: Re(Z) فرکانس‌پایین منفی — سیستم passive نیست",
        "electrochemistry": (
            "R2 < 0: بازخورد مثبت پوشش-پتانسیل (مسمومیت/آزادسازی سایت)\n"
            "قوس فرکانس‌پایین در ربع دوم — نزدیک ناپایداری هاپف\n"
            "فیت فقط با allow_negative_r=True؛ اعتبارسنجی passivity باید غیرفعال شود"
        ),
        "param_names": ["R0", "CPE1_Q", "CPE1_n", "R1", "R2", "L2"],
        "param_ranges": {
            "R0":      (0.5,   20.0),
            "CPE1_Q":  (1e-5,  1e-2),
            "CPE1_n":  (0.75,  0.98),
            "R1":      (10.0,  1000.0),
            # R2 منفی: بازه به‌صورت (lo, hi) با هر دو منفی — نمونه‌گیری
            # log-uniform روی قدر مطلق انجام و سپس علامت منفی اعمال می‌شود.
            "R2":      (-2000.0, -15.0),
            "L2":      (0.5,   5000.0),
        },
        "negative_params": ["R2"],
        "allow_negative_real": True,
    },
]


@dataclass
class AORSyntheticRecord:
    """یک نمونه مصنوعی AOR با metadata کامل."""

    circuit_label: int
    circuit_name: str
    circuit_string: str
    parameters: dict[str, float]
    frequency: np.ndarray
    z_real: np.ndarray
    z_imag: np.ndarray
    noise_level: float
    # metadata الکتروشیمیایی اضافی
    catalyst_type: str = ""
    electrolyte: str = ""
    alcohol: str = ""
    potential_v: float = 0.0


class AORDatasetGenerator:
    """
    تولیدکننده دیتاست مصنوعی EIS مخصوص AOR.

    Parameters
    ----------
    n_samples_per_circuit : int
        تعداد نمونه برای هر مدار (پیش‌فرض: 2000).
    noise_range : tuple
        بازه نویز واقع‌بینانه برای پتانسیواستات‌های مختلف.
        (0.005, 0.03) = 0.5% تا 3% از |Z|
    seed : int
        seed تصادفی برای بازتولیدپذیری نتایج.
    """

    # کاتالیست‌های رایج در AOR
    CATALYSTS = ["Pt/C", "Pd/C", "PtRu/C", "PtSn/C", "PdAu/C", "Pt-black"]
    # الکترولیت‌های رایج
    ELECTROLYTES = ["0.5M H2SO4", "1M KOH", "0.1M HClO4", "1M NaOH"]
    # الکل‌های رایج
    ALCOHOLS = ["methanol", "ethanol", "ethylene glycol", "glycerol"]

    def __init__(
        self,
        n_samples_per_circuit: int = 2000,
        noise_range: tuple[float, float] = (0.005, 0.03),
        seed: int = 42,
    ) -> None:
        self.n_samples_per_circuit = n_samples_per_circuit
        self.noise_range = noise_range
        self.rng = np.random.default_rng(seed)

    def generate(self, verbose: bool = True) -> list[AORSyntheticRecord]:
        """
        تولید دیتاست کامل برای همه مدارهای AOR.

        Returns
        -------
        list[AORSyntheticRecord]
        """
        records: list[AORSyntheticRecord] = []

        for circuit_def in AOR_CIRCUIT_LIBRARY:
            if verbose:
                logger.info(
                    "تولید %d نمونه: %s ...",
                    self.n_samples_per_circuit,
                    circuit_def["name"],
                )

            count, attempts = 0, 0
            while count < self.n_samples_per_circuit and attempts < self.n_samples_per_circuit * 10:
                attempts += 1
                record = self._generate_single(circuit_def)
                if record is not None:
                    records.append(record)
                    count += 1

            if verbose:
                logger.info("  ✅ %d نمونه تولید شد.", count)

        logger.info("📊 دیتاست کامل: %d نمونه از %d مدار",
                    len(records), len(AOR_CIRCUIT_LIBRARY))
        return records

    def _generate_single(
        self, circuit_def: dict
    ) -> Optional[AORSyntheticRecord]:
        """تولید یک نمونه مصنوعی با پارامترهای تصادفی."""

        # نمونه‌گیری log-uniform از پارامترها
        # پارامترهای فهرست‌شده در negative_params با قدر مطلق log-uniform
        # نمونه‌گیری و سپس منفی می‌شوند (NDR).
        negative_params = set(circuit_def.get("negative_params", []))
        initial_guess = []
        param_values = {}

        for name, (lo, hi) in circuit_def["param_ranges"].items():
            if name in negative_params:
                mag_lo, mag_hi = sorted([abs(lo), abs(hi)])
                val = -float(np.exp(self.rng.uniform(np.log(mag_lo), np.log(mag_hi))))
            else:
                val = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            param_values[name] = val
            initial_guess.append(val)

        # محاسبه طیف نظری
        try:
            circuit = CustomCircuit(
                circuit=circuit_def["string"],
                initial_guess=initial_guess,
            )
            Z_theory = circuit.predict(AOR_FREQUENCIES)
        except Exception:
            return None

        if not np.all(np.isfinite(Z_theory)):
            return None

        # فیلتر passivity — فقط برای مدارهای passive:
        # در رژیم NDR (allow_negative_real=True) Re(Z) فرکانس‌پایین منفی
        # کاملاً فیزیکی است؛ ولی Re(Z) در بالاترین فرکانس باید ≥ 0 بماند
        # (مقاومت محلول). AOR_FREQUENCIES صعودی است → آخرین نقطه = HF.
        if circuit_def.get("allow_negative_real", False):
            if Z_theory.real[-1] < 0:
                return None
        else:
            if np.any(Z_theory.real < 0):
                return None

        # اضافه کردن نویز واقع‌بینانه
        noise_level = float(self.rng.uniform(*self.noise_range))
        z_mod = np.abs(Z_theory)
        Z_noisy = Z_theory + self.rng.normal(0, noise_level * z_mod) \
                           + 1j * self.rng.normal(0, noise_level * z_mod)

        # metadata تصادفی
        catalyst   = str(self.rng.choice(self.CATALYSTS))
        electrolyte = str(self.rng.choice(self.ELECTROLYTES))
        alcohol    = str(self.rng.choice(self.ALCOHOLS))
        potential  = float(self.rng.uniform(0.3, 0.9))  # V vs RHE

        return AORSyntheticRecord(
            circuit_label=circuit_def["label"],
            circuit_name=circuit_def["name"],
            circuit_string=circuit_def["string"],
            parameters=param_values,
            frequency=AOR_FREQUENCIES.copy(),
            z_real=Z_noisy.real,
            z_imag=-Z_noisy.imag,
            noise_level=noise_level,
            catalyst_type=catalyst,
            electrolyte=electrolyte,
            alcohol=alcohol,
            potential_v=potential,
        )

    def to_dataframe(self, records: list[AORSyntheticRecord]) -> pd.DataFrame:
        """تبدیل به DataFrame برای آموزش ML."""
        rows = []
        for rec in records:
            row: dict = {
                "circuit_label":  rec.circuit_label,
                "circuit_name":   rec.circuit_name,
                "circuit_string": rec.circuit_string,
                "noise_level":    rec.noise_level,
                "catalyst":       rec.catalyst_type,
                "electrolyte":    rec.electrolyte,
                "alcohol":        rec.alcohol,
                "potential_v":    rec.potential_v,
            }
            for k, v in rec.parameters.items():
                row[f"param_{k}"] = v

            for i in range(len(rec.frequency)):
                z_mod = np.sqrt(rec.z_real[i]**2 + rec.z_imag[i]**2)
                z_phz = np.degrees(np.arctan2(rec.z_imag[i], rec.z_real[i]))
                row[f"Zre_{i}"]  = rec.z_real[i]
                row[f"Zim_{i}"]  = rec.z_imag[i]
                row[f"Zmod_{i}"] = z_mod
                row[f"Zphz_{i}"] = z_phz

            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, records: list[AORSyntheticRecord], path: str) -> None:
        """ذخیره به فرمت parquet."""
        df = self.to_dataframe(records)
        df.to_parquet(path, index=False)
        logger.info("✅ دیتاست ذخیره شد: %s (%d نمونه)", path, len(records))

    def summary(self, records: list[AORSyntheticRecord]) -> str:
        """خلاصه آماری دیتاست."""
        from collections import Counter
        counts = Counter(r.circuit_name for r in records)
        lines = ["📊 خلاصه دیتاست AOR:", "─" * 50]
        for name, count in counts.items():
            lines.append(f"  {name:<35} {count:>5} نمونه")
        lines.append(f"{'─'*50}")
        lines.append(f"  {'مجموع':<35} {len(records):>5} نمونه")
        return "\n".join(lines)


# ── اسکریپت سریع تولید دیتاست ────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🔬 تولید دیتاست AOR برای EISForge ...")
    gen = AORDatasetGenerator(n_samples_per_circuit=2000, seed=42)
    records = gen.generate()
    print(gen.summary(records))
    gen.save(records, "models/aor_dataset.parquet")
    print("✅ تمام شد!")
