"""ابزارهای کمکی برای خواندن فایل با encoding صحیح."""
import pandas as pd

ENCODINGS = ["latin-1", "cp1252", "utf-8", "utf-16"]

def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """خواندن CSV با امتحان encoding های مختلف."""
    for enc in ENCODINGS:
        try:
            return pd.read_csv(
                path, encoding=enc,
                sep=None, engine="python",
                comment="#", **kwargs
            )
        except (UnicodeDecodeError, UnicodeError):
            continue
    # آخرین تلاش
    return pd.read_csv(
        path, encoding="latin-1",
        errors="replace", sep=None,
        engine="python", comment="#", **kwargs
    )