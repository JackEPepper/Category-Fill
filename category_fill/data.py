"""Loading and cleaning tabular data (CSV / Excel / Parquet)."""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Values that mean "blank" in typical loan-system exports. Compared case-insensitively.
DEFAULT_MISSING_TOKENS = ("", "na", "n/a", "nan", "null", "none", "#n/a", "-", "--", "?")

_NUMERIC_JUNK = re.compile(r"[\$,%\s]")
_DATE_HINT = re.compile(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def load_table(source, filename: str | None = None) -> pd.DataFrame:
    """Load a CSV, TSV, Excel or Parquet file from a path or an uploaded file object."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))
    ext = Path(name).suffix.lower()
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(source)
    elif ext == ".parquet":
        df = pd.read_parquet(source)
    elif ext in (".tsv", ".tab"):
        df = pd.read_csv(source, sep="\t", dtype=str, keep_default_na=False)
    else:
        # Default: CSV. Read everything as text so cleaning decides types consistently.
        df = pd.read_csv(source, dtype=str, keep_default_na=False, encoding_errors="replace")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _try_numeric(s: pd.Series, min_share: float = 0.95) -> pd.Series | None:
    """Convert '$1,250,000', '4.5%', '(300)' style text to numbers if nearly all values parse."""
    nonnull = s.dropna()
    if nonnull.empty:
        return None
    # Leading zeros ('00123') usually mean a code, not a quantity: keep as text.
    if nonnull.str.match(r"^0\d").mean() > 0.05:
        return None
    cleaned = s.str.replace(_NUMERIC_JUNK, "", regex=True)
    negative = cleaned.str.match(r"^\(.*\)$").fillna(False)
    cleaned = cleaned.str.strip("()")
    parsed = pd.to_numeric(cleaned, errors="coerce")
    if parsed.notna().sum() < min_share * len(nonnull):
        return None
    parsed = parsed.astype("float64")
    parsed[negative.to_numpy(dtype=bool)] *= -1
    return parsed


def _try_datetime(s: pd.Series, min_share: float = 0.9) -> pd.Series | None:
    nonnull = s.dropna()
    if nonnull.empty or nonnull.head(200).str.contains(_DATE_HINT).mean() < 0.8:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    if parsed.notna().sum() < min_share * len(nonnull):
        return None
    return parsed


def clean_frame(
    df: pd.DataFrame,
    extra_missing_tokens: tuple[str, ...] | list[str] = (),
    force_categorical: tuple[str, ...] | list[str] = (),
    expand_dates: bool = True,
) -> pd.DataFrame:
    """Standardise blanks, detect numeric/date columns, and expand dates into model-friendly parts.

    Returns a new frame where every column is either float64, or pandas 'string' (categorical text).
    Date columns become ``<col>_year``, ``<col>_month`` and ``<col>_days`` (days since 1970).
    """
    tokens = {t.strip().lower() for t in (*DEFAULT_MISSING_TOKENS, *extra_missing_tokens)}
    force_categorical = set(force_categorical)
    out = {}

    for col in df.columns:
        name = str(col).strip()
        s = df[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and name not in force_categorical:
            out[name] = s.astype("float64")
            continue
        else:
            text = _to_text(s)
            text = text.mask(text.str.lower().isin(tokens))
            if name in force_categorical:
                out[name] = text
                continue
            num = _try_numeric(text)
            if num is not None:
                out[name] = num
                continue
            dt = _try_datetime(text) if expand_dates else None
            if dt is None:
                out[name] = text
                continue

        if expand_dates:
            out[f"{name}_year"] = dt.dt.year.astype("float64")
            out[f"{name}_month"] = dt.dt.month.astype("float64")
            days = (dt - pd.Timestamp("1970-01-01")).dt.days
            out[f"{name}_days"] = days.astype("float64")
        else:
            out[name] = _to_text(dt.astype("string"))

    cleaned = pd.DataFrame(out, index=df.index)
    # Drop columns that are completely empty.
    return cleaned.loc[:, cleaned.notna().any()]


def is_categorical(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s)


def missing_mask(s: pd.Series) -> np.ndarray:
    return s.isna().to_numpy()
