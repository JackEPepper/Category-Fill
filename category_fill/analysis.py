"""Exploratory analysis: which column is the target, and which columns help predict it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import is_categorical


def target_labels(s: pd.Series) -> pd.Series:
    """Render a target column as text labels (numeric codes like 3.0 become '3')."""
    if pd.api.types.is_numeric_dtype(s):
        vals = s.astype("float64")
        is_int = vals.dropna().mod(1).eq(0).all()
        text = vals.map(lambda v: f"{int(v)}" if is_int else f"{v:g}", na_action="ignore")
        return text.astype("string")
    return s.astype("string")


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, missingness, cardinality and most common values."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        top = nonnull.value_counts().head(3)
        rows.append(
            {
                "column": col,
                "type": "categorical" if is_categorical(s) else "numeric",
                "missing": int(s.isna().sum()),
                "missing_pct": round(100 * s.isna().mean(), 1) if n else 0.0,
                "unique": int(nonnull.nunique()),
                "top_values": ", ".join(f"{k} ({v})" for k, v in top.items()),
            }
        )
    return pd.DataFrame(rows)


def suggest_targets(df: pd.DataFrame, max_classes: int = 200) -> list[str]:
    """Rank columns that look like 'a category with a chunk missing' (closest to ~15% missing first)."""
    scored = []
    for col in df.columns:
        s = df[col]
        miss = s.isna().mean()
        nunique = s.nunique(dropna=True)
        if not (0.005 <= miss <= 0.6 and 2 <= nunique <= max_classes):
            continue
        if not is_categorical(s) and nunique > 20:
            continue
        scored.append((abs(miss - 0.15) + (0 if is_categorical(s) else 0.5), col))
    return [c for _, c in sorted(scored)]


def _as_levels(s: pd.Series, bins: int = 10, max_levels: int = 50) -> pd.Series:
    """Discretise a column so any type can go into a contingency table. NaN becomes its own level."""
    if is_categorical(s):
        s = s.astype("object")
        counts = s.value_counts()
        if len(counts) > max_levels:
            keep = set(counts.index[:max_levels])
            s = s.where(s.isin(keep) | s.isna(), "__OTHER__")
        return s.fillna("__MISSING__").astype(str)
    if s.nunique() <= bins:
        return s.astype("object").fillna("__MISSING__").astype(str)
    q = pd.qcut(s, q=bins, duplicates="drop")
    return q.astype("object").fillna("__MISSING__").astype(str)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (0 = no association, 1 = one fully determines the other)."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    obs = table.to_numpy(dtype=float)
    n = obs.sum()
    expected = obs.sum(1, keepdims=True) * obs.sum(0, keepdims=True) / n
    chi2 = ((obs - expected) ** 2 / expected).sum()
    r, k = obs.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def feature_associations(df: pd.DataFrame, target: str, features: list[str] | None = None) -> pd.DataFrame:
    """Score every candidate column against the target and flag ones that should be excluded.

    Columns:
      association           Cramér's V with the target on labelled rows (0-1, higher = more predictive)
      missing_pct           % blank across all rows
      missing_pct_unlabeled % blank on the rows where the target is missing - if this is high the
                            column can't help fill those rows even if it's predictive
      flags / recommended   automatic screening
    """
    features = features or [c for c in df.columns if c != target]
    y_missing = df[target].isna()
    labelled = df.loc[~y_missing]
    y = target_labels(labelled[target]).astype(str)
    n_lab = len(labelled)

    rows = []
    for col in features:
        s_all = df[col]
        s = labelled[col]
        nonnull = s_all.dropna()
        nunique = nonnull.nunique()
        unique_ratio = nunique / max(len(nonnull), 1)
        miss_unlab = float(s_all[y_missing].isna().mean()) if y_missing.any() else 0.0
        assoc = cramers_v(_as_levels(s), y) if n_lab > 1 else 0.0

        flags = []
        if nunique <= 1:
            flags.append("constant")
        if is_categorical(s_all) and unique_ratio > 0.5 and nunique > 50:
            flags.append("ID-like / free text")
        if not is_categorical(s_all) and unique_ratio > 0.999 and nonnull.mod(1).eq(0).all():
            flags.append("possible ID")
        if s_all.isna().mean() > 0.95:
            flags.append("mostly blank")
        if miss_unlab > 0.95:
            flags.append("blank where target is blank")
        if assoc >= 0.95:
            flags.append("near-deterministic (check for leakage)")
        if assoc < 0.05 and not flags:
            flags.append("weak or no signal")

        hard = {
            "constant",
            "ID-like / free text",
            "possible ID",
            "mostly blank",
            "blank where target is blank",
            "weak or no signal",
        }
        rows.append(
            {
                "feature": col,
                "type": "categorical" if is_categorical(s_all) else "numeric",
                "association": round(assoc, 3),
                "missing_pct": round(100 * s_all.isna().mean(), 1),
                "missing_pct_unlabeled": round(100 * miss_unlab, 1),
                "unique": int(nunique),
                "flags": "; ".join(flags),
                "recommended": not (hard & set(flags)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["recommended", "association"], ascending=[False, False]).reset_index(drop=True)


def missingness_shift(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    """How different are the rows missing the target from the labelled rows?

    High values mean the blanks aren't random (e.g. one branch never fills the field). The model can
    still work, but its test-set accuracy may be optimistic for those rows.
    """
    is_missing = df[target].isna().map({True: "missing", False: "labelled"})
    rows = []
    for col in features:
        rows.append({"feature": col, "shift": round(cramers_v(_as_levels(df[col]), is_missing), 3)})
    return pd.DataFrame(rows).sort_values("shift", ascending=False).reset_index(drop=True)


def class_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    y = target_labels(df[target])
    counts = y.value_counts(dropna=True)
    return pd.DataFrame(
        {"class": counts.index.astype(str), "count": counts.to_numpy(), "pct": (100 * counts / counts.sum()).round(1).to_numpy()}
    )


def _allowed_set(counts: pd.Series, min_coverage: float) -> list[str]:
    """Smallest group of classes (most common first) covering ``min_coverage`` of ``counts``."""
    counts = counts[counts > 0].sort_values(ascending=False)
    k = int((counts.cumsum() / counts.sum() < min_coverage).sum()) + 1
    return list(map(str, counts.index[:k]))


def value_rules(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    min_rows: int = 30,
    min_coverage: float = 0.98,
    min_holdout_coverage: float = 0.97,
    max_share_of_classes: float = 0.5,
    max_book_share: float = 0.5,
    min_rows_per_class: float = 10.0,
    max_levels: int = 10,
    flag_levels: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Find values that limit the target to a small set of classes.

    'Whenever <feature> is <value>, the target is one of <classes>'. A one-to-one rule is the case of a
    single class. For example, ``bcc_indicator = Y`` might always mean one of the BCC risk-rating
    templates, without saying which.

    Deliberately strict, so chance patterns (e.g. a small state that happens to contain few templates)
    are not reported:
      * only flag-like columns with at most ``max_levels`` distinct values are searched;
      * the value must be on ``min_rows`` labelled loans, with at least ``min_rows_per_class`` loans
        per allowed class on average;
      * it must narrow things down: at most ``max_share_of_classes`` of the classes, which together
        make up at most ``max_book_share`` of all labelled loans;
      * holdout check: the allowed set is built on a random half of the loans and must still cover
        ``min_holdout_coverage`` of the other half. A real rule passes; a coincidence doesn't.
    Rules from columns with at most ``flag_levels`` values (yes/no indicators) are marked
    ``recommended``; others are shown but not pre-selected.
    """
    labelled = df.loc[df[target].notna()]
    y = target_labels(labelled[target]).astype(str)
    n_classes_total = y.nunique()
    max_allowed = max(1, int(n_classes_total * max_share_of_classes))
    book_share = y.value_counts(normalize=True)
    half = np.random.default_rng(seed).random(len(labelled)) < 0.5
    blank = df.loc[df[target].isna()]
    rows = []
    for col in features:
        s = labelled[col]
        n_levels = s.nunique()
        if col == target or n_levels > max_levels:
            continue
        tab = pd.crosstab(s, y)
        tab_a = pd.crosstab(s[half], y[half]).reindex(index=tab.index, columns=tab.columns, fill_value=0)
        tab_b = tab - tab_a
        for value, counts in tab.iterrows():
            n = int(counts.sum())
            if n < min_rows or tab_a.loc[value].sum() == 0 or tab_b.loc[value].sum() == 0:
                continue
            allowed = _allowed_set(counts, min_coverage)
            k = len(allowed)
            if k > max_allowed or n / k < min_rows_per_class or book_share[allowed].sum() > max_book_share:
                continue
            from_a = _allowed_set(tab_a.loc[value], min_coverage)
            holdout = float(tab_b.loc[value, from_a].sum() / tab_b.loc[value].sum())
            if holdout < min_holdout_coverage:
                continue
            # Always text, so the table never mixes types ("Y" next to 1); matching converts back as needed.
            shown = str(value)
            if not is_categorical(s):
                shown = str(int(value)) if float(value).is_integer() else f"{float(value):g}"
            rows.append(
                {
                    "feature": col,
                    "value": shown,
                    "classes": sorted(allowed),
                    "n_classes": k,
                    "labelled_rows": n,
                    "coverage": round(float(counts[allowed].sum() / n), 4),
                    "holdout_coverage": round(holdout, 4),
                    "blank_rows_with_value": int((blank[col] == value).sum()),
                    "recommended": bool(n_levels <= flag_levels),
                }
            )
    cols = [
        "feature", "value", "classes", "n_classes", "labelled_rows", "coverage", "holdout_coverage",
        "blank_rows_with_value", "recommended",
    ]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["recommended", "n_classes", "labelled_rows"], ascending=[False, True, False]).reset_index(
        drop=True
    )
