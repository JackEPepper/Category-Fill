"""Population rules: drop loans from the analysis by the values they hold in one or more columns.

A rule picks rows in a single column by listed values, a numeric range, and/or blanks.
``exclude`` rules remove the rows they match; ``keep`` rules remove every row they *don't* match.
A loan stays in the population only if it survives every rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .data import is_categorical

BLANK_TOKEN = "<blank>"


def fmt_num(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,}"


@dataclass
class Rule:
    column: str
    values: list = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    blank: bool = False
    mode: str = "exclude"  # "exclude" or "keep"

    def __post_init__(self):
        if self.mode not in ("exclude", "keep"):
            raise ValueError(f"Rule mode must be 'exclude' or 'keep', got {self.mode!r}")

    @property
    def is_empty(self) -> bool:
        return not (self.values or self.blank or self.min is not None or self.max is not None)

    def matches(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df:
            raise ValueError(f"Rule column {self.column!r} not found in data")
        s = df[self.column]
        hit = pd.Series(False, index=df.index)
        if self.values:
            if is_categorical(s):
                hit |= s.astype("string").isin([str(v) for v in self.values]).fillna(False).astype(bool)
            else:
                nums = pd.to_numeric(pd.Series(self.values), errors="coerce").dropna()
                hit |= s.isin(nums.to_numpy())
        if (self.min is not None or self.max is not None) and not is_categorical(s):
            lo = -np.inf if self.min is None else self.min
            hi = np.inf if self.max is None else self.max
            hit |= s.between(lo, hi).fillna(False).astype(bool)
        if self.blank:
            hit |= s.isna()
        return hit

    def describe(self) -> str:
        parts = []
        if self.values:
            shown = ", ".join(map(str, self.values[:8])) + (f" (+{len(self.values) - 8} more)" if len(self.values) > 8 else "")
            parts.append(f"is one of [{shown}]")
        if self.min is not None and self.max is not None:
            parts.append(f"is between {fmt_num(self.min)} and {fmt_num(self.max)}")
        elif self.min is not None:
            parts.append(f"is at least {fmt_num(self.min)}")
        elif self.max is not None:
            parts.append(f"is at most {fmt_num(self.max)}")
        if self.blank:
            parts.append("is blank")
        verb = "Exclude" if self.mode == "exclude" else "Keep only"
        return f"{verb} loans where {self.column} " + " or ".join(parts or ["(no criteria)"])


def population_mask(df: pd.DataFrame, rules: list[Rule]) -> pd.Series:
    """True for loans that stay in the population. Rules with no criteria are ignored."""
    keep = pd.Series(True, index=df.index)
    for r in rules:
        if r.is_empty:
            continue
        hit = r.matches(df)
        keep &= ~hit if r.mode == "exclude" else hit
    return keep


def rules_summary(df: pd.DataFrame, rules: list[Rule]) -> pd.DataFrame:
    """Per rule: how many loans it matches, and how many it removes on top of the rules above it."""
    keep = pd.Series(True, index=df.index)
    rows = []
    for r in rules:
        if r.is_empty:
            continue
        hit = r.matches(df)
        removed_by_rule = hit if r.mode == "exclude" else ~hit
        newly = int((keep & removed_by_rule).sum())
        keep &= ~removed_by_rule
        rows.append(
            {
                "rule": r.describe(),
                "loans_matched": int(hit.sum()),
                "loans_removed": newly,
                "remaining": int(keep.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["rule", "loans_matched", "loans_removed", "remaining"])


def rules_to_json(rules: list[Rule]) -> str:
    return json.dumps([asdict(r) for r in rules], indent=2)


def rules_from_json(text: str) -> list[Rule]:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("rules", [])
    return [Rule(**{k: v for k, v in d.items() if k in Rule.__dataclass_fields__}) for d in data]


def parse_rule_arg(text: str, mode: str) -> Rule:
    """CLI form: 'column=value1|value2|<blank>' or 'column=100000..5000000' (either end optional)."""
    if "=" not in text:
        raise ValueError(f"Expected column=values, got {text!r}")
    column, spec = (p.strip() for p in text.split("=", 1))
    rule = Rule(column=column, mode=mode)
    for part in (p.strip() for p in spec.split("|")):
        if part == BLANK_TOKEN:
            rule.blank = True
        elif ".." in part:
            lo, hi = (x.strip() for x in part.split("..", 1))
            rule.min = float(lo) if lo else None
            rule.max = float(hi) if hi else None
        elif part:
            rule.values.append(part)
    return rule
