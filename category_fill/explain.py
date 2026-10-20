"""Explain the model with a single decision tree you can show to people.

The XGBoost model is hundreds of trees (one set per class), which no one can read. Instead we fit one
shallow "surrogate" decision tree to imitate the XGBoost model's answers, and report how often the two
agree (fidelity). A high-fidelity surrogate is an honest picture of what the model is doing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .analysis import target_labels
from .model import FillResult

_SEP = " = "


@dataclass
class Surrogate:
    tree: DecisionTreeClassifier
    feature_names: list[str]
    class_names: list[str]
    fidelity: float  # share of held-out loans where the tree and XGBoost give the same answer
    accuracy: float  # share of held-out labelled loans the tree gets right
    model_accuracy: float  # XGBoost on the same labelled loans (for comparison)
    n_rows: int
    depth: int
    baseline_fidelity: float = float("nan")  # agreement of a "tree" that always gives the most common answer
    focus: str | None = None  # set when explaining one category against the rest
    has_blanks: frozenset = frozenset()  # numeric columns with missing values (splits must say where blanks go)


def _encode(X: pd.DataFrame) -> pd.DataFrame:
    """One column per category level ('collateral_type = Equipment' -> 0/1); numbers kept as-is (NaN allowed)."""
    parts = []
    for col in X.columns:
        s = X[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            levels = [lv for lv in s.cat.categories if (s == lv).any()]
            parts.append(pd.DataFrame({f"{col}{_SEP}{lv}": (s == lv).astype(float) for lv in levels}, index=X.index))
        else:
            parts.append(s.astype(float).to_frame(col))
    return pd.concat(parts, axis=1)


OTHER = "(any other category)"


def fit_surrogate(
    df: pd.DataFrame,
    result: FillResult,
    max_depth: int = 4,
    focus: str | None = None,
    min_leaf_share: float = 0.002,
    seed: int = 0,
) -> Surrogate:
    """Train a single tree on the model's own predictions for every loan in ``df`` (the population).

    With ``focus`` set to a category, the tree only explains "this category vs. everything else", which
    is far easier to show faithfully in a few levels than all categories at once.
    """
    X = result.preprocessor.transform(df)
    model_pred = np.asarray(result.classes, dtype=object)[result.model.predict(X)]
    labels = target_labels(df[result.target]).astype(object)
    if focus is not None:
        model_pred = np.where(model_pred == focus, focus, OTHER).astype(object)
        labels = labels.where(labels.isna() | (labels == focus), OTHER)
    E = _encode(X)
    idx_tr, idx_te = train_test_split(np.arange(len(E)), test_size=0.25, random_state=seed)

    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=max(5, int(min_leaf_share * len(idx_tr))), random_state=seed
    )
    tree.fit(E.iloc[idx_tr], model_pred[idx_tr])
    tree_pred = tree.predict(E.iloc[idx_te])  # agreement is measured on loans the tree didn't see...
    tree.fit(E, model_pred)  # ...then it is refit on every loan so the counts in the picture match the book

    truth = labels.iloc[idx_te].to_numpy(dtype=object)
    has_truth = pd.notna(truth)
    return Surrogate(
        tree=tree,
        feature_names=list(E.columns),
        class_names=[str(c) for c in tree.classes_],
        fidelity=float((tree_pred == model_pred[idx_te]).mean()),
        accuracy=float((tree_pred[has_truth] == truth[has_truth]).mean()) if has_truth.any() else float("nan"),
        model_accuracy=float((model_pred[idx_te][has_truth] == truth[has_truth]).mean()) if has_truth.any() else float("nan"),
        n_rows=len(E),
        depth=max_depth,
        focus=focus,
        has_blanks=frozenset(c for c in E.columns if E[c].isna().any()),
        baseline_fidelity=float(pd.Series(model_pred[idx_te]).value_counts(normalize=True).iloc[0]),
    )


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


def _condition(s: Surrogate, node: int, go_left: bool) -> str:
    """Readable text for the split at ``node``. sklearn sends rows with value <= threshold to the left."""
    name = s.feature_names[s.tree.tree_.feature[node]]
    thr = s.tree.tree_.threshold[node]
    if _SEP in name:  # one-hot column: <= 0.5 means "is not"
        col, level = name.split(_SEP, 1)
        return f"{col} is not {level}" if go_left else f"{col} is {level}"
    if np.isinf(thr):  # sklearn's way of splitting blanks (right) from filled-in values (left)
        return f"{name} is filled in" if go_left else f"{name} is blank"
    text = f"{name} ≤ {_fmt(thr)}" if go_left else f"{name} > {_fmt(thr)}"
    if name in s.has_blanks and bool(s.tree.tree_.missing_go_to_left[node]) == go_left:
        text += " (or blank)"
    return text


def _node_summary(s: Surrogate, node: int) -> tuple[str, float, int]:
    counts = s.tree.tree_.value[node][0]
    n = int(s.tree.tree_.n_node_samples[node])
    k = int(np.argmax(counts))
    return s.class_names[k], float(counts[k] / counts.sum()), n


def rules_text(s: Surrogate) -> str:
    """Plain-English IF/THEN rules, one per leaf, biggest groups first."""
    t = s.tree.tree_
    total = t.n_node_samples[0]
    leaves = []

    def walk(node, path):
        if t.children_left[node] == -1:
            cls, share, n = _node_summary(s, node)
            leaves.append((n, path, cls, share))
            return
        walk(t.children_left[node], path + [_condition(s, node, True)])
        walk(t.children_right[node], path + [_condition(s, node, False)])

    walk(0, [])
    what = f"Decision rules for {s.focus} vs. all other categories" if s.focus else "Decision rules"
    lines = [
        f"{what} that imitate the model (agree with it on {s.fidelity:.1%} of held-out loans).",
        f"Tree depth {s.depth}; {len(leaves)} rules; built from {s.n_rows:,} loans.",
        "",
    ]
    for i, (n, path, cls, share) in enumerate(sorted(leaves, key=lambda x: -x[0]), 1):
        cond = "\n    AND ".join(path) if path else "(all loans)"
        lines.append(f"Rule {i}: IF {cond}\n    THEN {cls}  ({share:.0%} of {n:,} loans, {n / total:.1%} of book)\n")
    return "\n".join(lines)


def rules_frame(s: Surrogate) -> pd.DataFrame:
    text_rules = []
    t = s.tree.tree_

    def walk(node, path):
        if t.children_left[node] == -1:
            cls, share, n = _node_summary(s, node)
            text_rules.append({"conditions": " AND ".join(path) or "(all loans)", "predicts": cls, "share": round(share, 3), "loans": n})
            return
        walk(t.children_left[node], path + [_condition(s, node, True)])
        walk(t.children_right[node], path + [_condition(s, node, False)])

    walk(0, [])
    return pd.DataFrame(text_rules).sort_values("loans", ascending=False).reset_index(drop=True)


def tree_figure(s: Surrogate, title: str | None = None):
    """Matplotlib figure of the tree with plain-English node labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    leaves = s.tree.get_n_leaves()
    width = min(80, max(10, 2.6 * leaves))  # inches; very wide trees get a smaller font instead
    fig, ax = plt.subplots(figsize=(width, max(5, 2.3 * (s.tree.get_depth() + 1))), dpi=110)
    artists = plot_tree(
        s.tree, ax=ax, feature_names=s.feature_names, class_names=s.class_names, filled=True, rounded=True,
        impurity=False, fontsize=9 if leaves <= 24 else 7,
    )
    # The root's arrows are labelled True/False; our questions ask the "right branch" condition.
    nodes = []
    for a in artists:
        txt = a.get_text().strip()
        if txt in ("True", "False"):
            a.set_text({"True": "No  ", "False": "  Yes"}[txt])
            a.set_fontsize(10)
        else:
            nodes.append(a)
    # plot_tree draws nodes in the same depth-first order as the tree's node ids: rewrite each label.
    t = s.tree.tree_
    total = t.n_node_samples[0]
    for node, artist in enumerate(nodes):
        cls, share, n = _node_summary(s, node)
        body = f"{n:,} loans ({n / total:.0%})\nmostly {cls} ({share:.0%})"
        if t.children_left[node] == -1:
            artist.set_text(f"→ {cls}\n{share:.0%} of {n:,} loans")
        else:
            artist.set_text(f"{_condition(s, node, False)}?\n{body}")
    what = f"How the model picks out {s.focus}" if s.focus else "How the model decides"
    ax.set_title(
        title
        or f"{what} (simplified): one tree that agrees with the full model on {s.fidelity:.0%} of loans\n"
        "Each box asks a yes/no question. Left = No, Right = Yes.",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def figure_bytes(fig, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    return buf.getvalue()
