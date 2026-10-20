"""XGBoost training, evaluation and prediction of a missing categorical column."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from .analysis import target_labels
from .data import is_categorical

OTHER = "__OTHER__"
CONFIDENCE_BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


@dataclass
class TrainConfig:
    test_size: float = 0.2  # share of labelled rows held out for the final readout
    val_size: float = 0.15  # share of the remaining rows used for early stopping
    min_class_count: int = 10  # classes rarer than this are not predicted (too few examples)
    max_categories: int = 100  # per feature; rarer levels are grouped into __OTHER__
    n_estimators: int = 2000  # upper bound, early stopping picks the real number
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    early_stopping_rounds: int = 50
    balance_classes: bool = False  # up-weight rare classes (better recall on them, lower overall accuracy)
    refit_on_all: bool = True  # retrain on every labelled row before filling the blanks
    random_state: int = 42
    n_jobs: int = -1


class Preprocessor:
    """Fixes the feature set and category levels so training and prediction see identical columns."""

    def __init__(self, max_categories: int = 100):
        self.max_categories = max_categories
        self.features: list[str] = []
        self.categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, features: list[str]) -> "Preprocessor":
        self.features = list(features)
        self.categories = {}
        for col in self.features:
            if is_categorical(df[col]):
                counts = df[col].value_counts()
                levels = [str(v) for v in counts.index[: self.max_categories]]
                if len(counts) > self.max_categories:
                    levels.append(OTHER)
                self.categories[col] = levels
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.features:
            s = df[col]
            if col in self.categories:
                levels = self.categories[col]
                s = s.astype("object")
                if OTHER in levels:
                    s = s.where(s.isin(levels) | s.isna(), OTHER)
                out[col] = pd.Categorical(s, categories=levels)
            else:
                out[col] = s.astype("float64")
        return pd.DataFrame(out, index=df.index)


@dataclass
class FillResult:
    target: str
    features: list[str]
    config: TrainConfig
    classes: list[str]
    dropped_classes: dict[str, int]
    metrics: dict
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    importance: pd.DataFrame
    confidence_bins: pd.DataFrame
    thresholds: pd.DataFrame
    test_predictions: pd.DataFrame
    predictions: pd.DataFrame
    model: xgb.XGBClassifier = field(repr=False)
    preprocessor: Preprocessor = field(repr=False)


def _make_model(cfg: TrainConfig, n_classes: int, n_estimators: int | None = None, early_stopping: bool = True):
    params = dict(
        n_estimators=n_estimators or cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        min_child_weight=cfg.min_child_weight,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=4,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
    )
    if n_classes > 2:
        params.update(objective="multi:softprob", num_class=n_classes, eval_metric="mlogloss")
    else:
        params.update(objective="binary:logistic", eval_metric="logloss")
    if early_stopping:
        params["early_stopping_rounds"] = cfg.early_stopping_rounds
    return xgb.XGBClassifier(**params)


def _weights(y: np.ndarray, cfg: TrainConfig):
    return compute_sample_weight("balanced", y) if cfg.balance_classes else None


def _proba(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)


def _top_k(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> pd.DataFrame:
    k = min(k, proba.shape[1])
    order = np.argsort(-proba, axis=1)[:, :k]
    data = {}
    for i in range(k):
        data[f"choice_{i + 1}"] = classes[order[:, i]]
        data[f"prob_{i + 1}"] = np.round(proba[np.arange(len(proba)), order[:, i]], 4)
    return pd.DataFrame(data)


def confidence_tables(correct: np.ndarray, confidence: np.ndarray, unlabeled_confidence: np.ndarray | None = None):
    """How accurate is the model at each confidence level, and what does a cut-off buy you?"""
    bins = pd.cut(confidence, CONFIDENCE_BINS, right=False)
    t = (
        pd.DataFrame({"bin": bins, "correct": correct})
        .groupby("bin", observed=False)
        .agg(rows=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
    t["bin"] = t["bin"].map(lambda b: f"{b.left:.0%}-{min(b.right, 1):.0%}")
    t["share_of_test"] = (t["rows"] / max(len(correct), 1)).round(3)
    t["accuracy"] = t["accuracy"].round(3)

    rows = []
    for th in THRESHOLDS:
        keep = confidence >= th
        row = {
            "min_confidence": th,
            "test_coverage": round(keep.mean(), 3),
            "test_accuracy": round(correct[keep].mean(), 3) if keep.any() else np.nan,
        }
        if unlabeled_confidence is not None:
            row["blank_rows_filled"] = int((unlabeled_confidence >= th).sum())
        rows.append(row)
    return t, pd.DataFrame(rows)


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, repeats: int = 3, seed: int = 0) -> pd.Series:
    """Drop in test accuracy when each column is shuffled. Model-agnostic and honest about usefulness."""
    rng = np.random.default_rng(seed)
    base = accuracy_score(y, model.predict(X))
    drops = {}
    for col in X.columns:
        scores = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))]
            if isinstance(X[col].dtype, pd.CategoricalDtype):
                Xp[col] = pd.Categorical(Xp[col], categories=X[col].cat.categories)
            scores.append(accuracy_score(y, model.predict(Xp)))
        drops[col] = base - float(np.mean(scores))
    return pd.Series(drops)


@dataclass
class _Prepared:
    pre: Preprocessor
    X_lab: pd.DataFrame
    y_lab: np.ndarray
    X_unlab: pd.DataFrame
    unlab_index: pd.Index
    classes: np.ndarray
    dropped: dict
    splits: dict


def _prepare(df: pd.DataFrame, target: str, features: list[str], cfg: TrainConfig) -> _Prepared:
    if target in features:
        features = [f for f in features if f != target]
    if not features:
        raise ValueError("Select at least one feature column.")

    labels = target_labels(df[target])
    is_unlab = labels.isna().to_numpy()
    counts = labels[~is_unlab].value_counts()
    keep_classes = counts[counts >= cfg.min_class_count].index
    dropped = {str(k): int(v) for k, v in counts[counts < cfg.min_class_count].items()}
    if len(keep_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {cfg.min_class_count} labelled rows; found {len(keep_classes)}."
        )

    pre = Preprocessor(cfg.max_categories).fit(df, features)
    lab_mask = (~is_unlab) & labels.isin(keep_classes).fillna(False).to_numpy()
    X_lab = pre.transform(df.loc[lab_mask])
    classes = np.array(sorted(map(str, keep_classes)))
    y_lab = np.searchsorted(classes, labels[lab_mask].astype(str).to_numpy())

    idx = np.arange(len(X_lab))
    idx_trval, idx_test = train_test_split(idx, test_size=cfg.test_size, stratify=y_lab, random_state=cfg.random_state)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=cfg.val_size, stratify=y_lab[idx_trval], random_state=cfg.random_state
    )
    return _Prepared(
        pre=pre,
        X_lab=X_lab,
        y_lab=y_lab,
        X_unlab=pre.transform(df.loc[is_unlab]),
        unlab_index=df.index[is_unlab],
        classes=classes,
        dropped=dropped,
        splits={"train": idx_train, "val": idx_val, "test": idx_test},
    )


def _fit_eval(p: _Prepared, cfg: TrainConfig, columns: list[str] | None = None):
    cols = columns or list(p.X_lab.columns)
    tr, va, te = p.splits["train"], p.splits["val"], p.splits["test"]
    X = p.X_lab[cols]
    model = _make_model(cfg, len(p.classes))
    model.fit(
        X.iloc[tr],
        p.y_lab[tr],
        sample_weight=_weights(p.y_lab[tr], cfg),
        eval_set=[(X.iloc[va], p.y_lab[va])],
        verbose=False,
    )
    proba = _proba(model, X.iloc[te])
    return model, proba


def train_and_fill(
    df: pd.DataFrame, target: str, features: list[str], config: TrainConfig | None = None, progress=None
) -> FillResult:
    """Full pipeline: split -> train with early stopping -> evaluate on held-out test -> fill blanks."""
    cfg = config or TrainConfig()
    say = progress or (lambda msg: None)

    say("Preparing data")
    p = _prepare(df, target, features, cfg)
    te = p.splits["test"]
    y_test = p.y_lab[te]

    say("Training XGBoost (early stopping on validation set)")
    model, proba = _fit_eval(p, cfg)
    best_iter = int(getattr(model, "best_iteration", cfg.n_estimators - 1) or 0) + 1
    pred = proba.argmax(1)
    conf = proba.max(1)
    correct = pred == y_test

    majority = np.bincount(p.y_lab[p.splits["train"]]).argmax()
    top3 = (np.argsort(-proba, axis=1)[:, :3] == y_test[:, None]).any(1).mean()
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "baseline_accuracy": float((y_test == majority).mean()),
        "baseline_class": str(p.classes[majority]),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "weighted_f1": f1_score(y_test, pred, average="weighted"),
        "top3_accuracy": float(top3),
        "log_loss": log_loss(y_test, proba, labels=np.arange(len(p.classes))),
        "n_labelled": int(len(p.y_lab)),
        "n_train": int(len(p.splits["train"])),
        "n_val": int(len(p.splits["val"])),
        "n_test": int(len(te)),
        "n_blank": int(len(p.X_unlab)),
        "n_classes": int(len(p.classes)),
        "trees": best_iter,
    }

    report = classification_report(
        y_test, pred, labels=np.arange(len(p.classes)), target_names=list(p.classes), output_dict=True, zero_division=0
    )
    per_class = (
        pd.DataFrame({c: report[c] for c in p.classes})
        .T.rename(columns={"f1-score": "f1", "support": "test_rows"})
        .assign(train_rows=np.bincount(p.y_lab[p.splits["train"]], minlength=len(p.classes)))
    )
    per_class["test_rows"] = per_class["test_rows"].astype(int)
    per_class = per_class.round(3).sort_values("test_rows", ascending=False)
    per_class.index.name = "class"

    cm = confusion_matrix(y_test, pred, labels=np.arange(len(p.classes)))
    confusion = pd.DataFrame(cm, index=p.classes, columns=p.classes)
    confusion.index.name = "actual \\ predicted"

    say("Measuring feature importance")
    X_test = p.X_lab.iloc[te]
    gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"))
    perm = permutation_importance(model, X_test, y_test, seed=cfg.random_state)
    importance = pd.DataFrame(
        {
            "feature": p.X_lab.columns,
            "accuracy_drop_if_shuffled": [round(perm.get(c, 0.0), 4) for c in p.X_lab.columns],
            "gain_share": [gain.get(c, 0.0) for c in p.X_lab.columns],
        }
    )
    importance["gain_share"] = (importance["gain_share"] / max(importance["gain_share"].sum(), 1e-12)).round(4)
    importance = importance.sort_values(["accuracy_drop_if_shuffled", "gain_share"], ascending=False).reset_index(
        drop=True
    )

    test_predictions = pd.concat(
        [
            pd.DataFrame({"actual": p.classes[y_test], "predicted": p.classes[pred], "correct": correct}),
            _top_k(proba, p.classes),
        ],
        axis=1,
    )

    final_model = model
    if cfg.refit_on_all and len(p.X_unlab):
        say("Refitting on all labelled rows")
        n_trees = max(1, int(best_iter * 1.1))
        final_model = _make_model(cfg, len(p.classes), n_estimators=n_trees, early_stopping=False)
        final_model.fit(p.X_lab, p.y_lab, sample_weight=_weights(p.y_lab, cfg), verbose=False)

    say("Predicting blank rows")
    if len(p.X_unlab):
        unlab_proba = _proba(final_model, p.X_unlab)
        predictions = _top_k(unlab_proba, p.classes)
        predictions.insert(0, "row", p.unlab_index)
        predictions.insert(1, "predicted", predictions["choice_1"])
        predictions.insert(2, "confidence", predictions["prob_1"])
        predictions = predictions.drop(columns=["choice_1", "prob_1"])
        unlab_conf = unlab_proba.max(1)
    else:
        predictions = pd.DataFrame(columns=["row", "predicted", "confidence"])
        unlab_conf = np.array([])

    bins, thresholds = confidence_tables(correct, conf, unlab_conf)
    return FillResult(
        target=target,
        features=list(p.X_lab.columns),
        config=cfg,
        classes=list(p.classes),
        dropped_classes=p.dropped,
        metrics=metrics,
        per_class=per_class,
        confusion=confusion,
        importance=importance,
        confidence_bins=bins,
        thresholds=thresholds,
        test_predictions=test_predictions,
        predictions=predictions,
        model=final_model,
        preprocessor=p.pre,
    )


def feature_subset_curve(
    df: pd.DataFrame, target: str, ranked_features: list[str], config: TrainConfig | None = None, sizes=None, progress=None
) -> pd.DataFrame:
    """Retrain using only the top-k features (in the given order) to find the smallest set that works."""
    cfg = config or TrainConfig()
    say = progress or (lambda msg: None)
    p = _prepare(df, target, ranked_features, cfg)
    n = len(ranked_features)
    sizes = sorted({k for k in (sizes or [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]) if k < n} | {n})
    y_test = p.y_lab[p.splits["test"]]
    rows = []
    prev = 0
    for k in sizes:
        say(f"Training with top {k} feature(s)")
        cols = ranked_features[:k]
        _, proba = _fit_eval(p, cfg, cols)
        pred = proba.argmax(1)
        added, prev = ranked_features[prev:k], k
        rows.append(
            {
                "n_features": k,
                "added": ", ".join(added),
                "accuracy": round(accuracy_score(y_test, pred), 4),
                "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
            }
        )
    return pd.DataFrame(rows)


def fill_frame(
    original: pd.DataFrame, result: FillResult, min_confidence: float = 0.0, in_population: pd.Series | None = None
) -> pd.DataFrame:
    """Return the original data with the target filled in, plus provenance columns.

    ``original`` must share its index with the frame the model was trained on.
    Rows below ``min_confidence`` are left blank and marked 'blank (low confidence)'.
    Rows where ``in_population`` is False were excluded by population rules: they keep their original
    value and are marked 'excluded'.
    """
    out = original.copy()
    t = result.target
    pred = result.predictions.set_index("row")
    filled_col = f"{t}_filled"
    out[filled_col] = original[t].astype("object") if t in original else None
    out[f"{t}_source"] = "original"
    if len(pred):
        accept = pred.index[pred["confidence"] >= min_confidence]
        reject = pred.index[pred["confidence"] < min_confidence]
        out.loc[pred.index, filled_col] = None
        out.loc[accept, filled_col] = pred.loc[accept, "predicted"].to_numpy()
        out.loc[accept, f"{t}_source"] = "predicted"
        out.loc[reject, f"{t}_source"] = "blank (low confidence)"
        out[f"{t}_confidence"] = pred["confidence"].reindex(out.index)
        if "choice_2" in pred:
            out[f"{t}_2nd_choice"] = pred["choice_2"].reindex(out.index)
    if in_population is not None:
        out.loc[~in_population.reindex(out.index, fill_value=True).to_numpy(dtype=bool), f"{t}_source"] = "excluded"
    return out
