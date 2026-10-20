"""Export results to a multi-sheet Excel workbook."""

from __future__ import annotations

import io

import pandas as pd

from .model import FillResult


def summary_frame(result: FillResult) -> pd.DataFrame:
    m = result.metrics
    rows = [
        ("Target column", result.target),
        ("Classes predicted", m["n_classes"]),
        ("Labelled rows", m["n_labelled"]),
        ("Train / validation / test rows", f'{m["n_train"]} / {m["n_val"]} / {m["n_test"]}'),
        ("Blank rows filled", m["n_blank"]),
        ("Test accuracy", round(m["accuracy"], 4)),
        ("Baseline accuracy (always guess most common class)", round(m["baseline_accuracy"], 4)),
        ("Most common class", m["baseline_class"]),
        ("Test top-3 accuracy", round(m["top3_accuracy"], 4)),
        ("Macro F1 (every class weighted equally)", round(m["macro_f1"], 4)),
        ("Weighted F1", round(m["weighted_f1"], 4)),
        ("Log loss", round(m["log_loss"], 4)),
        ("Trees (early stopping)", m["trees"]),
        ("Features used", ", ".join(result.features)),
        ("Classes too rare to predict", ", ".join(f"{k} ({v})" for k, v in result.dropped_classes.items()) or "none"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def report_workbook(
    result: FillResult,
    filled: pd.DataFrame | None = None,
    associations: pd.DataFrame | None = None,
    subset_curve: pd.DataFrame | None = None,
    population_rules: pd.DataFrame | None = None,
) -> bytes:
    sheets = {
        "summary": summary_frame(result),
        "per_class": result.per_class.reset_index(),
        "confusion_matrix": result.confusion.reset_index(),
        "feature_importance": result.importance,
        "confidence_bins": result.confidence_bins,
        "confidence_cutoffs": result.thresholds,
        "blank_row_predictions": result.predictions,
        "test_set_predictions": result.test_predictions,
    }
    if population_rules is not None and len(population_rules):
        sheets["population_rules"] = population_rules
    if associations is not None:
        sheets["feature_screening"] = associations
    if subset_curve is not None:
        sheets["feature_subsets"] = subset_curve
    if filled is not None:
        sheets["filled_data"] = filled
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            frame.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()
