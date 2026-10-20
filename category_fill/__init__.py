"""Predict a partially-missing categorical column with XGBoost."""

from .analysis import class_distribution, feature_associations, missingness_shift, profile_columns, suggest_targets
from .data import clean_frame, load_table
from .filters import Rule, population_mask, rules_from_json, rules_summary, rules_to_json
from .model import FillResult, TrainConfig, feature_subset_curve, fill_frame, train_and_fill
from .report import report_workbook

__all__ = [
    "FillResult",
    "Rule",
    "TrainConfig",
    "class_distribution",
    "clean_frame",
    "feature_associations",
    "feature_subset_curve",
    "fill_frame",
    "load_table",
    "missingness_shift",
    "population_mask",
    "profile_columns",
    "report_workbook",
    "rules_from_json",
    "rules_summary",
    "rules_to_json",
    "suggest_targets",
    "train_and_fill",
]
