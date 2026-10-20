"""Command line: python -m category_fill data.csv [--target COLUMN] [--out results/]"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import (
    TrainConfig,
    clean_frame,
    feature_associations,
    feature_subset_curve,
    fill_frame,
    load_table,
    missingness_shift,
    report_workbook,
    suggest_targets,
    train_and_fill,
)
from .report import summary_frame


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill a partially-missing category column with XGBoost.")
    ap.add_argument("data", help="CSV / Excel / Parquet file")
    ap.add_argument("--target", help="column to fill (default: auto-detect)")
    ap.add_argument("--features", nargs="+", help="feature columns (default: automatic screening)")
    ap.add_argument("--exclude", nargs="+", default=[], help="columns to never use as features")
    ap.add_argument("--categorical", nargs="+", default=[], help="numeric-looking columns to treat as categories")
    ap.add_argument("--missing-tokens", nargs="+", default=[], help="extra values meaning blank, e.g. Unknown")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="leave predictions below this blank")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-class-count", type=int, default=10)
    ap.add_argument("--balance-classes", action="store_true")
    ap.add_argument("--subset-curve", action="store_true", help="also test accuracy with top-k features")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args(argv)

    raw = load_table(args.data)
    df = clean_frame(raw, extra_missing_tokens=args.missing_tokens, force_categorical=args.categorical)
    target = args.target or next(iter(suggest_targets(df)), None)
    if not target or target not in df:
        ap.error(f"could not find target column; pass --target. Columns: {list(df.columns)}")
    print(f"Target: {target}  ({int(df[target].isna().sum())} blank of {len(df)} rows)")

    candidates = [c for c in df.columns if c != target and c not in args.exclude]
    assoc = feature_associations(df, target, candidates)
    print("\nFeature screening (association = Cramér's V with target):")
    print(assoc.to_string(index=False))
    shift = missingness_shift(df, target, candidates)
    if shift["shift"].iloc[0] > 0.1:
        print(f"\nNote: blank rows differ from labelled rows most on: {', '.join(shift.loc[shift['shift'] > 0.1, 'feature'])}")

    features = args.features or list(assoc.loc[assoc["recommended"], "feature"])
    cfg = TrainConfig(test_size=args.test_size, min_class_count=args.min_class_count, balance_classes=args.balance_classes)
    result = train_and_fill(df, target, features, cfg, progress=lambda m: print(f"... {m}"))

    print("\n" + summary_frame(result).to_string(index=False))
    print("\nPer class (test set):\n" + result.per_class.to_string())
    print("\nFeature importance:\n" + result.importance.to_string(index=False))
    print("\nConfidence cut-offs:\n" + result.thresholds.to_string(index=False))

    curve = None
    if args.subset_curve:
        curve = feature_subset_curve(df, target, list(result.importance["feature"]), cfg, progress=lambda m: print(f"... {m}"))
        print("\nAccuracy by number of features:\n" + curve.to_string(index=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filled = fill_frame(raw, result, args.min_confidence)
    filled.to_csv(out / "filled_data.csv", index=False)
    result.predictions.to_csv(out / "blank_row_predictions.csv", index=False)
    (out / "report.xlsx").write_bytes(report_workbook(result, filled, assoc, curve))
    print(f"\nWrote {out}/filled_data.csv, blank_row_predictions.csv, report.xlsx")


if __name__ == "__main__":
    main()
