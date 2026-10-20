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
from . import explain
from .analysis import value_rules
from .filters import parse_rule_arg, population_mask, rules_from_json, rules_summary
from .report import summary_frame


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fill a partially-missing category column with XGBoost.")
    ap.add_argument("data", help="CSV / Excel / Parquet file")
    ap.add_argument("--target", help="column to fill (default: auto-detect)")
    ap.add_argument("--features", nargs="+", help="feature columns (default: automatic screening)")
    ap.add_argument("--exclude", nargs="+", default=[], help="columns to never use as features")
    ap.add_argument("--categorical", nargs="+", default=[], help="numeric-looking columns to treat as categories")
    ap.add_argument("--missing-tokens", nargs="+", default=[], help="extra values meaning blank, e.g. Unknown")
    ap.add_argument(
        "--exclude-rows",
        action="append",
        default=[],
        metavar="COL=VALUES",
        help="drop loans from the population, e.g. 'loan_type=SBA 7(a)|Line of Credit', 'state=<blank>', "
        "'loan_amount=..50000'. Repeat for more columns.",
    )
    ap.add_argument(
        "--keep-rows",
        action="append",
        default=[],
        metavar="COL=VALUES",
        help="keep only loans matching, same syntax as --exclude-rows",
    )
    ap.add_argument("--rules", help="population rules JSON (as saved from the app)")
    ap.add_argument("--require", nargs="+", default=[], help="features that must be in the model")
    ap.add_argument(
        "--apply-value-rules",
        action="store_true",
        help="limit predictions to the classes each detected value rule allows "
        "(e.g. bcc_indicator=Y -> one of the BCC templates); also makes those columns required",
    )
    ap.add_argument("--tree-depth", type=int, default=4, help="levels in the exported explanation tree")
    ap.add_argument("--min-confidence", type=float, default=0.0, help="leave predictions below this blank")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--min-class-count", type=int, default=10)
    ap.add_argument("--balance-classes", action="store_true")
    ap.add_argument("--subset-curve", action="store_true", help="also test accuracy with top-k features")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args(argv)

    raw = load_table(args.data)
    df = clean_frame(raw, extra_missing_tokens=args.missing_tokens, force_categorical=args.categorical)
    rules = rules_from_json(Path(args.rules).read_text()) if args.rules else []
    rules += [parse_rule_arg(r, "exclude") for r in args.exclude_rows]
    rules += [parse_rule_arg(r, "keep") for r in args.keep_rows]
    in_pop = population_mask(df, rules)
    rule_table = rules_summary(df, rules)
    if rules:
        print("Population rules:\n" + rule_table.to_string(index=False))
        print(f"Population: {int(in_pop.sum())} of {len(df)} loans\n")
    df = df.loc[in_pop]

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

    rules = value_rules(df, target, candidates)
    if len(rules):
        shown = rules.assign(classes=rules["classes"].map(" | ".join))
        print("\nValues that limit the category (value -> one of these classes):\n" + shown.to_string(index=False))
    features = args.features or list(assoc.loc[assoc["recommended"], "feature"])
    required = list(dict.fromkeys(args.require + (list(rules["feature"].unique()) if args.apply_value_rules else [])))
    cfg = TrainConfig(
        test_size=args.test_size,
        min_class_count=args.min_class_count,
        balance_classes=args.balance_classes,
        required_features=tuple(required),
        apply_value_rules=args.apply_value_rules,
    )
    result = train_and_fill(df, target, features, cfg, progress=lambda m: print(f"... {m}"), value_rules=rules)
    if len(result.rule_check):
        print("\nDo the rules hold, and does the model respect them?\n" + result.rule_check.to_string(index=False))
        weak = result.rule_check[result.rule_check["actual_in_allowed"] < 0.95]
        if len(weak):
            print("WARNING: these rules hold on under 95% of held-out loans: " + "; ".join(weak["rule"]))
        if args.apply_value_rules:
            m = result.metrics
            print(f"Test accuracy with rule limits {m['accuracy']:.1%}, without {m['accuracy_without_rules']:.1%}")

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
    filled = fill_frame(raw, result, args.min_confidence, in_population=in_pop)
    filled.to_csv(out / "filled_data.csv", index=False)
    result.predictions.to_csv(out / "blank_row_predictions.csv", index=False)
    (out / "report.xlsx").write_bytes(report_workbook(result, filled, assoc, curve, rule_table))

    import matplotlib.pyplot as plt

    sur = explain.fit_surrogate(df, result, max_depth=args.tree_depth)
    fig = explain.tree_figure(sur)
    for fmt in ("png", "svg", "pdf"):
        (out / f"decision_tree.{fmt}").write_bytes(explain.figure_bytes(fig, fmt))
    (out / "decision_tree_rules.txt").write_text(explain.rules_text(sur), encoding="utf-8")
    plt.close(fig)
    print(f"\nExplanation tree (all categories) agrees with the model on {sur.fidelity:.1%} of held-out loans.")

    # One small, faithful tree per category ("this category vs. the rest").
    (out / "trees").mkdir(exist_ok=True)
    for cls in result.classes:
        one = explain.fit_surrogate(df, result, max_depth=3, focus=cls)
        fig = explain.tree_figure(one)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cls)
        (out / "trees" / f"{safe}.png").write_bytes(explain.figure_bytes(fig, "png"))
        (out / "trees" / f"{safe}_rules.txt").write_text(explain.rules_text(one), encoding="utf-8")
        plt.close(fig)
        print(f"  {cls}: tree agrees with the model on {one.fidelity:.1%} (always 'other': {one.baseline_fidelity:.1%})")
    print(
        f"Wrote {out}/filled_data.csv, blank_row_predictions.csv, report.xlsx, "
        "decision_tree.png/.svg/.pdf, decision_tree_rules.txt, trees/<category>.png"
    )


if __name__ == "__main__":
    main()
