import io

import numpy as np
import pandas as pd
import pytest

import category_fill as cf
from category_fill.analysis import cramers_v
from category_fill.sample import make_loans


@pytest.fixture(scope="module")
def loans():
    raw = make_loans(n=3000, seed=1)
    return raw, cf.clean_frame(raw)


def test_clean_frame_types(loans):
    _, df = loans
    assert df["loan_amount"].dtype == "float64"  # "$1,250,000" parsed
    assert {"origination_date_year", "origination_date_month", "origination_date_days"} <= set(df.columns)
    assert df["industry_sector"].isna().mean() > 0.05  # "" became missing
    assert df["branch"].dtype == "string"


def test_clean_frame_edge_cases():
    raw = pd.DataFrame(
        {
            "code": ["001", "002", "010", "003"],
            "amt": ["(1,000)", "$2,000", "N/A", "3000"],
            "cat": ["a", "Unknown", "b", "a"],
        }
    )
    df = cf.clean_frame(raw, extra_missing_tokens=["unknown"])
    assert df["code"].dtype == "string"  # leading zeros kept as text
    assert df["amt"].tolist()[:2] == [-1000.0, 2000.0] and np.isnan(df["amt"].iloc[2])
    assert df["cat"].isna().sum() == 1


def test_load_table_roundtrip():
    buf = io.BytesIO(b" a ,b\n1,x\n2,\n")
    df = cf.load_table(buf, filename="f.csv")
    assert list(df.columns) == ["a", "b"]


def test_cramers_v_bounds():
    x = pd.Series(list("aabbcc") * 50)
    assert cramers_v(x, x) > 0.95
    rng = np.random.default_rng(0)
    assert cramers_v(pd.Series(rng.choice(list("ab"), 2000)), pd.Series(rng.choice(list("xy"), 2000))) < 0.05


def test_suggest_and_screen(loans):
    _, df = loans
    assert cf.suggest_targets(df)[0] == "industry_sector"
    a = cf.feature_associations(df, "industry_sector")
    row = a.set_index("feature")
    assert not row.loc["loan_id", "recommended"]
    assert row.loc["collateral_type", "recommended"]
    assert row.loc["collateral_type", "association"] > row.loc["state", "association"]
    shift = cf.missingness_shift(df, "industry_sector", ["branch", "state"])
    assert shift.iloc[0]["feature"] == "branch"


def test_train_and_fill(loans):
    raw, df = loans
    feats = ["collateral_type", "property_type", "loan_type", "naics_code", "term_months", "loan_amount"]
    cfg = cf.TrainConfig(n_estimators=300, early_stopping_rounds=20)
    r = cf.train_and_fill(df, "industry_sector", feats, cfg)
    m = r.metrics
    assert m["accuracy"] > m["baseline_accuracy"] + 0.3
    assert m["n_train"] + m["n_val"] + m["n_test"] == m["n_labelled"]
    assert len(r.predictions) == df["industry_sector"].isna().sum()
    assert set(r.predictions["predicted"]) <= set(r.classes)
    assert r.confusion.to_numpy().sum() == m["n_test"]
    assert r.importance["feature"].iloc[0] in feats

    filled = cf.fill_frame(raw, r, min_confidence=0.0)
    assert filled["industry_sector_filled"].notna().all()
    assert (filled["industry_sector_source"] == "predicted").sum() == len(r.predictions)
    strict = cf.fill_frame(raw, r, min_confidence=0.99)
    assert (strict["industry_sector_source"] == "blank (low confidence)").sum() > 0

    import pyarrow as pa
    from category_fill.report import summary_frame

    for frame in (summary_frame(r), r.per_class.reset_index(), r.predictions, r.rule_check, filled):
        pa.Table.from_pandas(frame)  # every table the app shows must be Arrow-compatible

    xlsx = cf.report_workbook(r, filled)
    assert pd.ExcelFile(io.BytesIO(xlsx)).sheet_names[0] == "summary"


def test_rare_classes_dropped(loans):
    _, df = loans
    df = df.copy()
    df.loc[df.index[:3], "industry_sector"] = "Tiny"
    r = cf.train_and_fill(df, "industry_sector", ["collateral_type", "loan_type"], cf.TrainConfig(n_estimators=50))
    assert "Tiny" in r.dropped_classes and "Tiny" not in r.classes


def test_numeric_target_and_binary():
    rng = np.random.default_rng(0)
    x = rng.choice(["a", "b"], 600)
    y = np.where(x == "a", 1.0, 2.0)
    y[:60] = np.nan
    df = pd.DataFrame({"x": pd.Series(x, dtype="string"), "noise": rng.normal(size=600), "y": y})
    r = cf.train_and_fill(df, "y", ["x", "noise"], cf.TrainConfig(n_estimators=50))
    assert r.classes == ["1", "2"] and r.metrics["accuracy"] > 0.95


def test_subset_curve(loans):
    _, df = loans
    curve = cf.feature_subset_curve(
        df, "industry_sector", ["loan_type", "collateral_type", "state"], cf.TrainConfig(n_estimators=100), sizes=[1, 2]
    )
    assert curve["n_features"].tolist() == [1, 2, 3]


def test_population_rules(loans):
    raw, df = loans
    rules = [
        cf.Rule("loan_type", ["Line of Credit", "SBA 7(a)"]),
        cf.Rule("state", ["TX", "CA"], mode="keep"),
        cf.Rule("loan_amount", min=None, max=100000),
        cf.Rule("branch"),  # empty rule: ignored
    ]
    keep = cf.population_mask(df, rules)
    kept = df[keep]
    assert not kept["loan_type"].isin(["Line of Credit", "SBA 7(a)"]).any()
    assert kept["state"].isin(["TX", "CA"]).all()
    assert (kept["loan_amount"] > 100000).all()
    summary = cf.rules_summary(df, rules)
    assert len(summary) == 3 and summary["remaining"].iloc[-1] == keep.sum()
    assert cf.rules_from_json(cf.rules_to_json(rules)) == rules

    r = cf.train_and_fill(kept, "industry_sector", ["collateral_type", "property_type"], cf.TrainConfig(n_estimators=50))
    assert set(r.predictions["row"]) <= set(kept.index)
    filled = cf.fill_frame(raw, r, in_population=keep)
    assert (filled.loc[~keep, "industry_sector_source"] == "excluded").all()
    assert filled.loc[~keep, "industry_sector_filled"].equals(raw.loc[~keep, "industry_sector"].astype(object))


def test_rule_blank_numeric_values_and_cli_parse():
    from category_fill.filters import parse_rule_arg

    df = pd.DataFrame({"x": pd.Series(["a", None, "b"], dtype="string"), "n": [1.0, 2.0, 3.0]})
    assert cf.population_mask(df, [cf.Rule("x", blank=True)]).tolist() == [True, False, True]
    assert cf.population_mask(df, [cf.Rule("n", ["2"])]).tolist() == [True, False, True]
    r = parse_rule_arg("loan_amount=1000..|<blank>|", "keep")
    assert (r.column, r.min, r.max, r.blank, r.mode) == ("loan_amount", 1000.0, None, True, "keep")
    r = parse_rule_arg("loan_type=SBA 7(a)|Term Loan", "exclude")
    assert r.values == ["SBA 7(a)", "Term Loan"]
    with pytest.raises(ValueError):
        cf.population_mask(df, [cf.Rule("nope", ["a"])])


def test_excel_mixed_type_columns(tmp_path):
    import pyarrow as pa

    raw = pd.DataFrame(
        {
            "product": [101, 102, "ABL - General", 101, None],
            "amount": [1.0, 2.5, 3.0, 4.0, 5.0],
            "when": [pd.Timestamp("2024-01-31"), "2024-02-01", None, "2024-03-01", "2024-04-01"],
            "dup": ["a", "b", "c", "d", "e"],
            "dup ": ["v", "w", "x", "y", "z"],
        }
    )
    path = tmp_path / "mixed.xlsx"
    raw.to_excel(path, index=False)
    df = cf.load_table(path)
    assert list(df.columns) == ["product", "amount", "when", "dup", "dup_2"]
    assert df["product"].dropna().tolist() == ["101", "102", "ABL - General", "101"]
    assert df["amount"].dtype == "float64"
    pa.Table.from_pandas(df)  # used to raise ArrowInvalid: could not convert 'ABL - General' ... to int64
    cleaned = cf.clean_frame(df)
    assert cleaned["product"].dtype == "string" and cleaned["product"].isna().sum() == 1


def test_value_rules_required_features_and_override():
    from category_fill.analysis import value_rules

    raw = make_loans(n=4000, seed=3)
    df = cf.clean_frame(raw)
    rules = value_rules(df, "industry_sector", [c for c in df.columns if c != "industry_sector"])
    flag = rules[rules["feature"] == "farm_program_flag"]
    assert flag[["value", "class"]].values.tolist() == [["Y", "Agriculture"]]
    assert rules["value"].map(type).eq(str).all()

    cfg = cf.TrainConfig(n_estimators=150, required_features=("farm_program_flag",), apply_value_rules=True)
    r = cf.train_and_fill(df, "industry_sector", ["collateral_type", "property_type"], cfg, value_rules=flag)
    assert "farm_program_flag" in r.features
    assert r.rule_check["model_agrees_on_test"].iloc[0] > 0.95
    blank_y = df.loc[df["industry_sector"].isna() & (df["farm_program_flag"] == "Y")].index
    preds = r.predictions.set_index("row").loc[blank_y]
    assert (preds["predicted"] == "Agriculture").all() and preds["set_by"].str.startswith("rule").all()

    with pytest.raises(ValueError):
        cf.train_and_fill(df, "industry_sector", ["collateral_type"], cf.TrainConfig(required_features=("nope",)))


def test_surrogate_tree_exports(loans):
    from category_fill import explain

    _, df = loans
    r = cf.train_and_fill(df, "industry_sector", ["collateral_type", "property_type", "loan_type", "loan_amount"],
                          cf.TrainConfig(n_estimators=100))
    s = explain.fit_surrogate(df, r, max_depth=3)
    assert 0 < s.fidelity <= 1 and s.tree.get_depth() <= 3
    assert s.tree.tree_.n_node_samples[0] == len(df)  # refit on every loan
    focused = explain.fit_surrogate(df, r, max_depth=3, focus="Real Estate")
    assert set(focused.class_names) <= {"Real Estate", explain.OTHER} and focused.fidelity > 0.8
    fig = explain.tree_figure(focused)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any(t.startswith("→ ") for t in texts) and {"No", "Yes"} <= {t.strip() for t in texts}
    assert explain.figure_bytes(fig, "png")[:4] == b"\x89PNG"
    assert b"<svg" in explain.figure_bytes(fig, "svg")
    assert explain.figure_bytes(fig, "pdf")[:4] == b"%PDF"
    assert "THEN" in explain.rules_text(focused) and len(explain.rules_frame(focused)) == focused.tree.get_n_leaves()
