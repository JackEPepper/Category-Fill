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
