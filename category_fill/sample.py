"""Synthetic commercial-loan data for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = {
    # sector: (naics prefix, likely collateral, likely property type)
    "Real Estate": ("53", ["Commercial Real Estate", "Multifamily"], ["Office", "Retail", "Multifamily", "Industrial"]),
    "Construction": ("23", ["Equipment", "Commercial Real Estate"], ["Land", "Industrial"]),
    "Manufacturing": ("31", ["Equipment", "Inventory", "Accounts Receivable"], ["Industrial"]),
    "Retail Trade": ("44", ["Inventory", "Accounts Receivable"], ["Retail"]),
    "Healthcare": ("62", ["Accounts Receivable", "Equipment"], ["Office", "Medical"]),
    "Hospitality": ("72", ["Commercial Real Estate", "Equipment"], ["Hotel", "Retail"]),
    "Agriculture": ("11", ["Equipment", "Land", "Inventory"], ["Land", "Farm"]),
    "Professional Services": ("54", ["Accounts Receivable", "Unsecured"], ["Office"]),
    "Transportation": ("48", ["Vehicles", "Equipment"], ["Industrial"]),
    "Wholesale Trade": ("42", ["Inventory", "Accounts Receivable"], ["Industrial"]),
}
WEIGHTS = np.array([0.22, 0.09, 0.12, 0.10, 0.09, 0.06, 0.04, 0.14, 0.07, 0.07])
STATES = ["TX", "CA", "FL", "NY", "IL", "GA", "OH", "NC", "AZ", "CO", "WA", "PA"]


def make_loans(n: int = 20000, missing_rate: float = 0.15, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(SECTORS)
    sector = rng.choice(names, size=n, p=WEIGHTS / WEIGHTS.sum())

    collateral, prop, naics, amount, term, loan_type = [], [], [], [], [], []
    all_coll = sorted({c for v in SECTORS.values() for c in v[1]})
    all_prop = sorted({p for v in SECTORS.values() for p in v[2]})
    for s in sector:
        prefix, colls, props = SECTORS[s]
        collateral.append(rng.choice(colls) if rng.random() < 0.75 else rng.choice(all_coll))
        prop.append(rng.choice(props) if rng.random() < 0.65 else rng.choice(all_prop))
        # NAICS is only populated for ~60% and is sometimes miscoded
        code = prefix if rng.random() < 0.85 else SECTORS[rng.choice(names)][0]
        naics.append(f"{code}{rng.integers(1000, 9999)}" if rng.random() < 0.6 else "")
        base = {"Real Estate": 3.5e6, "Construction": 1.8e6, "Hospitality": 4e6, "Agriculture": 9e5}.get(s, 1.2e6)
        amount.append(round(rng.lognormal(np.log(base), 0.8), -3))
        term.append(int(rng.choice([60, 84, 120, 240, 300] if s in ("Real Estate", "Hospitality") else [12, 36, 60, 84])))
        loan_type.append(
            rng.choice(["Term Loan", "CRE Mortgage", "Construction Loan"], p=[0.2, 0.7, 0.1])
            if s == "Real Estate"
            else rng.choice(["Term Loan", "Line of Credit", "Equipment Finance", "SBA 7(a)"], p=[0.4, 0.3, 0.2, 0.1])
        )

    df = pd.DataFrame(
        {
            "loan_id": [f"L{100000 + i}" for i in range(n)],
            "origination_date": pd.to_datetime("2015-01-01")
            + pd.to_timedelta(rng.integers(0, 3650, size=n), unit="D"),
            "branch": rng.choice([f"Branch {c}" for c in "ABCDEFGHIJ"], size=n),
            "state": rng.choice(STATES, size=n),
            "loan_type": loan_type,
            "collateral_type": collateral,
            "property_type": prop,
            "naics_code": naics,
            "borrower_entity": rng.choice(["LLC", "Corporation", "Partnership", "Sole Proprietor"], size=n),
            "loan_amount": [f"${a:,.0f}" for a in amount],
            "interest_rate": np.round(rng.normal(6.5, 1.2, size=n), 2),
            "term_months": term,
            "risk_rating": rng.integers(1, 9, size=n),
            "industry_sector": sector,
        }
    )
    df["origination_date"] = df["origination_date"].dt.strftime("%m/%d/%Y")
    # A yes/no flag that maps one-to-one to a sector when set (but is only set on some of those loans).
    df.insert(
        len(df.columns) - 1,
        "farm_program_flag",
        np.where((df["industry_sector"] == "Agriculture") & (rng.random(n) < 0.6), "Y", "N"),
    )
    # A flag that points to a *family* of sectors, not one sector (like a BCC indicator that always means
    # one of the BCC risk-rating templates, without saying which).
    family = df["industry_sector"].isin(["Real Estate", "Hospitality", "Construction"])
    df.insert(len(df.columns) - 1, "cre_secured_flag", np.where(family & (rng.random(n) < 0.7), "Y", "N"))

    # Missingness is not perfectly random: Branch J rarely fills the field.
    p_miss = np.where(df["branch"] == "Branch J", missing_rate * 3, missing_rate * 0.8)
    df.loc[rng.random(n) < p_miss, "industry_sector"] = ""
    return df


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_loans.csv"
    make_loans().to_csv(out, index=False)
    print(f"wrote {out}")
