"""
retention.py
Shared logic for the Customer Engagement & Product Utilization project.
Both analysis.py (for the research paper) and app.py (Streamlit) import from here,
so the numbers in your paper and your dashboard always match.
"""
import pandas as pd

BINARY_COLS = ["HasCrCard", "IsActiveMember", "Exited"]


# ---------------------------------------------------------------- 1. Load & validate
def load_data(path="European_Bank.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset="CustomerId")
    return df


def validate(df):
    """Returns a dict of checks for the 'Data Ingestion & Validation' step."""
    checks = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_customer_ids": int(df["CustomerId"].duplicated().sum()),
    }
    for c in BINARY_COLS:
        checks[f"{c}_is_binary"] = set(df[c].unique()) <= {0, 1}
    checks["products_in_1_to_4"] = df["NumOfProducts"].between(1, 4).all()
    checks["no_negative_balance"] = (df["Balance"] >= 0).all()
    checks["churn_rate"] = round(df["Exited"].mean(), 4)
    return checks


# ---------------------------------------------------------------- 2. Feature engineering
def add_features(df, high_balance_q=0.75, high_salary_q=0.75):
    df = df.copy()
    bal_cut = df["Balance"].quantile(high_balance_q)
    sal_cut = df["EstimatedSalary"].quantile(high_salary_q)

    df["HighBalance"] = (df["Balance"] > bal_cut).astype(int)
    df["HighSalary"] = (df["EstimatedSalary"] > sal_cut).astype(int)
    df["ProductGroup"] = df["NumOfProducts"].map(
        {1: "1 product", 2: "2 products", 3: "3+ products", 4: "3+ products"}
    )
    df["MultiProduct"] = (df["NumOfProducts"] >= 2).astype(int)

    # Engagement profiles (the four required by the brief)
    def profile(r):
        if r["IsActiveMember"] == 1:
            return "Active engaged" if r["NumOfProducts"] >= 2 else "Active, low-product"
        return "Inactive, high-balance" if r["HighBalance"] == 1 else "Inactive, disengaged"

    df["EngagementProfile"] = df.apply(profile, axis=1)

    # Salary-balance mismatch: how much of a year's salary is kept at the bank
    df["BalanceToSalary"] = df["Balance"] / df["EstimatedSalary"].clip(lower=1)
    df["Mismatch"] = "Aligned"
    df.loc[(df["HighSalary"] == 1) & (df["Balance"] == 0), "Mismatch"] = "High salary, zero balance"
    df.loc[(df["HighBalance"] == 1) & (df["HighSalary"] == 0), "Mismatch"] = "High balance, modest salary"

    # At-risk premium customers: valuable (high balance OR high salary) but inactive
    df["AtRiskPremium"] = (
        ((df["HighBalance"] == 1) | (df["HighSalary"] == 1)) & (df["IsActiveMember"] == 0)
    ).astype(int)

    # Relationship Strength Index (0-100). Weights are simple and explainable:
    #   activity 40, product depth 30, tenure 20, credit card 10.
    #   2 products scores highest because the data shows 3-4 products churn heavily.
    product_pts = df["NumOfProducts"].map({1: 15, 2: 30, 3: 5, 4: 0})
    df["RSI"] = (
        df["IsActiveMember"] * 40
        + product_pts
        + (df["Tenure"] / 10).clip(upper=1) * 20
        + df["HasCrCard"] * 10
    ).round(1)
    df["RSIBand"] = pd.cut(df["RSI"], bins=[-1, 40, 70, 100], labels=["Weak", "Moderate", "Strong"])

    # "Sticky customer": active, exactly 2 products, 3+ years tenure
    df["Sticky"] = (
        (df["IsActiveMember"] == 1) & (df["NumOfProducts"] == 2) & (df["Tenure"] >= 3)
    ).astype(int)
    return df


# ---------------------------------------------------------------- 3. Summaries
def churn_by(df, col):
    out = (
        df.groupby(col, observed=True)
        .agg(customers=("Exited", "size"), churned=("Exited", "sum"), churn_rate=("Exited", "mean"),
             balance_at_risk=("Balance", lambda s: s[df.loc[s.index, "Exited"] == 1].sum()))
        .reset_index()
    )
    out["churn_rate"] = (out["churn_rate"] * 100).round(1)
    return out


def kpis(df):
    def rate(mask):
        return df.loc[mask, "Exited"].mean() if mask.any() else float("nan")

    active, inactive = df["IsActiveMember"] == 1, df["IsActiveMember"] == 0
    hb = df["HighBalance"] == 1
    card, nocard = df["HasCrCard"] == 1, df["HasCrCard"] == 0
    return {
        "Overall churn rate": rate(df["Exited"] >= 0),
        # how many times more likely an inactive customer is to leave
        "Engagement Retention Ratio": rate(inactive) / rate(active),
        # single-product churn divided by multi-product churn
        "Product Depth Index": rate(df["NumOfProducts"] == 1) / rate(df["NumOfProducts"] >= 2),
        # churn among inactive high-balance customers
        "High-Balance Disengagement Rate": rate(hb & inactive),
        # percentage points of churn saved by holding a card
        "Credit Card Stickiness Score": (rate(nocard) - rate(card)) * 100,
        "Avg Relationship Strength Index": df["RSI"].mean(),
        "Deposits lost to churn": df.loc[df["Exited"] == 1, "Balance"].sum(),
        "Share of deposits lost": df.loc[df["Exited"] == 1, "Balance"].sum() / df["Balance"].sum(),
    }


def at_risk_premium(df):
    """Still-active accounts (Exited == 0) that look like premium churners: the retention target list."""
    cols = ["CustomerId", "Surname", "Geography", "Age", "Tenure", "Balance",
            "EstimatedSalary", "NumOfProducts", "HasCrCard", "RSI"]
    return (df[(df["AtRiskPremium"] == 1) & (df["Exited"] == 0)][cols]
            .sort_values("Balance", ascending=False))
