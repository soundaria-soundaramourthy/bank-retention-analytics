"""
analysis.py  -  run:  python analysis.py
Produces every number and chart you need for the research paper:
  - prints validation checks, KPIs and segment tables
  - saves charts as PNGs into ./charts
  - writes findings.txt and the at-risk customer list (at_risk_premium_customers.csv)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import retention as r

os.makedirs("charts", exist_ok=True)
BLUE, RED, GREY = "#1F5F8B", "#C0392B", "#8A97A3"

raw = r.load_data("European_Bank.csv")
df = r.add_features(raw)
lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def bar(table, x, title, fname, color=BLUE, highlight=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [RED if highlight and str(v) in highlight else color for v in table[x]]
    bars = ax.bar(table[x].astype(str), table["churn_rate"], color=colors)
    ax.axhline(df["Exited"].mean() * 100, ls="--", color=GREY, lw=1, label="Overall churn")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"charts/{fname}.png", dpi=160)
    plt.close(fig)


# 1. Validation -------------------------------------------------------------
log("=== 1. DATA VALIDATION ===")
for k, v in r.validate(raw).items():
    log(f"{k}: {v}")

# 2. KPIs -------------------------------------------------------------------
log("\n=== 2. KPIs ===")
k = r.kpis(df)
log(f"Overall churn rate:               {k['Overall churn rate']:.1%}")
log(f"Engagement Retention Ratio:       {k['Engagement Retention Ratio']:.2f}x (inactive vs active churn)")
log(f"Product Depth Index:              {k['Product Depth Index']:.2f}x (1 product vs 2+ products churn)")
log(f"High-Balance Disengagement Rate:  {k['High-Balance Disengagement Rate']:.1%}")
log(f"Credit Card Stickiness Score:     {k['Credit Card Stickiness Score']:.2f} pp")
log(f"Avg Relationship Strength Index:  {k['Avg Relationship Strength Index']:.1f} / 100")
log(f"Deposits lost to churn:           EUR {k['Deposits lost to churn']:,.0f} "
    f"({k['Share of deposits lost']:.1%} of all balances)")

# 3. Segment tables + charts --------------------------------------------------
sections = [
    ("IsActiveMember", "Churn: inactive (0) vs active (1)", "01_activity", None),
    ("ProductGroup", "Churn by number of products", "02_products", ["3+ products", "1 product"]),
    ("EngagementProfile", "Churn by engagement profile", "03_profiles", ["Inactive, high-balance"]),
    ("HasCrCard", "Churn: no credit card (0) vs card (1)", "04_credit_card", None),
    ("Mismatch", "Churn by salary-balance pattern", "05_mismatch", None),
    ("RSIBand", "Churn by Relationship Strength Index band", "06_rsi", ["Weak"]),
    ("Sticky", "Churn: sticky customers (1) vs others (0)", "07_sticky", None),
    ("Geography", "Churn by country (context)", "08_geography", ["Germany"]),
]
for col, title, fname, hl in sections:
    t = r.churn_by(df, col)
    log(f"\n=== {title} ===")
    log(t.to_string(index=False))
    bar(t, col, title, fname, highlight=hl)

# Balance vs activity heat table
log("\n=== Balance quartile x activity (churn %) ===")
df["BalanceBand"] = r.pd.cut(df["Balance"], [-1, 0, 100000, 130000, 1e9],
                             labels=["Zero", "Up to 100k", "100k-130k", "130k+"])
pivot = (df.pivot_table(index="BalanceBand", columns="IsActiveMember", values="Exited",
                        aggfunc="mean", observed=True) * 100).round(1)
pivot.columns = ["Inactive", "Active"]
log(pivot.to_string())

# Engagement threshold: churn by RSI score in steps of 10
log("\n=== Churn by RSI score (threshold search) ===")
df["RSIStep"] = (df["RSI"] // 10 * 10).astype(int)
log(r.churn_by(df, "RSIStep")[["RSIStep", "customers", "churn_rate"]].to_string(index=False))

# 4. At-risk list -------------------------------------------------------------
risk = r.at_risk_premium(df)
risk.to_csv("at_risk_premium_customers.csv", index=False)
log(f"\nAt-risk premium customers still with the bank: {len(risk):,} "
    f"holding EUR {risk['Balance'].sum():,.0f} in balances -> saved to at_risk_premium_customers.csv")

with open("findings.txt", "w") as f:
    f.write("\n".join(lines))
print("\nDone. Charts in ./charts, summary in findings.txt")
