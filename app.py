"""
app.py  -  run:  streamlit run app.py
Customer Engagement & Product Utilization dashboard.
"""
import pandas as pd
import plotly.express as px
import streamlit as st
import retention as r

st.set_page_config(page_title="Retention Analytics", page_icon="🏦", layout="wide")
BLUE, RED, GREY = "#1F5F8B", "#C0392B", "#8A97A3"


@st.cache_data
def get_data():
    return r.add_features(r.load_data("European_Bank.csv"))


full = get_data()

# ---------------------------------------------------------------- Sidebar filters
st.sidebar.header("Filters")
engagement = st.sidebar.radio("Engagement", ["All", "Active only", "Inactive only"])
prod_range = st.sidebar.slider("Number of products", 1, 4, (1, 4))
bal_min, bal_max = float(full["Balance"].min()), float(full["Balance"].max())
bal_range = st.sidebar.slider("Balance (EUR)", bal_min, bal_max, (bal_min, bal_max), step=1000.0)
sal_min, sal_max = float(full["EstimatedSalary"].min()), float(full["EstimatedSalary"].max())
sal_range = st.sidebar.slider("Estimated salary (EUR)", sal_min, sal_max, (sal_min, sal_max), step=1000.0)
countries = st.sidebar.multiselect("Country", sorted(full["Geography"].unique()),
                                   default=sorted(full["Geography"].unique()))

df = full[
    full["NumOfProducts"].between(*prod_range)
    & full["Balance"].between(*bal_range)
    & full["EstimatedSalary"].between(*sal_range)
    & full["Geography"].isin(countries)
]
if engagement == "Active only":
    df = df[df["IsActiveMember"] == 1]
elif engagement == "Inactive only":
    df = df[df["IsActiveMember"] == 0]

st.title("Customer engagement and retention")
st.caption(f"Showing {len(df):,} of {len(full):,} customers")

if df.empty:
    st.warning("No customers match these filters. Widen the balance, salary or product range in the sidebar.")
    st.stop()

k = r.kpis(df)


def fmt_ratio(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}x"


c = st.columns(4)
c[0].metric("Churn rate", f"{k['Overall churn rate']:.1%}")
c[1].metric("Deposits lost to churn", f"€{k['Deposits lost to churn'] / 1e6:,.1f}M",
            f"{k['Share of deposits lost']:.1%} of balances", delta_color="off")
c[2].metric("Engagement Retention Ratio", fmt_ratio(k["Engagement Retention Ratio"]),
            "inactive vs active churn", delta_color="off")
c[3].metric("Avg Relationship Strength", f"{k['Avg Relationship Strength Index']:.0f} / 100")


def churn_bar(table, x, title, highlight=()):
    t = table.copy()
    t[x] = t[x].astype(str)
    t["color"] = t[x].apply(lambda v: "Focus" if v in highlight else "Segment")
    fig = px.bar(t, x=x, y="churn_rate", text="churn_rate", color="color",
                 color_discrete_map={"Focus": RED, "Segment": BLUE},
                 hover_data=["customers", "churned"], title=title)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=df["Exited"].mean() * 100, line_dash="dash", line_color=GREY,
                  annotation_text="Average")
    fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)", xaxis_title=None,
                      margin=dict(t=50, b=10))
    return fig


tab1, tab2, tab3, tab4 = st.tabs([
    "Engagement vs churn", "Product utilization", "High-value disengaged", "Retention strength"])

# ---------------------------------------------------------------- Module 1
with tab1:
    a, b = st.columns(2)
    act = r.churn_by(df, "IsActiveMember")
    act["IsActiveMember"] = act["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    a.plotly_chart(churn_bar(act, "IsActiveMember", "Active vs inactive", ("Inactive",)),
                   width="stretch")
    b.plotly_chart(churn_bar(r.churn_by(df, "EngagementProfile"), "EngagementProfile",
                             "Engagement profiles", ("Inactive, high-balance",)),
                   width="stretch")
    st.dataframe(r.churn_by(df, "EngagementProfile"), width="stretch", hide_index=True)

# ---------------------------------------------------------------- Module 2
with tab2:
    m = st.columns(2)
    m[0].metric("Product Depth Index", fmt_ratio(k["Product Depth Index"]),
                "1-product vs 2+ churn", delta_color="off")
    m[1].metric("Credit Card Stickiness", f"{k['Credit Card Stickiness Score']:.2f} pp",
                "churn reduction from holding a card", delta_color="off")
    a, b = st.columns(2)
    a.plotly_chart(churn_bar(r.churn_by(df, "NumOfProducts"), "NumOfProducts",
                             "Churn by number of products", ("3", "4")), width="stretch")
    card = r.churn_by(df, "HasCrCard")
    card["HasCrCard"] = card["HasCrCard"].map({0: "No card", 1: "Has card"})
    b.plotly_chart(churn_bar(card, "HasCrCard", "Credit card ownership"), width="stretch")
    st.info("Two products is the retention sweet spot. Customers pushed to 3 or 4 products "
            "churn far more, which suggests over-selling rather than real engagement.")

# ---------------------------------------------------------------- Module 3
with tab3:
    risk = r.at_risk_premium(df)
    m = st.columns(3)
    m[0].metric("High-Balance Disengagement Rate", f"{k['High-Balance Disengagement Rate']:.1%}"
                if pd.notna(k["High-Balance Disengagement Rate"]) else "n/a")
    m[1].metric("At-risk premium customers (still here)", f"{len(risk):,}")
    m[2].metric("Balances held by them", f"€{risk['Balance'].sum() / 1e6:,.1f}M")
    sample = df.sample(min(len(df), 3000), random_state=1)
    sample = sample.assign(Status=sample["Exited"].map({0: "Stayed", 1: "Churned"}))
    fig = px.scatter(sample, x="EstimatedSalary", y="Balance", color="Status",
                     color_discrete_map={"Stayed": BLUE, "Churned": RED}, opacity=0.55,
                     title="Balance vs salary (sample of up to 3,000 customers)")
    fig.update_layout(legend_title=None)
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(churn_bar(r.churn_by(df, "Mismatch"), "Mismatch", "Salary-balance patterns"),
                    width="stretch")
    st.subheader("Retention target list")
    st.caption("Inactive customers with high balance or high salary who have not left yet.")
    st.dataframe(risk, width="stretch", hide_index=True)
    st.download_button("Download target list (CSV)", risk.to_csv(index=False),
                       "at_risk_premium_customers.csv", "text/csv")

# ---------------------------------------------------------------- Module 4
with tab4:
    st.caption("Relationship Strength Index (0-100) = activity 40 + product depth 30 "
               "+ tenure 20 + credit card 10. Two products score highest.")
    a, b = st.columns(2)
    a.plotly_chart(churn_bar(r.churn_by(df, "RSIBand"), "RSIBand", "Churn by RSI band", ("Weak",)),
                   width="stretch")
    sticky = r.churn_by(df, "Sticky")
    sticky["Sticky"] = sticky["Sticky"].map({0: "Other", 1: "Sticky"})
    b.plotly_chart(churn_bar(sticky, "Sticky", "Sticky customers (active, 2 products, 3+ yrs)"),
                   width="stretch")
    steps = df.assign(RSIStep=(df["RSI"] // 10 * 10).astype(int))
    t = r.churn_by(steps, "RSIStep")
    fig = px.line(t, x="RSIStep", y="churn_rate", markers=True,
                  title="Churn by RSI score: where retention kicks in")
    fig.update_traces(line_color=BLUE)
    fig.update_layout(xaxis_title="RSI score (bucket start)", yaxis_title="Churn rate (%)")
    st.plotly_chart(fig, width="stretch")
