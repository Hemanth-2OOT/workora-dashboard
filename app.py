import pandas as pd, plotly.express as px, streamlit as st
from math import erf, sqrt

st.set_page_config(page_title="Workora analytics dashboard", layout="wide")
st.title("Workora analytics dashboard")
st.caption("All records are synthetic and for learning only. They are not real Workora business results.")

@st.cache_data
def load():
    web = pd.read_csv("data/web_traffic.csv", parse_dates=["Date"])
    ab = pd.read_csv("data/checkout_ab_test.csv", parse_dates=["Timestamp"])
    reg = pd.read_csv("data/regional_market.csv")
    c = pd.read_csv("data/customer_churn.csv")
    log = {"raw_rows": len(c)}
    c = c.drop_duplicates(); log["dups_removed"] = log["raw_rows"] - len(c)
    log["missing_region"] = int(c.Region.isna().sum()); c["Region"] = c.Region.fillna("Unknown")
    c["TotalChargesINR"] = pd.to_numeric(c.TotalChargesINR, errors="coerce")
    log["total_charges_unknown"] = int(c.TotalChargesINR.isna().sum())
    c["Outlier"] = c.MonthlyChargesINR > 5000; log["outliers"] = int(c.Outlier.sum())
    c["Churned"] = (c.Churn == "Yes").astype(int)
    web["Registered_f"] = (web.Registered == "Yes").astype(int)
    web["Started_f"] = (web.StartedRegistration == "Yes").astype(int)
    web["Bounced_f"] = (web.Bounced == "Yes").astype(int)
    web["Month"] = web.Date.dt.to_period("M").astype(str)
    ab["Day"] = ab.Timestamp.dt.normalize(); ab["Conv"] = (ab.Converted == "Yes").astype(int)
    return web, ab, reg, c, log
web, ab, reg, cust, log = load()

st.sidebar.header("Filters")
d = st.sidebar.date_input("Date range", (web.Date.min().date(), web.Date.max().date()))
d1, d2 = (d if len(d) == 2 else (d[0], d[0]))
states = st.sidebar.multiselect("State (web traffic and regional markets)", sorted(web.Region.unique()))
plans = st.sidebar.multiselect("Subscription plan (customers)", sorted(cust.SubscriptionType.unique()))
st.sidebar.caption("Date range also filters the checkout test (1–22 Jul 2026). Customer regions are not states, so they are not joined.")

W = web[(web.Date.dt.date >= d1) & (web.Date.dt.date <= d2)]
if states: W = W[W.Region.isin(states)]
A = ab[(ab.Day.dt.date >= d1) & (ab.Day.dt.date <= d2)]
C = cust[cust.SubscriptionType.isin(plans)] if plans else cust
R = reg[reg.State.isin(states)] if states else reg

rev = A.OrderValueINR.sum(); orders = A.Conv.sum()
k = st.columns(5)
k[0].metric("Total revenue (experiment)", f"₹{rev:,.0f}")
k[1].metric("Active customers", f"{(C.Churned == 0).sum():,}")
k[2].metric("Churn %", f"{C.Churned.mean()*100:.1f}%" if len(C) else "–")
k[3].metric("Avg ticket size", f"₹{rev/orders:,.0f}" if orders else "–")
k[4].metric("Registration rate", f"{W.Registered_f.mean()*100:.1f}%" if len(W) else "–")

t1, t2, t3, t4, t5 = st.tabs(["Acquisition", "Checkout experiment", "Retention", "Regional demand", "Method"])
with t1:
    m = W.groupby("Month").agg(Sessions=("SessionID", "count"), Registrations=("Registered_f", "sum")).reset_index()
    st.plotly_chart(px.bar(m, x="Month", y=["Sessions", "Registrations"], barmode="group", title="Sessions and registrations by month"), width='stretch')
    a, b, c3 = st.columns(3)
    f = pd.DataFrame({"Stage": ["Sessions", "Started registration", "Registered"], "Count": [len(W), W.Started_f.sum(), W.Registered_f.sum()]})
    a.plotly_chart(px.funnel(f, x="Count", y="Stage", title="Registration funnel"), width='stretch')
    s = W.groupby("Source").Registered_f.mean().mul(100).reset_index()
    b.plotly_chart(px.bar(s, x="Source", y="Registered_f", labels={"Registered_f": "Registration %"}, title="Registration rate by source"), width='stretch')
    dv = W.groupby("DeviceCategory").Bounced_f.mean().mul(100).reset_index()
    c3.plotly_chart(px.bar(dv, x="DeviceCategory", y="Bounced_f", labels={"Bounced_f": "Bounce %"}, title="Bounce rate by device"), width='stretch')
with t2:
    g = A.groupby("Variant").Conv.agg(["sum", "count"]).reset_index(); g["Rate"] = g["sum"] / g["count"] * 100
    x, y = st.columns(2)
    x.plotly_chart(px.bar(g, x="Variant", y="Rate", labels={"Rate": "Conversion %"}, title="Conversion by variant"), width='stretch')
    gd = A.groupby(["Device", "Variant"]).Conv.mean().mul(100).reset_index()
    y.plotly_chart(px.bar(gd, x="Device", y="Conv", color="Variant", barmode="group", labels={"Conv": "Conversion %"}, title="Conversion by device and variant"), width='stretch')
    if set(g.Variant) == {"A", "B"}:
        ga, gb = g.set_index("Variant").loc["A"], g.set_index("Variant").loc["B"]
        p1, p2 = ga["sum"] / ga["count"], gb["sum"] / gb["count"]
        pp = (ga["sum"] + gb["sum"]) / (ga["count"] + gb["count"])
        z = (p2 - p1) / sqrt(pp * (1 - pp) * (1 / ga["count"] + 1 / gb["count"]))
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        st.info(f"B vs A: {(p2-p1)*100:+.1f} pts, z = {z:.2f}, p = {p:.3f} — {'statistically significant' if p < .05 else 'not statistically significant'}.")
    else: st.warning("No experiment visitors in this date range.")
with t3:
    a, b, c3 = st.columns(3)
    for col, key, ttl in [(a, "ContractType", "Churn % by contract"), (b, "Region", "Churn % by customer region")]:
        r = C.groupby(key).Churned.mean().mul(100).reset_index()
        col.plotly_chart(px.bar(r, x=key, y="Churned", labels={"Churned": "Churn %"}, title=ttl), width='stretch')
    C2 = C.assign(Tenure=pd.cut(C.TenureMonths, [-1, 12, 24, 48, 100], labels=["0–12", "13–24", "25–48", "49+"]))
    r = C2.groupby("Tenure", observed=True).Churned.mean().mul(100).reset_index()
    c3.plotly_chart(px.bar(r, x="Tenure", y="Churned", labels={"Churned": "Churn %"}, title="Churn % by tenure (months)"), width='stretch')
with t4:
    R = R.assign(ExpectedLeads=R.MonthlySearchDemand * R.LeadConversionRate)
    cty = R.groupby("City").agg(Markets=("RegionID", "count"), Lat=("Latitude", "mean"), Lon=("Longitude", "mean"),
        Demand=("MonthlySearchDemand", "sum"), ExpectedLeads=("ExpectedLeads", "sum"), LeadConv=("LeadConversionRate", "mean"),
        AcqCostINR=("EstimatedAcquisitionCostINR", "mean"), Competitors=("ExistingCompetitors", "mean")).reset_index()
    fig = px.scatter_map(cty, lat="Lat", lon="Lon", size="Demand", color="LeadConv", hover_name="City", zoom=3.6, height=520, map_style="carto-positron", title="Markets by city (size = search demand, colour = lead conversion)") if hasattr(px, "scatter_map") else px.scatter_mapbox(cty, lat="Lat", lon="Lon", size="Demand", color="LeadConv", hover_name="City", zoom=3.6, height=520, mapbox_style="carto-positron")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(cty.sort_values("ExpectedLeads", ascending=False).drop(columns=["Lat", "Lon"]).round(2), width='stretch', hide_index=True)
with t5:
    st.markdown(f"""
**Cleaning (customer_churn.csv)**
- Removed {log['dups_removed']} exact duplicate rows ({log['raw_rows']} → {log['raw_rows']-log['dups_removed']}).
- {log['missing_region']} missing Region values labelled "Unknown" and kept.
- {log['total_charges_unknown']} TotalChargesINR values read "unknown", set to null; the field is not used in any chart.
- {log['outliers']} customers with MonthlyChargesINR above ₹5,000 (median about ₹730) are flagged as outliers: kept in churn counts, excluded from charge-based figures.
- Other files had no missing values or duplicate keys; no rows excluded.

**Joins:** the four files share no common key, so there are no row-level joins. They connect through shared filters: State (web and regional markets), date range (web and experiment), plan (customers).

**Calculated fields:** Total revenue = sum of OrderValueINR in the experiment (not company revenue). Avg ticket = revenue ÷ converted orders. Churn % = churned ÷ customers. Active customers = Churn is No. Registration rate = registered ÷ sessions. Expected leads = monthly search demand × lead conversion rate. Experiment significance = two-sided two-proportion z-test.

All data is synthetic.""")
