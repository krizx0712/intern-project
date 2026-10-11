"""
Sales Analytics Dashboard
-------------------------
A Streamlit data science project:
  * Data loading (built-in sample data OR your own CSV upload)
  * Interactive filters + KPIs
  * EDA charts (trend, category, region, heatmap)
  * Sales forecasting (Linear Regression with seasonality features)
  * Customer segmentation (RFM + KMeans)
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")

REQUIRED_COLS = ["order_date", "customer_id", "category", "region", "quantity", "unit_price"]


# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
@st.cache_data
def generate_sample_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic sales dataset (2 years)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")

    # More orders in Oct-Dec (festival season) and a slow upward trend
    day_weights = np.array(
        [1.0 + 0.6 * (d.month in (10, 11, 12)) + 0.0008 * (d - dates[0]).days for d in dates]
    )
    day_weights /= day_weights.sum()

    categories = {
        "Electronics": (8000, 40000),
        "Clothing": (500, 5000),
        "Groceries": (100, 1500),
        "Home & Kitchen": (800, 12000),
        "Books": (150, 900),
    }
    regions = ["North", "South", "East", "West"]

    cat = rng.choice(list(categories), size=n_rows, p=[0.2, 0.25, 0.3, 0.15, 0.1])
    price = np.array([rng.uniform(*categories[c]) for c in cat]).round(2)

    df = pd.DataFrame(
        {
            "order_date": rng.choice(dates, size=n_rows, p=day_weights),
            "customer_id": rng.integers(1000, 1800, size=n_rows),
            "category": cat,
            "region": rng.choice(regions, size=n_rows, p=[0.22, 0.34, 0.2, 0.24]),
            "quantity": rng.integers(1, 6, size=n_rows),
            "unit_price": price,
        }
    )
    return df.sort_values("order_date").reset_index(drop=True)


@st.cache_data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data and add derived columns."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["order_date", "quantity", "unit_price"])
    df = df.drop_duplicates()
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = df["order_date"].dt.day_name()
    return df


# ----------------------------------------------------------------------------
# 2. ANALYSIS HELPERS
# ----------------------------------------------------------------------------
def forecast_revenue(monthly: pd.DataFrame, horizon: int):
    """Linear regression on time index + month-of-year dummies."""
    m = monthly.copy().reset_index(drop=True)
    m["t"] = np.arange(len(m))
    m["moy"] = m["month"].dt.month

    X = pd.concat([m[["t"]], pd.get_dummies(m["moy"], prefix="m", dtype=float)], axis=1)
    y = m["revenue"]

    model = LinearRegression().fit(X, y)
    fitted = model.predict(X)
    metrics = {"R²": r2_score(y, fitted), "MAE": mean_absolute_error(y, fitted)}

    future_months = pd.date_range(m["month"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    fut = pd.DataFrame({"month": future_months})
    fut["t"] = np.arange(len(m), len(m) + horizon)
    fut["moy"] = fut["month"].dt.month
    Xf = pd.concat([fut[["t"]], pd.get_dummies(fut["moy"], prefix="m", dtype=float)], axis=1)
    Xf = Xf.reindex(columns=X.columns, fill_value=0.0)
    fut["revenue"] = model.predict(Xf).clip(min=0)
    return fut[["month", "revenue"]], metrics


def rfm_segments(df: pd.DataFrame, k: int):
    """RFM features + KMeans clustering."""
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("revenue", "sum"),
    )
    if len(rfm) < k:
        return None
    scaled = StandardScaler().fit_transform(np.log1p(rfm))
    rfm["segment"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)

    # Name segments by average monetary value (high -> low)
    order = rfm.groupby("segment")["monetary"].mean().sort_values(ascending=False).index
    labels = ["🏆 Champions", "💎 Loyal", "🌱 Potential", "😴 At Risk", "❄️ Lost"]
    mapping = {seg: labels[i] if i < len(labels) else f"Segment {i+1}" for i, seg in enumerate(order)}
    rfm["segment"] = rfm["segment"].map(mapping)
    return rfm.reset_index()


def read_uploaded_csv(file) -> pd.DataFrame:
    """Read a CSV; fall back to latin-1 if it is not valid UTF-8."""
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")


# ----------------------------------------------------------------------------
# 3. SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
upload = st.sidebar.file_uploader("Upload your CSV (optional)", type="csv")
st.sidebar.caption("Required columns: " + ", ".join(REQUIRED_COLS))

if upload is not None:
    raw = read_uploaded_csv(upload)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Your CSV is missing columns: {missing}")
        st.stop()
    source = "Uploaded data"
else:
    raw = generate_sample_data()
    source = "Sample data (synthetic)"

# Currency: sample data is in rupees, uploaded Superstore data is in dollars.
# The selector converts the values (not just the symbol) using the rate you enter.
NATIVE = "$" if upload is not None else "₹"
CURRENCY = st.sidebar.selectbox(
    "Show values in",
    ["$", "₹"],
    index=0 if NATIVE == "$" else 1,
    key=f"currency_{NATIVE}",  # separate state for sample vs uploaded data
)

factor = 1.0
if CURRENCY != NATIVE:
    rate = st.sidebar.number_input(
        "Exchange rate: 1 USD = ? INR",
        min_value=1.0,
        value=85.0,
        step=0.5,
        help="Enter the current USD to INR rate. This is not fetched automatically.",
    )
    factor = rate if NATIVE == "$" else 1.0 / rate


def fmt_money(x: float) -> str:
    return f"{CURRENCY}{x:,.0f}"


data = prepare(raw).copy()
data["revenue"] = data["revenue"] * factor
data["unit_price"] = data["unit_price"] * factor

min_d, max_d = data["order_date"].min().date(), data["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
cats = st.sidebar.multiselect("Category", sorted(data["category"].unique()), default=sorted(data["category"].unique()))
regs = st.sidebar.multiselect("Region", sorted(data["region"].unique()), default=sorted(data["region"].unique()))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = pd.to_datetime(min_d), pd.to_datetime(max_d)

df = data[
    data["order_date"].between(start, end) & data["category"].isin(cats) & data["region"].isin(regs)
]

# ----------------------------------------------------------------------------
# 4. MAIN PAGE
# ----------------------------------------------------------------------------
st.title("📊 Sales Analytics Dashboard")
st.caption(f"Data source: **{source}** · {len(df):,} orders after filters")

if df.empty:
    st.warning("No data for the selected filters. Change the filters in the sidebar.")
    st.stop()

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", fmt_money(df["revenue"].sum()))
k2.metric("Orders", f"{len(df):,}")
k3.metric("Avg Order Value", fmt_money(df["revenue"].mean()))
k4.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "🛍️ Products & Regions", "🔮 Forecast", "👥 Customer Segments", "🗂️ Data"]
)

# --- Overview ---------------------------------------------------------------
with tab1:
    monthly = df.groupby("month", as_index=False)["revenue"].sum()
    fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Revenue Trend")
    fig.update_layout(yaxis_title=f"Revenue ({CURRENCY})", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        wd_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        wd = df.groupby("weekday", as_index=False)["revenue"].sum()
        wd["weekday"] = pd.Categorical(wd["weekday"], wd_order, ordered=True)
        st.plotly_chart(
            px.bar(wd.sort_values("weekday"), x="weekday", y="revenue", title="Revenue by Weekday"),
            width="stretch",
        )
    with c2:
        heat = df.assign(year=df["order_date"].dt.year, mon=df["order_date"].dt.month).pivot_table(
            index="year", columns="mon", values="revenue", aggfunc="sum"
        )
        # Use text labels so the axes show 2024, 2025 and 1-12 (not 2,024.5)
        heat.index = heat.index.astype(str)
        heat.columns = heat.columns.astype(str)
        st.plotly_chart(
            px.imshow(
                heat,
                aspect="auto",
                color_continuous_scale="Blues",
                labels={"x": "Month", "y": "Year", "color": "Revenue"},
                title="Revenue Heatmap (Year × Month)",
            ),
            width="stretch",
        )

# --- Products & Regions -----------------------------------------------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        by_cat = df.groupby("category", as_index=False)["revenue"].sum()
        st.plotly_chart(
            px.pie(by_cat, names="category", values="revenue", hole=0.45, title="Revenue Share by Category"),
            width="stretch",
        )
    with c2:
        by_reg = df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.plotly_chart(
            px.bar(by_reg, x="region", y="revenue", color="region", title="Revenue by Region"),
            width="stretch",
        )

    cr = df.groupby(["month", "category"], as_index=False)["revenue"].sum()
    st.plotly_chart(
        px.area(cr, x="month", y="revenue", color="category", title="Category Revenue Over Time"),
        width="stretch",
    )

# --- Forecast ---------------------------------------------------------------
with tab3:
    st.subheader("Revenue Forecast")
    horizon = st.slider("Months to forecast", 1, 12, 6)
    monthly = df.groupby("month", as_index=False)["revenue"].sum()
    if len(monthly) < 6:
        st.info("Need at least 6 months of data to forecast. Widen the date range.")
    else:
        fut, metrics = forecast_revenue(monthly, horizon)
        hist = monthly.assign(type="Actual")
        pred = fut.assign(type="Forecast")
        fig = px.line(
            pd.concat([hist, pred]),
            x="month",
            y="revenue",
            color="type",
            markers=True,
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
        )
        fig.update_layout(yaxis_title=f"Revenue ({CURRENCY})", xaxis_title="")
        st.plotly_chart(fig, width="stretch")

        m1, m2, m3 = st.columns(3)
        m1.metric("Model R² (fit)", f"{metrics['R²']:.2f}")
        m2.metric("MAE", fmt_money(metrics["MAE"]))
        m3.metric(f"Forecast next {horizon}m", fmt_money(fut["revenue"].sum()))
        st.caption(
            "Model: Linear Regression on a time trend + month-of-year seasonality. "
            "R² here is in-sample, so treat the forecast as a rough guide."
        )

# --- Segments ---------------------------------------------------------------
with tab4:
    st.subheader("Customer Segmentation (RFM + KMeans)")
    k = st.slider("Number of segments", 2, 5, 4)
    rfm = rfm_segments(df, k)
    if rfm is None:
        st.info("Not enough customers for this many segments.")
    else:
        summary = (
            rfm.groupby("segment")
            .agg(
                customers=("customer_id", "count"),
                avg_recency_days=("recency", "mean"),
                avg_orders=("frequency", "mean"),
                avg_spend=("monetary", "mean"),
            )
            .round(1)
            .sort_values("avg_spend", ascending=False)
        )
        st.dataframe(summary, width="stretch")

        st.plotly_chart(
            px.scatter(
                rfm,
                x="frequency",
                y="monetary",
                color="segment",
                size="monetary",
                hover_data=["customer_id", "recency"],
                log_y=True,
                title="Customers: Frequency vs Spend",
            ),
            width="stretch",
        )
        st.download_button("⬇️ Download segments CSV", rfm.to_csv(index=False), "customer_segments.csv", "text/csv")

# --- Data -------------------------------------------------------------------
with tab5:
    st.dataframe(df.drop(columns=["month"]).head(1000), width="stretch")
    st.download_button("⬇️ Download filtered data", df.to_csv(index=False), "filtered_sales.csv", "text/csv")