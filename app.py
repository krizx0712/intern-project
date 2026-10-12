"""
Retail Analytics & ML Platform  (advanced version of the Sales Dashboard)
-------------------------------------------------------------------------
What's new compared to the basic version
  * Realistic synthetic data (customers with lifetimes, churn, seasonality)
  * Forecast Lab: walk-forward BACKTEST of 4 models (Seasonal Naive, Linear,
    Holt-Winters, Hybrid Trend+GradientBoosting) + prediction intervals
  * Anomaly detection (robust STL residuals)
  * Customers: RFM + KMeans with silhouette-based k, cohort retention heatmap
  * Churn model: Logistic Regression baseline vs HistGradientBoosting,
    ROC-AUC / PR-AUC, permutation importance, revenue-at-risk table
  * AI Assistant (optional, needs Anthropic API key):
        - executive insight summary from computed facts
        - "Ask your data": LLM returns a JSON query SPEC (never code), which is
          validated against an allow-list and run by this app -> safe
Run:  streamlit run app.py
"""

import hmac
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Retail Analytics & ML Platform", page_icon="📊", layout="wide")

REQUIRED_COLS = ["order_date", "customer_id", "category", "region", "quantity", "unit_price"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LLM_MODEL = "claude-sonnet-5-5"


# ============================================================================
# 1. DATA
# ============================================================================
@st.cache_data
def generate_sample_data(n_customers: int = 900, seed: int = 42) -> pd.DataFrame:
    """Synthetic retail data where every customer has a start date, a lifetime
    (so some churn), an order rate, a home region and category preferences.
    This gives the churn / cohort / CLV analyses real structure to find."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")
    categories = {
        "Electronics": (8000, 40000),
        "Clothing": (500, 5000),
        "Groceries": (100, 1500),
        "Home & Kitchen": (800, 12000),
        "Books": (150, 900),
    }
    cat_names = list(categories)
    base_p = np.array([0.2, 0.25, 0.3, 0.15, 0.1])
    regions = ["North", "South", "East", "West"]
    region_p = [0.22, 0.34, 0.2, 0.24]

    frames = []
    for cid in range(1000, 1000 + n_customers):
        first = start + pd.Timedelta(days=int(rng.integers(0, 600)))
        last = min(first + pd.Timedelta(days=int(rng.exponential(450)) + 30), end)
        span = max((last - first).days, 1)
        n = 1 + rng.poisson(rng.gamma(2.0, 0.7) * span / 30)
        offsets = np.concatenate([[0], rng.integers(0, span + 1, size=n - 1)])
        dates = first + pd.to_timedelta(offsets, unit="D")
        # seasonality: Oct-Dec orders are kept more often
        keep = rng.random(len(dates)) < np.where(dates.month.isin([10, 11, 12]), 1.0, 0.65)
        keep[0] = True
        dates = dates[keep]
        p = 0.5 * base_p + 0.5 * rng.dirichlet(np.ones(5) * 1.5)
        cats = rng.choice(cat_names, size=len(dates), p=p / p.sum())
        price = np.array([rng.uniform(*categories[c]) for c in cats]).round(2)
        frames.append(
            pd.DataFrame(
                {
                    "order_date": dates,
                    "customer_id": cid,
                    "category": cats,
                    "region": rng.choice(regions, p=region_p),
                    "quantity": rng.integers(1, 6, size=len(dates)),
                    "unit_price": price,
                }
            )
        )
    return pd.concat(frames).sort_values("order_date").reset_index(drop=True)


@st.cache_data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data and add derived columns."""
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id", "quantity", "unit_price"])
    df = df[(df["quantity"] > 0) & (df["unit_price"] > 0)].drop_duplicates()
    df["revenue"] = df["quantity"] * df["unit_price"]
    df["day"] = df["order_date"].dt.normalize()
    df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = pd.Categorical(df["order_date"].dt.day_name(), WEEKDAYS, ordered=True)
    return df


def fmt_inr(x: float) -> str:
    return f"₹{x:,.0f}"


# ============================================================================
# 2. FORECASTING  (models + walk-forward backtest)
# ============================================================================
def _future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def fc_seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    """Baseline: same month last year (or last value if < 12 months)."""
    v = y.values
    if len(v) >= 12:
        return np.array([v[len(v) - 12 + (i % 12)] for i in range(h)])
    return np.repeat(v[-1], h)


def fc_linear(y: pd.Series, h: int) -> np.ndarray:
    """Linear trend + month-of-year dummies."""

    def design(index: pd.DatetimeIndex, t0: int) -> np.ndarray:
        X = np.zeros((len(index), 13))
        X[:, 0] = np.arange(t0, t0 + len(index))
        X[np.arange(len(index)), index.month.values] = 1.0  # columns 1..12
        return X

    model = LinearRegression(fit_intercept=False).fit(design(y.index, 0), y.values)
    return np.clip(model.predict(design(_future_index(y, h), len(y))), 0, None)


def fc_holt_winters(y: pd.Series, h: int) -> np.ndarray:
    """Exponential smoothing. Seasonal only if we have >= 24 months."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(y) >= 24:
                m = ExponentialSmoothing(
                    y.values, trend="add", seasonal="add", seasonal_periods=12,
                    initialization_method="estimated",
                ).fit()
            else:
                m = ExponentialSmoothing(
                    y.values, trend="add", damped_trend=True, initialization_method="estimated"
                ).fit()
            return np.clip(np.asarray(m.forecast(h)), 0, None)
    except Exception:
        return fc_seasonal_naive(y, h)


def fc_hybrid_gbm(y: pd.Series, h: int) -> np.ndarray:
    """Hybrid: linear trend + GradientBoosting on the residuals (lags + month)."""
    n = len(y)
    if n < 10:
        return fc_seasonal_naive(y, h)
    t = np.arange(n).reshape(-1, 1)
    trend = LinearRegression().fit(t, y.values)
    resid = list(y.values - trend.predict(t))
    months = y.index.month.values

    def feats(hist, month):
        return [hist[-1], hist[-2], hist[-3], np.mean(hist[-3:]),
                np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]

    X = [feats(resid[:i], months[i]) for i in range(3, n)]
    T = [resid[i] for i in range(3, n)]
    gbr = GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
    ).fit(X, T)

    hist, out = resid[:], []
    for k, month in enumerate(_future_index(y, h).month):
        r = gbr.predict([feats(hist, month)])[0]
        hist.append(r)
        out.append(trend.predict([[n + k]])[0] + r)
    return np.clip(np.array(out), 0, None)


MODELS = {
    "Seasonal Naive (baseline)": fc_seasonal_naive,
    "Linear Trend + Seasonality": fc_linear,
    "Holt-Winters": fc_holt_winters,
    "Hybrid Trend + GBM": fc_hybrid_gbm,
}


@st.cache_data
def backtest(monthly: pd.DataFrame, min_train: int = 12) -> pd.DataFrame:
    """Walk-forward (expanding window), 1-step-ahead. Every prediction is made
    using ONLY data that was available before that month -> no leakage."""
    y = monthly.set_index("month")["revenue"]
    rows = []
    for t in range(min_train, len(y)):
        train, actual = y.iloc[:t], y.iloc[t]
        for name, fn in MODELS.items():
            rows.append({"month": y.index[t], "model": name, "actual": actual,
                         "pred": float(fn(train, 1)[0])})
    return pd.DataFrame(rows)


def score_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    bt = bt.assign(err=bt["actual"] - bt["pred"])
    g = bt.groupby("model")
    out = pd.DataFrame(
        {
            "MAE": g["err"].apply(lambda e: e.abs().mean()),
            "RMSE": g["err"].apply(lambda e: np.sqrt((e**2).mean())),
            "MAPE %": g.apply(lambda d: (d["err"].abs() / d["actual"]).mean() * 100, include_groups=False),
        }
    ).sort_values("MAE")
    base = out.loc["Seasonal Naive (baseline)", "MAE"]
    out["vs baseline"] = ((out["MAE"] / base - 1) * 100).map(lambda v: f"{v:+.1f}%")
    return out.round(1)


# ============================================================================
# 3. ANOMALIES, SEGMENTS, COHORTS
# ============================================================================
@st.cache_data
def detect_anomalies(df: pd.DataFrame, z_thresh: float = 3.0):
    """Robust STL (weekly seasonality) -> flag days with extreme residuals."""
    daily = df.groupby("day")["revenue"].sum().asfreq("D", fill_value=0.0)
    if len(daily) < 28:
        return None
    res = STL(daily, period=7, robust=True).fit()
    resid = res.resid
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    scale = mad if mad > 0 else resid.std() + 1e-9
    out = pd.DataFrame({"revenue": daily, "trend": res.trend,
                        "z": (resid - np.median(resid)) / scale})
    out["anomaly"] = out["z"].abs() > z_thresh
    return out


@st.cache_data
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snap = df["order_date"].max() + pd.Timedelta(days=1)
    return df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snap - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("revenue", "sum"),
    )


@st.cache_data
def segment_customers(rfm: pd.DataFrame, k: int | None):
    """KMeans on log-scaled RFM. k=None -> choose k with the best silhouette."""
    X = StandardScaler().fit_transform(np.log1p(rfm[["recency", "frequency", "monetary"]]))
    scores = {}
    for kk in range(2, min(7, len(rfm))):
        scores[kk] = silhouette_score(X, KMeans(kk, n_init=10, random_state=42).fit_predict(X))
    if not scores:
        return None, {}, 0
    best_k = k or max(scores, key=scores.get)
    labels = KMeans(best_k, n_init=10, random_state=42).fit_predict(X)

    out = rfm.copy()
    out["cluster"] = labels
    # rank clusters: low recency, high frequency, high monetary = best
    z = pd.DataFrame(X, index=rfm.index, columns=["r", "f", "m"])
    value = (-z["r"] + z["f"] + z["m"]).groupby(labels).mean().sort_values(ascending=False)
    names = ["🏆 Champions", "💎 Loyal", "🌱 Potential", "⚠️ Needs Attention", "😴 At Risk", "❄️ Lost"]
    picks = np.round(np.linspace(0, len(names) - 1, best_k)).astype(int)
    mapping = {c: names[picks[i]] for i, c in enumerate(value.index)}
    out["segment"] = out["cluster"].map(mapping)
    return out.drop(columns="cluster").reset_index(), scores, best_k


@st.cache_data
def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["customer_id", "month"]].drop_duplicates()
    d = d.join(d.groupby("customer_id")["month"].min().rename("cohort"), on="customer_id")
    d["age"] = (d["month"].dt.year - d["cohort"].dt.year) * 12 + (d["month"].dt.month - d["cohort"].dt.month)
    counts = d.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    ret = counts.div(counts[0], axis=0)
    last = d["month"].max()
    for c in ret.index:  # cells beyond the end of data are unknown, not zero
        max_age = (last.year - c.year) * 12 + (last.month - c.month)
        ret.loc[c, ret.columns > max_age] = np.nan
    ret.index = ret.index.strftime("%Y-%m")
    return ret


# ============================================================================
# 4. CHURN MODEL
# ============================================================================
@st.cache_data
def build_churn_dataset(df: pd.DataFrame, window_days: int):
    """Snapshot at `cutoff`. Features use ONLY orders <= cutoff.
    Label: churned = customer placed no order in the `window_days` after cutoff."""
    cutoff = df["order_date"].max() - pd.Timedelta(days=window_days)
    hist, future = df[df["order_date"] <= cutoff], df[df["order_date"] > cutoff]
    if hist.empty:
        return None, cutoff
    g = hist.groupby("customer_id")
    span = (cutoff - hist["order_date"].min()).days
    X = pd.DataFrame(
        {
            "recency": (cutoff - g["order_date"].max()).dt.days,
            "frequency": g.size(),
            "monetary": g["revenue"].sum(),
            "avg_order_value": g["revenue"].mean(),
            "tenure_days": (cutoff - g["order_date"].min()).dt.days,
            "n_categories": g["category"].nunique(),
            "avg_gap_days": g["order_date"].apply(lambda s: s.sort_values().diff().dt.days.mean()),
        }
    )
    X["avg_gap_days"] = X["avg_gap_days"].fillna(span)
    recent = hist[hist["order_date"] > cutoff - pd.Timedelta(days=90)].groupby("customer_id").size()
    X["orders_last_90d"] = recent.reindex(X.index).fillna(0)
    X["churned"] = (~X.index.isin(future["customer_id"].unique())).astype(int)
    return X, cutoff


@st.cache_data
def train_churn(data: pd.DataFrame, window_days: int):
    X, y = data.drop(columns="churned"), data["churned"]
    if y.nunique() < 2 or y.value_counts().min() < 10:
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    models = {
        "Logistic Regression (baseline)": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=42),
    }
    rows, curves, fitted = [], [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, p)
        curves.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": name}))
        rows.append({"model": name, "ROC-AUC": roc_auc_score(yte, p),
                     "PR-AUC": average_precision_score(yte, p)})
        fitted[name] = (m, p)
    res = pd.DataFrame(rows).set_index("model").round(3).sort_values("ROC-AUC", ascending=False)
    best = res.index[0]
    m, p_best = fitted[best]

    imp = permutation_importance(m, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=42)
    importance = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean}
                              ).sort_values("importance")

    scored = X.copy()
    scored["churn_prob"] = m.predict_proba(X)[:, 1]
    # simple expected spend over the next window, from historical monthly run-rate
    monthly_rate = scored["monetary"] / ((scored["tenure_days"] + 30) / 30)
    scored["expected_spend"] = monthly_rate * window_days / 30
    scored["revenue_at_risk"] = scored["churn_prob"] * scored["expected_spend"]
    cm = confusion_matrix(yte, (p_best >= 0.5).astype(int))
    return {"metrics": res, "curves": pd.concat(curves), "importance": importance,
            "scored": scored.reset_index().sort_values("revenue_at_risk", ascending=False),
            "cm": cm, "best": best, "base_rate": float(y.mean()), "n_test": len(yte)}


# ============================================================================
# 5. AI ASSISTANT  (optional)
# ============================================================================
METRICS = {
    "revenue": ("revenue", "sum"),
    "orders": ("revenue", "count"),
    "avg_order_value": ("revenue", "mean"),
    "customers": ("customer_id", "nunique"),
    "units": ("quantity", "sum"),
}
GROUPS = ["month", "category", "region", "weekday", "none"]

SPEC_SYSTEM = f"""You translate business questions about a retail sales table into a JSON query spec.
Return ONLY one JSON object, no prose, no markdown fences. Schema:
{{"metric": one of {list(METRICS)},
 "group_by": one of {GROUPS},
 "chart": one of ["bar","line","table"],
 "sort_desc": true|false,
 "top_n": integer or null,
 "filters": {{"category": [..] or null, "region": [..] or null,
              "date_from": "YYYY-MM-DD" or null, "date_to": "YYYY-MM-DD" or null}}}}
Use "line" for group_by=month. If the question cannot be answered with this schema, return {{"error": "reason"}}."""


def get_api_key() -> str | None:
    return st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY") or None


def call_llm(system: str, user: str, max_tokens: int = 900) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key())
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def run_spec(df: pd.DataFrame, spec: dict, categories: list, regions: list):
    """Validate an LLM-produced spec against an allow-list, then execute it."""
    if "error" in spec:
        raise ValueError(spec["error"])
    metric, group = spec.get("metric"), spec.get("group_by", "none")
    if metric not in METRICS or group not in GROUPS:
        raise ValueError("Spec uses an unsupported metric or grouping.")
    f = spec.get("filters") or {}
    d = df
    if f.get("category"):
        d = d[d["category"].isin([c for c in f["category"] if c in categories])]
    if f.get("region"):
        d = d[d["region"].isin([r for r in f["region"] if r in regions])]
    if f.get("date_from"):
        d = d[d["order_date"] >= pd.to_datetime(f["date_from"])]
    if f.get("date_to"):
        d = d[d["order_date"] <= pd.to_datetime(f["date_to"])]
    if d.empty:
        raise ValueError("No rows match those filters.")
    col, agg = METRICS[metric]
    if group == "none":
        res = pd.DataFrame({metric: [d[col].agg(agg)]})
    else:
        res = d.groupby(group, observed=True)[col].agg(agg).rename(metric).reset_index()
        if group != "month" and spec.get("sort_desc", True):
            res = res.sort_values(metric, ascending=False)
        if spec.get("top_n"):
            res = res.head(int(spec["top_n"]))
    return res, group, metric, spec.get("chart", "table")


def facts_digest(df: pd.DataFrame) -> dict:
    """Numbers the LLM is allowed to talk about (it must not invent others)."""
    monthly = df.groupby("month")["revenue"].sum()
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    by_reg = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    facts = {
        "period": f"{df['order_date'].min():%Y-%m-%d} to {df['order_date'].max():%Y-%m-%d}",
        "total_revenue_inr": round(df["revenue"].sum()),
        "orders": int(len(df)),
        "avg_order_value_inr": round(df["revenue"].mean()),
        "unique_customers": int(df["customer_id"].nunique()),
        "top_category": {"name": by_cat.index[0], "share_pct": round(100 * by_cat.iloc[0] / by_cat.sum(), 1)},
        "top_region": {"name": by_reg.index[0], "share_pct": round(100 * by_reg.iloc[0] / by_reg.sum(), 1)},
        "last_3_months_revenue_inr": {f"{k:%Y-%m}": round(v) for k, v in monthly.tail(3).items()},
    }
    if len(monthly) >= 2 and monthly.iloc[-2] > 0:
        facts["last_month_vs_previous_pct"] = round(100 * (monthly.iloc[-1] / monthly.iloc[-2] - 1), 1)
    an = detect_anomalies(df)
    if an is not None:
        top = an[an["anomaly"]].reindex(an["z"].abs().sort_values(ascending=False).index).dropna().head(5)
        facts["top_anomalous_days"] = [f"{i:%Y-%m-%d} (z={r.z:.1f})" for i, r in top.iterrows()]
    return facts


# ============================================================================
# 5b. STYLING, WELCOME PAGE & LOGIN
# ============================================================================
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px; padding: 14px 18px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 10px 16px;}
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #ec4899 100%);
    color: #fff; padding: 34px 34px 28px; border-radius: 20px; margin-bottom: 18px;
    box-shadow: 0 10px 30px rgba(79,70,229,.35);
}
.hero h1 {margin: 0 0 6px; font-size: 2.2rem; line-height: 1.15; color: #fff;}
.hero p {margin: 0; font-size: 1.05rem; opacity: .92;}
.chips {margin-top: 16px;}
.chip {display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
       padding: 4px 12px; border-radius: 999px; margin: 0 6px 6px 0; font-size: .85rem;}
.fcard {background: var(--secondary-background-color); border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px; padding: 16px 18px; height: 100%; box-shadow: 0 2px 8px rgba(0,0,0,.06);}
.fcard .ic {font-size: 1.6rem;}
.fcard b {display: block; margin: 4px 0 2px;}
.fcard span {opacity: .75; font-size: .9rem;}
.banner {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #fff;
    padding: 18px 26px; border-radius: 16px; margin-bottom: 14px;
}
.banner h2 {margin: 0; color: #fff; font-size: 1.6rem;}
.banner span {opacity: .85; font-size: .95rem;}
</style>
"""

FEATURES = [
    ("🔮", "Forecast Lab", "4 models judged by walk-forward backtesting, with prediction intervals."),
    ("👥", "Customer Intelligence", "RFM segments, auto-chosen k, and cohort retention heatmaps."),
    ("⚠️", "Churn Prediction", "ML model, ROC-AUC, feature importance and revenue-at-risk list."),
    ("🤖", "AI Assistant", "Ask questions in plain English and get validated, safe answers."),
]


def _users() -> dict:
    """Demo credentials. Set st.secrets['users'] or DASH_USER / DASH_PASS env vars."""
    try:
        if "users" in st.secrets:
            return dict(st.secrets["users"])
    except Exception:
        pass
    return {os.getenv("DASH_USER", "admin"): os.getenv("DASH_PASS", "admin123")}


def _check_login(user: str, pwd: str) -> bool:
    users = _users()
    return user in users and hmac.compare_digest(str(users[user]).encode(), pwd.encode())


def welcome_page() -> None:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(
            """<div class="hero">
            <h1>📊 Retail Analytics &amp; ML Platform</h1>
            <p>Turn raw sales data into forecasts, customer segments and churn alerts, in one place.</p>
            <div class="chips"><span class="chip">Time-series forecasting</span>
            <span class="chip">RFM + KMeans</span><span class="chip">Churn ML</span>
            <span class="chip">Anomaly detection</span><span class="chip">LLM insights</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
        for row in (FEATURES[:2], FEATURES[2:]):
            cols = st.columns(2, gap="medium")
            for col, (icon, title, text) in zip(cols, row):
                col.markdown(
                    f'<div class="fcard"><div class="ic">{icon}</div><b>{title}</b><span>{text}</span></div>',
                    unsafe_allow_html=True,
                )
            st.write("")
    with right:
        with st.container(border=True):
            st.markdown("### 🔐 Sign in")
            with st.form("login_form"):
                user = st.text_input("Username", placeholder="admin")
                pwd = st.text_input("Password", type="password", placeholder="••••••••")
                ok = st.form_submit_button("Sign in", type="primary", width="stretch")
            if ok:
                if _check_login(user.strip(), pwd):
                    st.session_state["user"] = user.strip()
                    st.rerun()
                else:
                    st.error("Wrong username or password.")
            st.markdown("<div style='text-align:center;opacity:.6'>or</div>", unsafe_allow_html=True)
            if st.button("Continue as guest →", width="stretch"):
                st.session_state["user"] = "Guest"
                st.rerun()
            st.caption("Demo login: **admin / admin123** (change it with DASH_USER / DASH_PASS or "
                       "Streamlit secrets). This is a demo gate, not production security.")


st.markdown(CSS, unsafe_allow_html=True)
if not st.session_state.get("user"):
    welcome_page()
    st.stop()


# ============================================================================
# 6. SIDEBAR + FILTERS
# ============================================================================
st.sidebar.markdown(f"### 👤 {st.session_state['user']}")
if st.sidebar.button("Logout", width="stretch"):
    st.session_state.pop("user", None)
    st.rerun()
st.sidebar.divider()
st.sidebar.title("⚙️ Controls")
upload = st.sidebar.file_uploader("Upload your CSV (optional)", type="csv")
st.sidebar.caption("Required columns: " + ", ".join(REQUIRED_COLS))

if upload is not None:
    raw = pd.read_csv(upload)
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        st.error(f"Your CSV is missing columns: {missing}")
        st.stop()
    source = "Uploaded data"
else:
    raw = generate_sample_data()
    source = "Sample data (synthetic)"

data = prepare(raw)
if data.empty:
    st.error("No valid rows after cleaning (check dates, quantity > 0, unit_price > 0).")
    st.stop()

min_d, max_d = data["order_date"].min().date(), data["order_date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
all_cats, all_regs = sorted(data["category"].unique()), sorted(data["region"].unique())
cats = st.sidebar.multiselect("Category", all_cats, default=all_cats)
regs = st.sidebar.multiselect("Region", all_regs, default=all_regs)

st.sidebar.divider()
st.sidebar.text_input("Anthropic API key (optional, for AI tab)", type="password", key="api_key")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = pd.to_datetime(min_d), pd.to_datetime(max_d)

df = data[data["order_date"].between(start, end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
          & data["category"].isin(cats) & data["region"].isin(regs)]

# ============================================================================
# 7. MAIN PAGE
# ============================================================================
st.markdown(
    f'<div class="banner"><h2>📊 Retail Analytics &amp; ML Platform</h2>'
    f'<span>Data source: <b>{source}</b> · {len(df):,} orders after filters · '
    f'{df["order_date"].min():%d %b %Y} → {df["order_date"].max():%d %b %Y}</span></div>',
    unsafe_allow_html=True,
)

if df.empty:
    st.warning("No data for the selected filters. Change the filters in the sidebar.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Revenue", fmt_inr(df["revenue"].sum()))
k2.metric("Orders", f"{len(df):,}")
k3.metric("Avg Order Value", fmt_inr(df["revenue"].mean()))
k4.metric("Unique Customers", f"{df['customer_id'].nunique():,}")

tab_ov, tab_fc, tab_cu, tab_ch, tab_ai, tab_data = st.tabs(
    ["📈 Overview", "🔮 Forecast Lab", "👥 Customers", "⚠️ Churn", "🤖 AI Assistant", "🗂️ Data"]
)

# --- Overview ---------------------------------------------------------------
with tab_ov:
    monthly = df.groupby("month", as_index=False)["revenue"].sum()
    st.plotly_chart(px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Revenue Trend")
                    .update_layout(yaxis_title="Revenue (₹)", xaxis_title=""), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        wd = df.groupby("weekday", as_index=False, observed=True)["revenue"].sum()
        st.plotly_chart(px.bar(wd, x="weekday", y="revenue", title="Revenue by Weekday"), width="stretch")
    with c2:
        heat = df.assign(year=df["order_date"].dt.year, mon=df["order_date"].dt.month).pivot_table(
            index="year", columns="mon", values="revenue", aggfunc="sum")
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                                  title="Revenue Heatmap (Year × Month)"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        by_cat = df.groupby("category", as_index=False)["revenue"].sum()
        st.plotly_chart(px.pie(by_cat, names="category", values="revenue", hole=0.45,
                               title="Revenue Share by Category"), width="stretch")
    with c2:
        by_reg = df.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        st.plotly_chart(px.bar(by_reg, x="region", y="revenue", color="region",
                               title="Revenue by Region"), width="stretch")

    st.subheader("Anomaly detection")
    z_thresh = st.slider("Sensitivity (|z| threshold, lower = more alerts)", 2.0, 6.0, 3.5, 0.5)
    an = detect_anomalies(df, z_thresh)
    if an is None:
        st.info("Need at least 28 days of data for anomaly detection.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=an.index, y=an["revenue"], mode="lines", name="Daily revenue", line=dict(width=1))
        fig.add_scatter(x=an.index, y=an["trend"], mode="lines", name="STL trend", line=dict(dash="dash"))
        hit = an[an["anomaly"]]
        fig.add_scatter(x=hit.index, y=hit["revenue"], mode="markers", name="Anomaly",
                        marker=dict(size=9, color="red"))
        fig.update_layout(title=f"Daily revenue with robust STL anomalies ({len(hit)} flagged)",
                          yaxis_title="Revenue (₹)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.caption("Method: STL decomposition (weekly seasonality, robust) → median/MAD z-score on residuals.")

# --- Forecast Lab -----------------------------------------------------------
with tab_fc:
    st.subheader("Forecast Lab — models are judged on data they have NOT seen")
    monthly = df.groupby("month", as_index=False)["revenue"].sum()
    last_month_end = monthly["month"].iloc[-1] + pd.offsets.MonthEnd(0)
    incomplete = df["order_date"].max().normalize() < last_month_end
    drop_last = st.checkbox("Drop last month (looks incomplete)", value=bool(incomplete))
    if drop_last:
        monthly = monthly.iloc[:-1]

    if len(monthly) < 18:
        st.info("Need at least 18 months of data for a walk-forward backtest. Widen the date range.")
    else:
        bt = backtest(monthly)
        scores = score_backtest(bt)
        best = scores.index[0]
        st.markdown(f"**Walk-forward backtest** ({bt['month'].nunique()} monthly test points, "
                    f"expanding window, 1-step-ahead)")
        st.dataframe(scores, width="stretch")
        if best == "Seasonal Naive (baseline)":
            st.warning("No model beat the seasonal-naive baseline here. That is a valid finding: "
                       "with this little history, simple wins.")
        else:
            st.success(f"Winner: **{best}**, MAE {scores.loc[best, 'MAE']:,.0f} "
                       f"({scores.loc[best, 'vs baseline']} vs baseline).")

        piv = bt.pivot(index="month", columns="model", values="pred")
        piv["Actual"] = bt.drop_duplicates("month").set_index("month")["actual"]
        st.plotly_chart(px.line(piv.reset_index().melt("month", var_name="series", value_name="revenue"),
                                x="month", y="revenue", color="series", markers=True,
                                title="Backtest: predictions vs actual"), width="stretch")

        c1, c2 = st.columns(2)
        horizon = c1.slider("Months to forecast", 1, 12, 6)
        chosen = c2.selectbox("Model for final forecast", list(scores.index), index=0)
        y = monthly.set_index("month")["revenue"]
        fc = MODELS[chosen](y, horizon)
        e = bt[bt["model"] == chosen]
        errs = (e["actual"] - e["pred"]).values
        sigma = errs.std(ddof=1) if len(errs) > 1 else np.abs(errs).mean()
        band = 1.28 * sigma * np.sqrt(np.arange(1, horizon + 1))  # ~80% interval, widens with horizon
        fidx = _future_index(y, horizon)

        fig = go.Figure()
        fig.add_scatter(x=y.index, y=y.values, mode="lines+markers", name="Actual")
        fig.add_scatter(x=fidx, y=np.clip(fc + band, 0, None), mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fidx, y=np.clip(fc - band, 0, None), mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(255,127,14,0.2)", name="~80% interval")
        fig.add_scatter(x=fidx, y=fc, mode="lines+markers", name=f"Forecast ({chosen})",
                        line=dict(color="#ff7f0e"))
        fig.update_layout(title="Revenue forecast with prediction interval",
                          yaxis_title="Revenue (₹)", xaxis_title="")
        st.plotly_chart(fig, width="stretch")
        st.metric(f"Forecast next {horizon} months", fmt_inr(fc.sum()))
        st.caption("Interval comes from the model's own out-of-sample backtest errors. "
                   "Holt-Winters only uses seasonality once a training window has ≥ 24 months.")

# --- Customers --------------------------------------------------------------
with tab_cu:
    st.subheader("Customer segmentation (RFM + KMeans)")
    rfm = build_rfm(df)
    auto = st.checkbox("Choose number of segments automatically (best silhouette)", value=True)
    k_manual = None if auto else st.slider("Number of segments", 2, 6, 4)
    seg, sil, used_k = segment_customers(rfm, k_manual)
    if seg is None:
        st.info("Not enough customers for segmentation.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(px.bar(x=list(sil), y=list(sil.values()), title="Silhouette score by k",
                                   labels={"x": "k", "y": "silhouette"}), width="stretch")
            st.caption(f"Using k = **{used_k}**")
        with c2:
            summary = seg.groupby("segment").agg(
                customers=("customer_id", "count"), avg_recency_days=("recency", "mean"),
                avg_orders=("frequency", "mean"), avg_spend=("monetary", "mean"),
                total_revenue=("monetary", "sum"),
            ).round(1).sort_values("avg_spend", ascending=False)
            st.dataframe(summary, width="stretch")
        st.plotly_chart(px.scatter(seg, x="frequency", y="monetary", color="segment", size="monetary",
                                   hover_data=["customer_id", "recency"], log_y=True,
                                   title="Customers: Frequency vs Spend"), width="stretch")
        st.download_button("⬇️ Download segments CSV", seg.to_csv(index=False),
                           "customer_segments.csv", "text/csv")

    st.subheader("Cohort retention")
    ret = cohort_retention(df)
    if ret.shape[0] < 2:
        st.info("Need at least two monthly cohorts.")
    else:
        st.plotly_chart(px.imshow(ret, text_auto=".0%", aspect="auto", color_continuous_scale="Greens",
                                  labels=dict(x="Months since first purchase", y="Cohort", color="Retained"),
                                  title="Share of each monthly cohort that is still buying"),
                        width="stretch")
        st.caption("Cohort = month of a customer's first purchase within the filtered data.")

# --- Churn ------------------------------------------------------------------
with tab_ch:
    st.subheader("Churn prediction")
    window = st.slider("Churn window (days with no purchase = churned)", 30, 180, 90, 15)
    churn_data, cutoff = build_churn_dataset(df, window)
    if churn_data is None or len(churn_data) < 60:
        st.info("Not enough history before the cutoff. Reduce the window or widen the date range.")
    else:
        st.caption(f"Snapshot date: **{cutoff:%Y-%m-%d}** · {len(churn_data):,} customers · "
                   f"features use only orders up to the snapshot, label = no order in the next {window} days.")
        out = train_churn(churn_data, window)
        if out is None:
            st.info("Too few churners / non-churners to train a model.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Churn rate (base)", f"{out['base_rate']:.0%}")
            m2.metric("Best model", out["best"])
            m3.metric("Test ROC-AUC", f"{out['metrics'].loc[out['best'], 'ROC-AUC']:.3f}")
            st.dataframe(out["metrics"], width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(out["curves"], x="fpr", y="tpr", color="model",
                                        title="ROC curve (hold-out 25%)")
                                .add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot")),
                                width="stretch")
            with c2:
                st.plotly_chart(px.bar(out["importance"], x="importance", y="feature", orientation="h",
                                       title="Permutation importance (drop in ROC-AUC)"), width="stretch")

            cm = out["cm"]
            st.caption(f"Confusion matrix at 0.5 threshold on {out['n_test']} test customers: "
                       f"TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

            st.markdown("**Highest revenue at risk** (churn probability × expected spend in the window)")
            show = out["scored"].head(25)[["customer_id", "churn_prob", "recency", "frequency",
                                           "expected_spend", "revenue_at_risk"]].round(2)
            st.dataframe(show, width="stretch")
            st.download_button("⬇️ Download all customer risk scores", out["scored"].to_csv(index=False),
                               "churn_scores.csv", "text/csv")
            st.caption("Scores for customers in the training split are in-sample; judge the model by the "
                       "hold-out metrics above. If your filters include only some categories, 'churn' "
                       "means 'stopped buying in those categories'.")

# --- AI Assistant -----------------------------------------------------------
with tab_ai:
    st.subheader("AI Assistant")
    if not get_api_key():
        st.info("Add an Anthropic API key in the sidebar (or set ANTHROPIC_API_KEY) to enable this tab. "
                "Everything else in the app works without it.")
    else:
        st.markdown("**1. Executive insight summary**")
        if st.button("Generate summary"):
            facts = facts_digest(df)
            with st.spinner("Writing summary..."):
                try:
                    st.markdown(call_llm(
                        "You are a retail analyst. Write 5 concise bullet insights and 2 recommended actions "
                        "for a business owner. Use ONLY the numbers in the JSON facts; never invent figures.",
                        json.dumps(facts)))
                    with st.expander("Facts sent to the model"):
                        st.json(facts)
                except Exception as exc:
                    st.error(f"LLM call failed: {exc}")

        st.markdown("**2. Ask your data**")
        st.caption("Try: 'Top 3 regions by revenue in Electronics', 'Monthly orders for the South region', "
                   "'Average order value by category'. The model returns a JSON spec, never code.")
        q = st.text_input("Your question")
        if q:
            with st.spinner("Thinking..."):
                try:
                    ctx = f"Categories: {all_cats}. Regions: {all_regs}. Data: {min_d} to {max_d}.\nQuestion: {q}"
                    reply = call_llm(SPEC_SYSTEM, ctx, max_tokens=400)
                    spec = json.loads(re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip())
                    res, group, metric, chart = run_spec(df, spec, all_cats, all_regs)
                    if chart == "line" and group != "none":
                        st.plotly_chart(px.line(res, x=group, y=metric, markers=True), width="stretch")
                    elif chart == "bar" and group != "none":
                        st.plotly_chart(px.bar(res, x=group, y=metric), width="stretch")
                    st.dataframe(res, width="stretch")
                    with st.expander("Query spec used (validated against an allow-list)"):
                        st.json(spec)
                except json.JSONDecodeError:
                    st.error("The model did not return valid JSON. Try rephrasing the question.")
                except ValueError as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"LLM call failed: {exc}")

# --- Data -------------------------------------------------------------------
with tab_data:
    st.dataframe(df.drop(columns=["month", "day"]).head(1000), width="stretch")
    st.download_button("⬇️ Download filtered data", df.to_csv(index=False), "filtered_sales.csv", "text/csv")