# 📊 Retail Analytics & ML Platform

End-to-end retail analytics app built with Streamlit: backtested forecasting, customer segmentation,
churn prediction with explainability, anomaly detection and an optional LLM assistant.

## Features
- **Welcome + login page** (demo auth) before the dashboard
- **Overview**: KPIs, trend, weekday, heatmap, category/region split, STL anomaly detection
- **Forecast Lab**: walk-forward backtest of 4 models (Seasonal Naive, Linear, Holt-Winters, Hybrid Trend+GBM), MAE/RMSE/MAPE, prediction intervals
- **Customers**: RFM + KMeans (k chosen by silhouette), cohort retention heatmap
- **Churn**: Logistic Regression vs HistGradientBoosting, ROC-AUC / PR-AUC, permutation importance, revenue-at-risk table
- **AI Assistant** (optional, Anthropic API key): executive summary from computed facts; "Ask your data" where the LLM returns a JSON query spec that is validated against an allow-list (no code execution)

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Demo login: `admin / admin123` (override with `DASH_USER` / `DASH_PASS` env vars or Streamlit secrets), or continue as guest.

## Run with Docker
```bash
docker build -t retail-analytics .
docker run -p 8501:8501 retail-analytics
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```
CI runs the same tests on every push (`.github/workflows/ci.yml`).

## Own CSV format
Columns (exact names): `order_date, customer_id, category, region, quantity, unit_price`

| order_date | customer_id | category | region | quantity | unit_price |
|------------|-------------|----------|--------|----------|-----------|
| 2025-03-14 | 1042 | Clothing | South | 2 | 1299.00 |

Rows with quantity or unit_price <= 0 are dropped. The forecast backtest needs at least 18 months of data.

## Method notes
- Backtest: expanding window, 1-step-ahead; each prediction uses only earlier months (no leakage).
- Churn: features use only orders up to a snapshot date; label = no order in the following N days; metrics are on a 25% hold-out.
- The sample data is synthetic. Swap in a real dataset for meaningful results.

## Limitations
- Login is a demo gate, not production security.
- Holt-Winters uses seasonality only with >= 24 months of training data.