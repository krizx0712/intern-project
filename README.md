# 📊 Sales Analytics Dashboard (Streamlit)

Oru Data Science project: sales data-va analyze panni KPIs, charts, forecast, customer segmentation kaattura web app.

## Features
- **Data**: built-in sample data (synthetic, 2 years) allathu unga own CSV upload
- **Filters**: date range, category, region
- **KPIs**: Revenue, Orders, Avg Order Value, Unique Customers
- **EDA charts**: monthly trend, weekday bar, year×month heatmap, category share, region comparison
- **Forecast**: Linear Regression (trend + month seasonality) for next 1-12 months
- **Segmentation**: RFM features + KMeans (Champions, Loyal, Potential, At Risk, Lost)
- **Export**: filtered data and segments CSV download

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Browser-la `http://localhost:8501` open aagum.

## Own CSV format
Columns (exact names): `order_date, customer_id, category, region, quantity, unit_price`

| order_date | customer_id | category | region | quantity | unit_price |
|------------|-------------|----------|--------|----------|-----------|
| 2025-03-14 | 1042 | Clothing | South | 2 | 1299.00 |

## Project structure
```
sales_dashboard/
├── app.py
├── requirements.txt
└── README.md
```

## Next steps (improve panna)
- Real dataset use pannunga (Kaggle "Superstore Sales" / "Online Retail")
- Prophet / SARIMA forecasting add pannunga, train/test split-oda evaluate pannunga
- Streamlit Community Cloud-la free-a deploy pannunga
