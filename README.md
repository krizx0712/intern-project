# 📊 Sales Analytics Dashboard

An interactive data science dashboard built with **Python, Streamlit, Plotly and scikit-learn**. It analyses retail sales data and shows KPIs, trends, a revenue forecast and customer segments.

## Features
- **Data:** built-in synthetic sample data, or upload your own CSV
- **Filters:** date range, category, region
- **KPIs:** total revenue, orders, average order value, unique customers
- **EDA charts:** monthly trend, revenue by weekday, year × month heatmap, category share, region comparison
- **Forecast:** Linear Regression (time trend + month-of-year seasonality) for the next 1-12 months
- **Customer segmentation:** RFM features + KMeans (Champions, Loyal, Potential, At Risk, Lost)
- **Currency:** view values in $ or ₹ using an exchange rate you enter
- **Export:** download the filtered data and customer segments as CSV

## Tech stack
Python · Streamlit · pandas · NumPy · Plotly · scikit-learn

## Dataset
Tested with the Superstore sales dataset from Kaggle:
https://www.kaggle.com/datasets/mayurjare/superstore-data

The dataset is not included in this repository. Download it, extract it to a `sample-dataset/` folder, then convert it with `prepare_data.py` (see below).

## Project structure
```
intern-project/
├── app.py               # Streamlit dashboard
├── prepare_data.py      # Converts Superstore.csv to the dashboard format
├── requirements.txt
├── README.md
└── sample-dataset/      # (not in repo) place Superstore.csv here
```

## Setup and run
```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Without any upload, the app shows the built-in sample data.

## Using the Superstore dataset
1. Download the dataset from Kaggle and put `Superstore.csv` in `sample-dataset/`.
2. Convert it:
   ```bash
   python prepare_data.py
   ```
   This creates `superstore_clean.csv`.
3. In the dashboard sidebar, upload `superstore_clean.csv`. Values are shown in $ (the original currency); choose ₹ and enter an exchange rate to convert.

## Using your own CSV
Required columns (exact names): `order_date, customer_id, category, region, quantity, unit_price`

| order_date | customer_id | category | region | quantity | unit_price |
|------------|-------------|----------|--------|----------|-----------|
| 2025-03-14 | 1042 | Clothing | South | 2 | 1299.00 |

## Notes and limitations
- The forecast model is evaluated on the data it was trained on, so its R² is optimistic. Treat the forecast as a rough guide.
- The exchange rate is entered manually and is not fetched automatically.

## Possible improvements
- Train/test split and backtesting for the forecast
- Prophet or SARIMA models
- Deploy on Streamlit Community Cloud