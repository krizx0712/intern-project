"""
prepare_data.py
Converts sample-dataset/Superstore.csv into the format the dashboard expects:
order_date, customer_id, category, region, quantity, unit_price
Output: superstore_clean.csv (same folder as this script)
Run:  python prepare_data.py
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
SRC = BASE / "sample-dataset" / "Superstore.csv"
OUT = BASE / "superstore_clean.csv"

df = pd.read_csv(SRC, encoding="latin-1")

out = pd.DataFrame(
    {
        "order_date": pd.to_datetime(df["Order Date"], format="%m/%d/%Y"),
        "customer_id": df["Customer ID"],
        "category": df["Category"],
        "region": df["Region"],
        "quantity": df["Quantity"],
        # Sales is the line total, so unit price = Sales / Quantity
        "unit_price": (df["Sales"] / df["Quantity"]).round(2),
    }
)

# Basic cleaning
before = len(out)
out = out.dropna().drop_duplicates()
out = out[out["quantity"] > 0]
out = out.sort_values("order_date")

out.to_csv(OUT, index=False)

print(f"Rows: {before} -> {len(out)} after cleaning")
print(f"Date range: {out['order_date'].min().date()} to {out['order_date'].max().date()}")
print(f"Customers: {out['customer_id'].nunique()}")
print(f"Categories: {sorted(out['category'].unique())}")
print(f"Regions: {sorted(out['region'].unique())}")
print(f"Saved: {OUT}")