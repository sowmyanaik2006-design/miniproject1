from pathlib import Path

import numpy as np
import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path).drop_duplicates(subset="trade_id", keep="first")

df["side"] = df["side"].astype("string").str.strip().str.lower()
df["symbol"] = (
    df["symbol"].astype("string").str.upper()
    .str.replace("/", "", regex=False)
    .str.replace("-", "", regex=False)
)

num_cols = [
    "price", "quantity",
    "bid_price_1", "bid_qty_1", "ask_price_1", "ask_qty_1",
    "bid_price_2", "bid_qty_2", "ask_price_2", "ask_qty_2",
    "bid_price_3", "bid_qty_3", "ask_price_3", "ask_qty_3",
]
for col in num_cols:
    df[col] = pd.to_numeric(
        df[col].astype("string").str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    )

def parse_time(value):
    if pd.isna(value):
        return pd.NaT
    value = str(value).strip()
    if value.isdigit():
        unit = "ms" if len(value) >= 13 else "s"
        return pd.to_datetime(int(value), unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", dayfirst=True, utc=True)

df["timestamp"] = df["timestamp"].apply(parse_time)
df = df.dropna(subset=["trade_id", "timestamp", "price", "quantity"])
df = df[(df["price"] > 0) & (df["quantity"] > 0)]
df = df[df["side"].isin(["buy", "sell"])]

qty_cols = [
    "bid_qty_1", "ask_qty_1", "bid_qty_2", "ask_qty_2",
    "bid_qty_3", "ask_qty_3",
]
for col in qty_cols:
    df[col] = df[col].fillna(df[col].median())

q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df["price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]
df = df[
    df["bid_price_1"].notna()
    & df["ask_price_1"].notna()
    & (df["bid_price_1"] > 0)
    & (df["ask_price_1"] > 0)
    & (df["bid_price_1"] <= df["ask_price_1"])
]
df = df.sort_values("timestamp").reset_index(drop=True)

# Spread
df["spread"] = df["ask_price_1"] - df["bid_price_1"]

# Mid-price
df["mid_price"] = (
    df["ask_price_1"] + df["bid_price_1"]
) / 2

# Order imbalance
denom = df["bid_qty_1"] + df["ask_qty_1"]

df["order_imbalance"] = np.where(
    denom > 0,
    (df["bid_qty_1"] - df["ask_qty_1"]) / denom,
    np.nan
)

# Price returns
df["returns"] = df["price"].pct_change()

# Rolling volatility over 10 observations
df["rolling_volatility"] = (
    df["returns"].rolling(10).std()
)

print(df[
    ["spread", "mid_price", "order_imbalance",
     "returns", "rolling_volatility"]
].head(10))
