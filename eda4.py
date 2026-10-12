from pathlib import Path

import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

# Remove duplicate trade IDs
df = df.drop_duplicates(subset="trade_id", keep="first")

# Standardize side and symbol
df["side"] = df["side"].astype("string").str.strip().str.lower()

df["symbol"] = (
    df["symbol"].astype("string")
    .str.upper().str.replace("/", "", regex=False)
    .str.replace("-", "", regex=False)
)

# Convert price and quantity columns to numbers
num_cols = [
    "price", "quantity",
    "bid_price_1", "bid_qty_1",
    "ask_price_1", "ask_qty_1",
    "bid_price_2", "bid_qty_2",
    "ask_price_2", "ask_qty_2",
    "bid_price_3", "bid_qty_3",
    "ask_price_3", "ask_qty_3"
]

for col in num_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )

# Convert timestamps, including Unix seconds/milliseconds
def parse_time(value):
    if pd.isna(value):
        return pd.NaT

    s = str(value).strip()

    if s.isdigit():
        unit = "ms" if len(s) >= 13 else "s"
        return pd.to_datetime(
            int(s), unit=unit, errors="coerce", utc=True
        )

    return pd.to_datetime(
        s, errors="coerce", dayfirst=True, utc=True
    )

df["timestamp"] = df["timestamp"].apply(parse_time)

# Remove rows missing essential trade information
df = df.dropna(
    subset=["trade_id", "timestamp", "price", "quantity"]
)

# Remove invalid prices and quantities
df = df[(df["price"] > 0) & (df["quantity"] > 0)]

# Keep valid buy/sell labels
df = df[df["side"].isin(["buy", "sell"])]

print("Rows after basic cleaning:", len(df))
print(df.isnull().sum())
