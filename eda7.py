from pathlib import Path

import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

for col in ["price", "quantity"]:
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
    year_first = len(value) >= 10 and value[:4].isdigit() and value[4] == "-"
    return pd.to_datetime(
        value, errors="coerce", dayfirst=not year_first, utc=True
    )

df["timestamp"] = df["timestamp"].apply(parse_time)
df = df.dropna(subset=["timestamp", "price", "quantity"])
df = df[(df["price"] > 0) & (df["quantity"] > 0)]

q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df["price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

df = df.set_index("timestamp").sort_index()

candles = df.resample("1min").agg(
    open=("price", "first"),
    high=("price", "max"),
    low=("price", "min"),
    close=("price", "last"),
    volume=("quantity", "sum")
)

candles = candles.dropna(subset=["open"])

print(candles.head(10))
