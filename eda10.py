from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent
df = pd.read_csv(base_dir / "trades_orderbook_STUDENT_dirty.csv")

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
df = df.sort_values("timestamp").reset_index(drop=True)

candles = (
    df.set_index("timestamp")
    .resample("1min")
    .agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
    )
    .dropna(subset=["open"])
)

df.to_csv(base_dir / "trading_data_cleaned.csv", index=False)
candles.to_csv(base_dir / "trading_1min_OHLCV.csv")

print("EDA completed successfully!")
print("Cleaned data:", df.shape)
print("Candle data:", candles.shape)
