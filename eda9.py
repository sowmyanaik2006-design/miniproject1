from pathlib import Path

import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

df["price"] = pd.to_numeric(
    df["price"].astype("string").str.replace(r"[$,]", "", regex=True),
    errors="coerce",
)
df["quantity"] = pd.to_numeric(
    df["quantity"].astype("string").str.replace(r"[$,]", "", regex=True),
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
    volume=("quantity", "sum"),
).dropna(subset=["open"])

# Predict whether the price will rise in the next 5 minutes
candles["future_close"] = candles["close"].shift(-5)

candles["target"] = (
    candles["future_close"] > candles["close"]
).astype("Int64")

# Remove rows without a future target
model_data = candles.dropna(subset=["future_close"]).copy()

# Select features
X = model_data[["open", "high", "low", "close", "volume"]]
y = model_data["target"]

# Split chronologically: first 70% for training
split = int(len(model_data) * 0.70)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
