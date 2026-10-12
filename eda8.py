from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

df["price"] = pd.to_numeric(
    df["price"].astype("string").str.replace(r"[$,]", "", regex=True),
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
df = df.dropna(subset=["timestamp", "price"])
df = df[df["price"] > 0].sort_values("timestamp")

q1, q3 = df["price"].quantile([0.25, 0.75])
iqr = q3 - q1
df = df[df["price"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

df = df.set_index("timestamp")
df["price"].plot(figsize=(10, 4), title="Trade Price Over Time")
plt.ylabel("Price")
plt.tight_layout()
output_path = Path(__file__).with_name("trade_price_over_time.png")
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Chart saved to: {output_path}")
