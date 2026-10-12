from pathlib import Path

import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

print(df.describe(include="all"))

print("\nSide counts:")
print(df["side"].value_counts(dropna=False))

print("\nSymbol counts:")
print(df["symbol"].value_counts(dropna=False))

for col in ["price", "quantity"]:
    values = pd.to_numeric(
        df[col].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )
    print(f"\n{col} statistics:")
    print("Min:", values.min())
    print("Max:", values.max())
    print("Mean:", values.mean())
    print("Median:", values.median())
