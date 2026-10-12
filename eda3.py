import pandas as pd

df = pd.read_csv("trades_orderbook_STUDENT_dirty.csv")

print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

if "trade_id" in df.columns:
    print("Duplicate trade IDs:", df["trade_id"].duplicated().sum())

print("Negative prices:",
      (pd.to_numeric(df["price"].astype(str)
       .str.replace(r"[$,]", "", regex=True),
       errors="coerce") < 0).sum())

print("Zero quantities:",
      (pd.to_numeric(df["quantity"], errors="coerce") == 0).sum())