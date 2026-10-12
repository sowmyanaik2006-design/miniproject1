from pathlib import Path

import pandas as pd

csv_path = Path(__file__).with_name("trades_orderbook_STUDENT_dirty.csv")
df = pd.read_csv(csv_path)

print("First 5 rows:")
print(df.head())
print("Shape:", df.shape)
print("\nDataFrame summary:")
df.info()
print("\nColumn data types:")
print(df.dtypes)
