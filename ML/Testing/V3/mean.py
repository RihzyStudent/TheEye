import pandas as pd
import numpy as np
df = pd.read_csv("filtered_file.csv")

# Replace -1 with NaN so they won't count in stats
df_replaced = df.replace(-1, np.nan)

# Compute mean and std per column (NaNs are ignored)
means = df_replaced.mean()
stds = df_replaced.std()

print("Column Means:")
print(means)

print("\nColumn Standard Deviations:")
print(stds)

percent_minus1 = (df.eq(-1).sum(axis=0) / len(df)) * 100

print("Percentage of -1s per column:")
print(percent_minus1)