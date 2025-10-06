import pandas as pd
import os
here = os.getcwd()
print(here)
df = pd.read_csv(rf"this_data.csv")

mask = (df == -1).sum(axis=1) <= 5

filtered_df = df[mask]

filtered_df.to_csv("filtered_file.csv", index=False)

print(filtered_df)

