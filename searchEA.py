import pandas as pd
df = pd.read_csv("LiIonDatabase.csv", skiprows=3)
print(df.columns.tolist())