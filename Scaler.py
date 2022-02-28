from sklearn.preprocessing import StandardScaler
import pandas as pd

data_to_scale = pd.read_csv("data/data.csv")
df = data_to_scale.copy()

columns = ["open","high","low","close","volume"]
df = df[columns]

scaler = StandardScaler()
scaled_df = scaler.fit(df)


