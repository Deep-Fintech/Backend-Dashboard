import pandas as pd
import numpy as np
import json
import requests
import time
import tensorflow as tf
import csv


req = requests.get(
    "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100")
data = req.content
json_data = json.loads(data)
candles_data_frame = pd.DataFrame(json_data)
cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
candles_data_frame.columns = cols

columns = ["open", "high", "low", "close", "volume"]
df_kline = candles_data_frame[columns]

print(df_kline)

df_kline.to_csv("data.csv")
