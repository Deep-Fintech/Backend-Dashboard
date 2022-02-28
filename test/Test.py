
from binance import ThreadedWebsocketManager
from Keys import api_key, api_secret
import json
import datetime
import csv
import pandas as pd
import numpy as np
import json
import requests
import time
import tensorflow as tf
import csv


req = requests.get(
    "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=40000")
data = req.content
json_data = json.loads(data)
candles_data_frame = pd.DataFrame(json_data)
cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
        "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
candles_data_frame.columns = cols

columns = ["open_time", "open", "high", "low", "close", "volume"]
df_kline = candles_data_frame[columns]

print(df_kline)

df_kline.to_csv("data.csv")

# with open('data.csv', 'w') as csvfile:
#     fieldnames = ['time', 'open', 'high', 'low', 'close']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()


def send_kline(data):
    kline = {
        'open_time': datetime.datetime.fromtimestamp(data['k']['t']/1000.0),
        'open': float(data['k']['o']),
        'high': float(data['k']['h']),
        'low': float(data['k']['l']),
        'close': float(data['k']['c']),
        'volume': float(data['k']['c'])
    }
    fieldnames = ['open_time', 'open', 'high', 'low', 'close', 'volume']

    if(data['k']['x']):
        print (kline)
        with open('data.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(kline)


symbol = 'BTCUSDT'
twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
twm.start()
twm.start_kline_socket(callback=send_kline, symbol=symbol)
twm.join()
