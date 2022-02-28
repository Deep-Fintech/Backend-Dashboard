import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import requests
from keras.models import load_model
from flask import Blueprint
import time
from ti_calculating import *
import tensorflow as tf
from Scaler import scaler


predict_by_B4_API = Blueprint('predict_by_B4', __name__)


# @predict_by_B4_API.route('/predict_by_B4')
def predict_by_B4():
    req = requests.get(
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=70")
    data = req.content
    json_data = json.loads(data)
    candles_data_frame = pd.DataFrame(json_data)

    candles_data_frame = candles_data_frame.iloc[:-1, :]

    last_time = candles_data_frame.iloc[-1, 0]

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    candles_data_frame.columns = cols

    columns = ["open", "high", "low", "close", "volume"]
    df_kline = candles_data_frame[columns]

    df = df_kline.iloc[-61:, :]

    # scaler = StandardScaler()
    scaled_kline_data = scaler.fit_transform(df)

    X = []

    for i in range(2):
        X.append(scaled_kline_data[i:i+60])

    X = np.array(X)

    model = load_model("model/model_b4.h5")
    scaled_predictY = model.predict(X)[1]

    scaled_prediction = model.predict(X)
    scaled_prediction_copies = np.repeat(
        scaled_prediction, scaled_kline_data.shape[1], axis=-1)
    prediction = scaler.inverse_transform(scaled_prediction_copies)[:, 3]

    times = {
        't1': (round(int(last_time), -3) + 1*60*1000) / 1000
    }

    predictions = {
        'p1': str(prediction[1])
    }

    response = {
        'times': times,
        'predictions': predictions
    }
    # print("B4 :" , response)

    return response
