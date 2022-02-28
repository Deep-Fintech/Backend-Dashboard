import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import requests
from keras.models import load_model
from flask import Blueprint
import time
# from Scaler import scaler

predict_api = Blueprint('predict_api', __name__)


@predict_api.route('/predict')
def predict():
    req = requests.get(
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=15")
    data = req.content
    json_data = json.loads(data)
    candles_data_frame = pd.DataFrame(json_data)

    last_time = candles_data_frame.iloc[-1, 0]
    df_kline = candles_data_frame.iloc[:, 1:6]

    scaler = StandardScaler()
    scaled_kline_data = scaler.fit_transform(df_kline)

    X = []

    for i in range(2):
        X.append(scaled_kline_data[i:i+14])

    X = np.array(X)

    model = load_model("model/model_1.h5")
    scaled_predictY = model.predict(X)[1]

    flatten_scaled_predictY = scaled_predictY.reshape(
        scaled_kline_data.shape[1], 1)
    predictY_copies = np.repeat(flatten_scaled_predictY,
                                scaled_kline_data.shape[1], axis=-1)
    predictY = scaler.inverse_transform(predictY_copies)[:, 3]
    predictY = np.array(predictY)

    times = {
        't1': round(int(last_time) + 1*60*1000)/1000,
        't2': round(int(last_time) + 2*60*1000)/1000,
        't3': round(int(last_time) + 3*60*1000)/1000,
        't4': round(int(last_time) + 4*60*1000)/1000,
        't5': round(int(last_time) + 5*60*1000)/1000,
    }

    predictions = {
        'p1': str(predictY[0]),
        'p2': str(predictY[1]),
        'p3': str(predictY[2]),
        'p4': str(predictY[3]),
        'p5': str(predictY[4]),
    }

    response = {
        'times': times,
        'predictions': predictions
    }

    print (predictions)

    return response
