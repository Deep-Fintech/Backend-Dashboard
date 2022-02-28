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
from scipy.signal import argrelextrema
from collections import deque
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy.signal import argrelextrema


predict_singlestep_ABCD_api = Blueprint(
    'predict_singlestep_ABCD_api', __name__)


# Custom loss function

alpha = 18


def custom_loss(y_true, y_pred):

    # extract the "next minute's price" of tensor
    y_true_next = y_true[1:]
    y_pred_next = y_pred[1:]

    # extract the "current's price" of tensor
    y_true_now = y_true[:-1]
    y_pred_now = y_pred[:-1]

    # substract to get up/down movement of the two tensors
    y_true_diff = tf.subtract(y_true_next, y_true_now)
    y_pred_diff = tf.subtract(y_pred_next, y_pred_now)

    # create a standard tensor with zero value for comparison
    standard = tf.zeros_like(y_pred_diff)

    # compare with the standard; if true, UP; else DOWN
    y_true_move = tf.greater_equal(y_true_diff, standard)
    y_pred_move = tf.greater_equal(y_pred_diff, standard)
    y_true_move = tf.reshape(y_true_move, [-1])
    y_pred_move = tf.reshape(y_pred_move, [-1])

    # find indices where the directions are not the same
    condition = tf.not_equal(y_true_move, y_pred_move)
    condition = tf.cast(condition, tf.float32)
    weights = condition*alpha+1

    flattened_difference = tf.reshape((y_pred_next-y_true_next)**2, [-1])

    return (tf.reduce_mean(flattened_difference*weights))


def get_trend(close):
    trend_up = False
    turn_high = True
    high_list = []
    low_list = []
    high_i = 0
    higher_high_count = 0
    lower_high_count = 0

    df = pd.DataFrame({"close": close})
    df["approx_highs"] = np.nan
    df["approx_lows"] = np.nan
    df["higher_high_count"] = np.nan
    df["lower_high_count"] = np.nan
    df["trend"] = np.nan

    for t in range(4, len(close)):

        if(turn_high and close[t-2] > close[t] and close[t-2] > close[t-1] and close[t-2] > close[t-3] and close[t-2] > close[t-4]):
            high_list.append(close[t-2])
            df["approx_highs"][t-2] = close[t-2]

            if(high_i == 0):
                higher_high_count = 1
                lower_high_count = 1

            elif(high_i > 0 and high_list[high_i] > high_list[high_i-1]):
                trend_up = True
                df["trend"][t-2] = 1
                higher_high_count = higher_high_count + 1
                lower_high_count = 0

            elif (high_i > 0 and high_list[high_i] < high_list[high_i-1]):
                trend_up = False
                df["trend"][t-2] = -1
                lower_high_count = lower_high_count + 1
                higher_high_count = 0

            high_i = high_i+1
            turn_high = not(turn_high)

        elif(not(turn_high) and close[t-2] < close[t] and close[t-2] < close[t-1] and close[t-2] < close[t-3] and close[t-2] < close[t-4]):
            low_list.append(close[t-2])
            df["approx_lows"][t-2] = close[t-2]
            turn_high = not(turn_high)

        df["higher_high_count"][t-2] = higher_high_count
        df["lower_high_count"][t-2] = lower_high_count

    df["approx_highs"][0] = df["close"][0]
    df["approx_lows"][0] = df["close"][0]
    df["approx_highs"][len(df["close"])-1] = df["close"][len(df["close"])-1]
    df["approx_lows"][len(df["close"])-1] = df["close"][len(df["close"])-1]
    df["higher_high_count"][0] = 0
    df["lower_high_count"][0] = 0
    df["higher_high_count"][0] = 0

    df["higher_high_count"] = df["higher_high_count"].ffill()
    df["lower_high_count"] = df["lower_high_count"].ffill()
    df["trend"] = df["trend"].ffill()
    df["trend"] = df["trend"].bfill()

    df = df.interpolate(method='piecewise_polynomial')
    return df


def get_scaled_data(df):
    columns_to_scale = ['open', 'high', 'low', 'close',
                        'volume', 'approx_highs', 'approx_lows']
    scaled_kline_data = scaler.fit_transform(df[columns_to_scale])
    scaled_kline_data = np.c_[scaled_kline_data, df["higher_high_count"].to_numpy(
    ), df["lower_high_count"].to_numpy()]
    return scaled_kline_data


# @predict_singlestep_ABCD_api.route('/predict_by_B0')
def predict_single_step_by_Dil():
    req = requests.get(
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=500")
    data = req.content
    json_data = json.loads(data)
    candles_data_frame = pd.DataFrame(json_data)

    last_time = candles_data_frame.iloc[-1, 0]

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    candles_data_frame.columns = cols

    columns = ["open", "high", "low", "close", "volume"]
    df_kline = candles_data_frame[columns]

    df_trend = get_trend(df_kline['close'])

    df_kline["approx_highs"] = df_trend["approx_highs"]
    df_kline["approx_lows"] = df_trend["approx_lows"]
    df_kline["higher_high_count"] = df_trend["higher_high_count"]
    df_kline["lower_high_count"] = df_trend["lower_high_count"]
    df_kline["trend"] = df_trend["trend"]

    
    print("DEBUG => ", df_kline.dtypes)
    df_kline["close"] = pd.to_numeric(df_kline["close"], downcast="float")

    df_kline = df_kline.iloc[-21:, :]


    # scaled_kline_data = get_scaled_data(df_kline)
    columns_to_scale = ['open', 'high', 'low', 'close',
                        'volume', 'approx_highs', 'approx_lows']
    scaled_kline_data = scaler.fit_transform(df_kline[columns_to_scale])
    scaled_kline_data = np.c_[scaled_kline_data, df_kline["higher_high_count"].to_numpy(
    ), df_kline["lower_high_count"].to_numpy()]

    X = []

    for i in range(2):
        X.append(scaled_kline_data[i:i+20])

    X = np.array(X)

    # LSTM-CNN model
    # model = load_model("model/model_custom_loss.h5",
    #                    custom_objects={'custom_loss': custom_loss})

    # CNN model
    model = load_model("model/model_LSTM_CNN.h5",
                       custom_objects={'custom_loss': custom_loss})

    scaled_predictY = model.predict(X)[1]

    scaled_prediction = model.predict(X)
    next_np_scaled = np.delete(scaled_kline_data, np.s_[7:], axis=1)
    scaled_prediction_copies = np.repeat(
        scaled_prediction, next_np_scaled.shape[1], axis=-1)
    prediction = scaler.inverse_transform(scaled_prediction_copies)[:, 3]

    times = {
        't1': round(int(last_time) + 1*60*1000)/1000
    }

    predictions = {
        'p1': str(prediction[1])
    }

    response = {
        'times': times,
        'predictions': predictions
    }

    print(response)

    return response
