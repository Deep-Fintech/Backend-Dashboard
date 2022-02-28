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


predict_singlestep_api = Blueprint('predict_singlestep_api', __name__)


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


def trading_indicators(df):
    ma5 = calculate_ma(df["close"], 5)  # calculate moving average 5
    ma8 = calculate_ma(df["close"], 8)  # calculate moving average 8
    ma13 = calculate_ma(df["close"], 13)  # calculate moving average 13

    # Calculate RSI
    rsi = calculate_rsi(df["close"])

    # Calculate MACD
    macd, macd_signal, macd_hist = calculate_macd(df["close"])

    # Calculate BBANDS
    bbands_upper, bbands_middle, bbands_lower = calculate_bbands(df["close"])

    # Calculate OBV
    obv = calculate_obv(df["close"], df["volume"])

    df["ma5"] = ma5
    df["ma8"] = ma8
    df["ma13"] = ma13
    df["rsi"] = rsi
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["bbands_upper"] = bbands_upper
    df["bbands_middle"] = bbands_middle
    df["bbands_lower"] = bbands_lower
    df["obv"] = obv

    return df


def get_HHLL(df):
    max_idx = argrelextrema(df['close'].values, np.greater, order=1)[0]
    min_idx = argrelextrema(df['close'].values, np.less, order=1)[0]

    min_max_list = np.concatenate((min_idx, max_idx))
    min_max_list.sort()
    list(min_max_list)

    df["approx"] = np.nan
    df["app_highs"] = np.nan
    df["app_lows"] = np.nan

    df["approx"].iloc[min_max_list] = df['close'].iloc[min_max_list]
    df["app_highs"].iloc[max_idx] = df['close'].iloc[max_idx]
    df["app_lows"].iloc[min_idx] = df['close'].iloc[min_idx]

    df["approx"] = df["approx"].interpolate(method='piecewise_polynomial')
    df["app_highs"] = df["app_highs"].interpolate(method='piecewise_polynomial')
    df["app_lows"] = df["app_lows"].interpolate(method='piecewise_polynomial')


    df["approx"].iloc[0] = df['close'].iloc[0]
    df["approx"].iloc[-1] = df['close'].iloc[-1]

    df["app_highs"].iloc[0] = df['close'].iloc[0]
    df["app_highs"].iloc[-1] = df['close'].iloc[-1]

    df["app_lows"].iloc[0] = df['close'].iloc[0]
    df["app_lows"].iloc[-1] = df['close'].iloc[-1]


    df["approx"] = df["approx"].interpolate(method='piecewise_polynomial')
    df["app_highs"] = df["app_highs"].interpolate(method='piecewise_polynomial')
    df["app_lows"] = df["app_lows"].interpolate(method='piecewise_polynomial')


    return df


def getMinMax(df):
    max_idx = argrelextrema(df['close'].values, np.greater, order=5)[0]
    min_idx = argrelextrema(df['close'].values, np.less, order=5)[0]

    df["min_max"] = 0
    df["min_max"].iloc[max_idx] = 1
    df["min_max"].iloc[min_idx] = -1

    min_max_list = np.concatenate((min_idx, max_idx))
    min_max_list.sort()
    list(min_max_list)

    df["approx"] = np.nan
    # df["app_highs"] = np.nan
    # df["app_lows"] = np.nan

    df["approx"].iloc[min_max_list] = df['close'].iloc[min_max_list]
    # df["app_highs"].iloc[max_idx] = df['close'].iloc[max_idx]
    # df["app_lows"].iloc[min_idx] = df['close'].iloc[min_idx]

    df["approx"] = df["approx"].interpolate(method='piecewise_polynomial')
    # df["app_highs"] = df["app_highs"].interpolate(method='piecewise_polynomial')
    # df["app_lows"] = df["app_lows"].interpolate(method='piecewise_polynomial')


    df["approx"].iloc[0] = df['close'].iloc[0]
    df["approx"].iloc[-1] = df['close'].iloc[-1]

    # df["app_highs"].iloc[0] = df['close'].iloc[0]
    # df["app_highs"].iloc[-1] = df['close'].iloc[-1]

    # df["app_lows"].iloc[0] = df['close'].iloc[0]
    # df["app_lows"].iloc[-1] = df['close'].iloc[-1]


    df["approx"] = df["approx"].interpolate(method='piecewise_polynomial')
    # df["app_highs"] = df["app_highs"].interpolate(method='piecewise_polynomial')
    # df["app_lows"] = df["app_lows"].interpolate(method='piecewise_polynomial')


    return df





# @predict_singlestep_api.route('/predict_by_B0')
def predict_single_step():
    req = requests.get(
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100")
    data = req.content
    json_data = json.loads(data)
    candles_data_frame = pd.DataFrame(json_data)

    last_time = candles_data_frame.iloc[-1, 0]

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    candles_data_frame.columns = cols

    columns = ["open", "high", "low", "close", "volume"]
    df_kline = candles_data_frame[columns]

    print ("DEBUG => ", df_kline.dtypes )
    df_kline["close"] = pd.to_numeric(df_kline["close"], downcast="float")
    df_kline = get_HHLL(df_kline)

    # Add Technical indicators
    # df_with_indicators = trading_indicators(df_kline)

    # df = df_with_indicators.iloc[-15:, :] # When tech indcatyors added

    df_kline = df_kline.iloc[-21:, :]
    # scaler = StandardScaler()
    scaled_kline_data = scaler.fit_transform(
        df_kline)  # df ==> when tech indicators

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
    scaled_prediction_copies = np.repeat(
        scaled_prediction, scaled_kline_data.shape[1], axis=-1)
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
