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
    df["app_highs"] = df["app_highs"].interpolate(
        method='piecewise_polynomial')
    df["app_lows"] = df["app_lows"].interpolate(method='piecewise_polynomial')

    df["approx"].iloc[0] = df['close'].iloc[0]
    df["approx"].iloc[-1] = df['close'].iloc[-1]

    df["app_highs"].iloc[0] = df['close'].iloc[0]
    df["app_highs"].iloc[-1] = df['close'].iloc[-1]

    df["app_lows"].iloc[0] = df['close'].iloc[0]
    df["app_lows"].iloc[-1] = df['close'].iloc[-1]

    df["approx"] = df["approx"].interpolate(method='piecewise_polynomial')
    df["app_highs"] = df["app_highs"].interpolate(
        method='piecewise_polynomial')
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


def getABCD(data):
    look_order = 20
    max_idx = argrelextrema(data['close'].values,
                            np.greater, order=look_order)[0]
    min_idx = argrelextrema(data['close'].values, np.less, order=look_order)[0]

    min_max_list = np.concatenate((min_idx, max_idx))
    min_max_list.sort()

    new_max_list = []
    temp_max_list = []

    new_min_list = []
    temp_min_list = []

    for count, idx in enumerate(min_max_list):
        if idx in max_idx:
            temp_max_list.append(idx)
            if(len(temp_min_list) > 0):
                new_min_list.append(temp_min_list)
                temp_min_list = []
        else:
            temp_min_list.append(idx)
            if(len(temp_max_list) > 0):
                new_max_list.append(temp_max_list)
                temp_max_list = []

    if(len(temp_max_list) > 0):
        new_max_list.append(temp_max_list)
    if(len(temp_min_list) > 0):
        new_min_list.append(temp_min_list)

    global_max_list = []
    global_min_list = []

    for i in new_max_list:
        #     print (i)
        global_max_list.append(data.iloc[i]['close'].idxmax())

    for i in new_min_list:
        #     print (i)
        global_min_list.append(data.iloc[i]['close'].idxmin())

    global_min_max_list = np.concatenate((global_min_list, global_max_list))
    global_min_max_list.sort()

    return global_min_list, global_max_list, global_min_max_list


def getTrendFeature(df, global_min_list, global_max_list, global_min_max_list):
    df['trend'] = np.nan
    for count in range(len(global_min_max_list)-1):
        if global_min_max_list[count] in global_max_list:
            for i in range(global_min_max_list[count], global_min_max_list[count+1]):
                #             print ("DOWN")
                #             train_data['trend'][i] = "DOWN"
                df['trend'][i] = -1

        else:
            for i in range(global_min_max_list[count], global_min_max_list[count+1]):
                #             print ("UP")
                df['trend'][i] = 1

    df['trend'] = df['trend'].interpolate(method='nearest').ffill().bfill()

    return df


def getHullMA(df):
    df_to_hull_MA = df[['open', 'high', 'low', 'close']]
    df_to_hull_MA_array = df_to_hull_MA.to_numpy()

    import numpy as np

    def lwma(Data, lookback):

        weighted = []
        for i in range(len(Data)):
            try:
                total = np.arange(1, lookback + 1, 1)

                matrix = Data[i - lookback + 1: i + 1, 3:4]
                matrix = np.ndarray.flatten(matrix)
                matrix = total * matrix
                wma = (matrix.sum()) / (total.sum())
                weighted = np.append(weighted, wma)

            except ValueError:
                pass

        Data = Data[lookback - 1:, ]
        weighted = np.reshape(weighted, (-1, 1))
        Data = np.concatenate((Data, weighted), axis=1)

        return Data
    # For this function to work, you need to have an OHLC array composed of the four usual columns, then you can use the below syntax to get a data array with the weighted moving average using the lookback you need
    Hull_MA = pd.DataFrame(lwma(df_to_hull_MA_array, 20))[4]
    df["Hull_MA"] = Hull_MA
    return df

# @predict_singlestep_ABCD_api.route('/predict_by_B0')


def predict_single_step():
    req = requests.get(
        "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=500")
    data = req.content
    json_data = json.loads(data)
    candles_data_frame = pd.DataFrame(json_data)

    candles_data_frame = candles_data_frame.iloc[:-1, :]

    last_time = candles_data_frame.iloc[-1, 0]
    print("PREDICT OUR MODEL ", last_time)

    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    candles_data_frame.columns = cols

    columns = ["open", "high", "low", "close", "volume"]
    df_kline = candles_data_frame[columns]

    # print("DEBUG => ", df_kline.dtypes)
    df_kline["close"] = pd.to_numeric(df_kline["close"], downcast="float")
    df_kline = get_HHLL(df_kline)

    # ABCD
    global_min_list, global_max_list, global_min_max_list = getABCD(df_kline)

    # Trend feature
    df_kline = getTrendFeature(
        df_kline, global_min_list, global_max_list, global_min_max_list)

    # Hull Moving Average
    df_kline = getHullMA(df_kline)

    df_kline = df_kline.iloc[-15:, :]

    columns_to_scale = ['open', 'high', 'low', 'close',
                        'volume', 'approx', 'app_highs', 'app_lows']

    # scaler = StandardScaler()
    scaled_kline_data = scaler.fit_transform(
        df_kline[columns_to_scale])  # df ==> when tech indicators

    scaled_kline_data = np.c_[scaled_kline_data,  df_kline["trend"].to_numpy()]

    X = []

    for i in range(2):
        X.append(scaled_kline_data[i:i+14])

    X = np.array(X)

    # LSTM-CNN model
    # model = load_model("model/model_custom_loss.h5",
    #                    custom_objects={'custom_loss': custom_loss})

    # CNN model
    model = load_model("model/model_ABCD.h5",
                       custom_objects={'custom_loss': custom_loss})

    scaled_predictY = model.predict(X)[1]

    scaled_prediction = model.predict(X)
    next_np_scaled = np.delete(scaled_kline_data, np.s_[8], axis=1)
    scaled_prediction_copies = np.repeat(
        scaled_prediction, next_np_scaled.shape[1], axis=-1)
    prediction = scaler.inverse_transform(scaled_prediction_copies)[:, 3]

    times = {
        't1': round(int(last_time) + 1*60*1000)/1000
        # 't1': round(int(last_time))/1000
    }

    print("TIMES TO RETURN ", times)

    predictions = {
        'p1': str(prediction[1])
    }

    response = {
        'times': times,
        'predictions': predictions
    }

    # print(response)

    return response
