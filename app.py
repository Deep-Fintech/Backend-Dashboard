from flask import Flask, jsonify
from flask_socketio import *
from binance import ThreadedWebsocketManager
import json
from flask_cors import CORS
from predict.Predict import predict_api
# from PredictSingleStep import predict_singlestep_api
from predict.PredictSingleStep import predict_singlestep_api
from predict.PredictSingleStepWithABCD import predict_singlestep_ABCD_api
from predict.PredictByB1 import predict_by_B1, predict_by_B1_API
from predict.PredictByB2 import predict_by_B2, predict_by_B2_API
import time
from predict.PredictByB4 import predict_by_B4, predict_by_B4_API
from keys.Keys import api_key, api_secret
import threading

# import method
from predict.PredictSingleStepWithABCD import predict_single_step
from predict.PredictWithDilankaABCD import predict_single_step_by_Dil
from profit.Profit import getActionProfit

app = Flask(__name__)
CORS(app)

# app.register_blueprint(predict_api)
# app.register_blueprint(predict_singlestep_api)
# app.register_blueprint(predict_singlestep_ABCD_api)
# app.register_blueprint(predict_by_B1_API)
# app.register_blueprint(predict_by_B2_API)
# app.register_blueprint(predict_by_B4_API)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


def main():
    symbol = 'BTCUSDT'
    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()
    twm.start_kline_socket(callback=send_kline, symbol=symbol)
    twm.join()


@app.route('/')
def hello():
    main()
    return 'KLine live update done'


@app.route('/hello')
def helloDeepFintech():
    return 'Hello Deep Fintech!'


@socketio.on('connect')
def connected():
    print('Socket Connected')


model_price = []


# Prediction by our model


@socketio.on('send_prediction')
def send_prediction():
    data = predict_single_step()
    print("Data from App.py", data)
    socketio.emit('PREDICTION', data)


@socketio.on('send_prediction_B1')  # Prediction  by benchmark one
def send_prediction_B1():
    data = predict_by_B1()
    print("Data from B1", data)
    socketio.emit("PREDICTION_B1", data)


@socketio.on('send_prediction_B2')  # Prediction  by benchmark two
def send_prediction_B2():
    data = predict_by_B2()
    print("Data from B2", data)
    socketio.emit("PREDICTION_B2", data)


@socketio.on('send_prediction_B3')  # Prediction  by benchmark three
def send_prediction_B3():
    time.sleep(2)
    data = predict_by_B4()
    print("Data from B3", data)
    socketio.emit("PREDICTION_B3", data)


@socketio.on('send_kline')
def send_kline(data):
    kline = {
        'time': round((data['k']['t']) / 1000),
        'open': float(data['k']['o']),
        'high': float(data['k']['h']),
        'low': float(data['k']['l']),
        'close': float(data['k']['c']),
        'status': data['k']['x']
    }
    # print(json.dumps(kline))
    socketio.emit('KLINE', kline)
    if (data['k']['x']):
        closeKLINE = {
            'time': round((data['k']['t']) / 1000),
            'close': float(data['k']['c']),
        }
        socketio.emit('CLOSE', closeKLINE)
        print("CLOSE : ", float(data['k']['c']))
        time.sleep(2)
        # threading.Thread(target=send_prediction).start()
        # threading.Thread(target=send_prediction_B1).start()
        # threading.Thread(target=send_prediction_B2).start()
        # threading.Thread(target=send_prediction_B3).start()
        send_prediction()
        send_prediction_B1()
        send_prediction_B2()
        send_prediction_B3()


if __name__ == '__main__':
    # main()
    socketio.run(app)

name = "dff"
