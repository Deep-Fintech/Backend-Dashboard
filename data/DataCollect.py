from binance.client import Client
from datetime import datetime
from pandas import DataFrame

api_key = "0baISLZo7wvww0D2IfjAnIbb678oGmHyNbcDlJBs2MR308O7lS08A9n3kXLBNJCu"
api_secret = "InrQIBW2Rb7Gn4aEjJGksyDEPojeaLSbmCq1hBeZGwDhiYQlXROg7yMcA5NITaoW"

def binance_price_train(symbol, Pkey, Skey):
    client = Client(api_key=Pkey, api_secret=Skey)
    candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "18 Jan, 2022", "18 Feb, 2022")

    candles_data_frame = DataFrame(candles)

    open_time = []

    for i in candles_data_frame[0]:
        open_time.append(datetime.fromtimestamp(int(i / 1000)))

    del candles_data_frame[0]
    del candles_data_frame[6]
    del candles_data_frame[11]

    final_dataframe = DataFrame(open_time).join(candles_data_frame)

    final_dataframe.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'asset_volume',
                               'number_of_trades', 'taker_buy_base', 'taker_buy_quote']

    return final_dataframe

binance_price_train("BTCUSDT", Pkey=api_key, Skey=api_secret).to_csv("data.csv", index=False)