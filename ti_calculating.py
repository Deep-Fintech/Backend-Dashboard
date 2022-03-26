from talib import *
import pandas as pd


def calculate_ma(close_df, time_period):
    real_df = MA(close_df, timeperiod=time_period, matype=0)
    return real_df


def calculate_wma(close_df):
    print(close_df)
    wma_df = WMA(close_df, timeperiod=14)
    return wma_df


def calculate_dema(close_df):
    dema_df = DEMA(close_df, timeperiod=14)
    return dema_df


def calculate_tema(close_df):
    tema_df = TEMA(close_df, timeperiod=14)
    return tema_df


def calculate_dx(high_df, low_df, close_df):
    dx_df = DX(high_df, low_df, close_df, timeperiod=14)
    return dx_df


def calculate_adx(high_df, low_df, close_df):
    adx_df = ADX(high_df, low_df, close_df, timeperiod=14)
    return adx_df


def calculate_adxr(high_df, low_df, close_df):
    adxr_df = ADXR(high_df, low_df, close_df, timeperiod=14)
    return adxr_df


def calculate_mfi(high_df, low_df, close_df, volume_df):
    mfi_df = MFI(high_df, low_df, close_df, volume_df, timeperiod=14)
    return mfi_df


def calculate_ppo(close_df):
    ppo_df = PPO(close_df, fastperiod=12, slowperiod=26, matype=0)
    return ppo_df


def calculate_bbands(close_df, time_period):
    upperband, middleband, lowerband = BBANDS(
        close_df, timeperiod=time_period, nbdevup=2, nbdevdn=2, matype=0)
    bbands = pd.DataFrame(
        {
            "upperband": upperband,
            "middleband": middleband,
            "lowerband": lowerband
        }
    )

    return upperband, middleband, lowerband


def calculate_t3(close_df):
    t3_df = T3(close_df, timeperiod=14, vfactor=0.7)
    return t3_df


def calculate_trima(close_df):
    trima_df = TRIMA(close_df, timeperiod=14)
    return trima_df


def calculate_trix(close_df):
    trix_df = TRIX(close_df, timeperiod=14)
    return trix_df


def calculate_willr(high_df, low_df, close_df):
    willr_df = WILLR(high_df, low_df, close_df, timeperiod=14)
    return willr_df


def calculate_ultosc(high_df, low_df, close_df):
    ultosc_df = ULTOSC(high_df, low_df, close_df,
                       timeperiod1=7, timeperiod2=14, timeperiod3=28)
    return ultosc_df


def calculate_mom(close_df):
    mom_df = MOM(close_df, timeperiod=14)
    return mom_df


def calculate_wclprice(high_df, low_df, close_df):
    wclprice_df = WCLPRICE(high_df, low_df, close_df)
    return wclprice_df


def calculate_natr(high_df, low_df, close_df):
    natr_df = NATR(high_df, low_df, close_df, timeperiod=14)
    return natr_df


def calculate_ht_dcperiod(close_df):
    ht_dcperiod_df = HT_DCPERIOD(close_df)
    return ht_dcperiod_df


def calculate_ht_dcphase(close_df):
    ht_dcphase_df = HT_DCPHASE(close_df)
    return ht_dcphase_df


def calculate_ht_trendline(close_df):
    ht_trendline_df = HT_TRENDLINE(close_df)
    return ht_trendline_df


def calculate_adosc(high_df, low_df, close_df, volume_df):
    adosc_df = ADOSC(high_df, low_df, close_df, volume_df,
                     fastperiod=12, slowperiod=26)
    return adosc_df


def calculate_obv(close_df, volume_df):
    obv_df = OBV(close_df, volume_df)
    return obv_df


def calculate_ema(close_df):
    ema_df = EMA(close_df, timeperiod=14)
    return ema_df


def calculate_macd(close_df):
    macd, macd_signal, macd_hist = MACD(
        close_df, fastperiod=12, slowperiod=26, signalperiod=9)
    df = pd.DataFrame(
        {
            "MACD": macd,
            "Signal": macd_signal,
            "Hist": macd_hist
        }
    )
    return macd, macd_signal, macd_hist


def calculate_roc(close_df):
    roc = ROC(close_df, timeperiod=9)
    return roc


def calculate_cci(high_df, low_df, close_df):
    cci = CCI(high_df, low_df, close_df, timeperiod=14)
    return cci


def calculate_atr(high_df, low_df, close_df):
    atr = ATR(high_df, low_df, close_df, timeperiod=14)
    return atr


def calculate_rsi(close_df, time_period):
    rsi = RSI(close_df, timeperiod=time_period)
    return rsi


def calculate_ad(high_df, low_df, close_df, volume_df):
    ad = AD(high_df, low_df, close_df, volume_df)
    return ad


def calculate_apo(close_df):
    apo = APO(close_df, fastperiod=12, slowperiod=26, matype=0)
    return apo


def calculate_midpoint(close_df):
    midpoint = MIDPOINT(close_df, timeperiod=14)
    return midpoint


def calculate_midprice(high_df, low_df):
    midprice = MIDPRICE(high_df, low_df, timeperiod=14)
    return midprice
