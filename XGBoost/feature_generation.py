import numpy as np
import pandas as pd
import talib
from arch import arch_model


def SMA(data, window, col="Close"):
    data[f"SMA_{col}_{window}"] = talib.SMA(data[col], window)


def SMA_ratio(data, window_1=5, window_2=15, col="Close"):
    sma_1 = talib.SMA(data[col], window_1)
    sma_2 = talib.SMA(data[col], window_2)
    data[f"SMA_ratio_{col}_{window_1}_{window_2}"] = sma_1 / sma_2


def Parabolic_SAR(data, high_col="High", low_col="Low", acceleration=0.02, maximum=0.2):
    data["Parabolic_SAR"] = talib.SAR(
        data[high_col], data[low_col], acceleration, maximum
    )


def ChaikinAD(
    data, high_col="High", low_col="Low", close_col="Close", volume_col="Volume"
):
    data["ChaikinAD"] = talib.AD(
        data[high_col], data[low_col], data[close_col], data[volume_col]
    )


def MA_on_RSI(data, rsi_window_1=5, rsi_window_2=15, sma_window=9, col="Close"):
    rsi_1 = talib.RSI(data[col], rsi_window_1)
    rsi_2 = talib.RSI(data[col], rsi_window_2)
    ratio = rsi_1 / rsi_2
    data[f"MA_on_RSI_{rsi_window_1}_{rsi_window_2}_{sma_window}_"] = talib.SMA(
        ratio, sma_window
    )


def ATR_ratio(
    data, window_1=5, window_2=15, high_col="High", low_col="Low", close_col="Close"
):
    atr_1 = talib.ATR(
        data[high_col], data[low_col], data[close_col], timeperiod=window_1
    )
    atr_2 = talib.ATR(
        data[high_col], data[low_col], data[close_col], timeperiod=window_2
    )
    data[f"ATR_ratio_{close_col}_{window_1}_{window_2}"] = atr_1 / atr_2


def RSI_ratio(data, window_1=5, window_2=15, col="Close"):
    rsi_1 = talib.RSI(data[col], window_1)
    rsi_2 = talib.RSI(data[col], window_2)
    data[f"RSI_ratio_{col}_{window_1}_{window_2}"] = rsi_1 / rsi_2


def EMA(data, window, col="Close"):
    data[f"EMA_{col}_{window}"] = talib.EMA(data[col], timeperiod=window)


def RSI(data, window, col="Close"):
    data[f"RSI_{col}_{window}"] = talib.RSI(data[col], window)


def MA(data, window, col="Close"):
    data[f"MA_{col}_{window}"] = talib.MA(data[col], timeperiod=window)


def MOM(data, window, col="Close"):
    data[f"MTM_{col}_{window}"] = talib.MOM(data[col], timeperiod=window)


def ROC(data, window, col="Close"):
    data[f"ROC_{col}_{window}"] = talib.ROC(data[col], timeperiod=window)


def MACD(data, col="Close", fastperiod=12, slowperiod=26, signalperiod=9):
    macd, macdsignal, macdhist = talib.MACD(
        data[col],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )

    data["MACD"] = macd
    data["MACD_SIG"] = macdsignal
    data["MACD_HIST"] = macdhist


def STOCHASTIC(
    data,
    high_col="High",
    low_col="Low",
    col="Close",
    fastk_period=5,
    fastd_period=3,
    fastd_matype=0,
    slowd_period=3,
    slowd_matype=0,
):

    FASTK, FASTD = talib.STOCHF(
        data[high_col],
        data[low_col],
        data[col],
        fastk_period=fastk_period,
        fastd_period=fastd_period,
        fastd_matype=fastd_matype,
    )
    SLOWK, SLOWD = talib.STOCH(
        data[high_col],
        data[low_col],
        data[col],
        fastk_period=fastk_period,
        slowk_period=fastd_period,
        slowk_matype=fastd_matype,
        slowd_period=slowd_period,
        slowd_matype=slowd_matype,
    )

    data["FASTK"] = FASTK
    data["FASTD"] = FASTD
    data["SLOWK"] = SLOWK
    data["SLOWD"] = SLOWD


def BOLL(data, window=20, col="Close"):
    _data = pd.DataFrame(None)
    _data["MA"] = data[col].rolling(window).mean()
    _data["SD"] = data[col].rolling(window).std()
    data[f"BOLL_UPPER_{window}"] = _data["MA"] + 2 * _data["SD"]
    data[f"BOLL_LOWER_{window}"] = _data["MA"] - 2 * _data["SD"]


def ATR(data, window, high_col="High", low_col="Low", col="Close"):
    data[f"ATR_{col}_{window}"] = talib.ATR(
        data[high_col], data[low_col], data[col], timeperiod=window
    )


def MFI(data, window, high_col="High", low_col="Low", col="Close", vol_col="Volume"):
    data[f"MFI_{col}_{window}"] = talib.MFI(
        data[high_col],
        data[low_col],
        data[col],
        data[vol_col],
        timeperiod=window,
    )


def WILLR(data, window, high_col="High", low_col="Low", col="Close"):
    data[f"WILLR_{col}_{window}"] = talib.WILLR(
        data[high_col], data[low_col], data[col], timeperiod=window
    )


def ADX(data, window, high_col="High", low_col="Low", col="Close"):
    data[f"ADX_{col}_{window}"] = talib.ADX(
        data[high_col], data[low_col], data[col], timeperiod=window
    )


def CCI(data, window, high_col="High", low_col="Low", col="Close"):
    data[f"CCI_{col}_{window}"] = talib.CCI(
        data[high_col], data[low_col], data[col], timeperiod=window
    )


def Wilder(data, window, col="Close"):
    start = np.where(~np.isnan(data[col]))[0][0]
    Wilder = np.array([np.nan] * len(data[col]))
    Wilder[start + window - 1] = data[col][start : (start + window)].mean()
    for i in range(start + window, len(data[col])):
        Wilder[i] = (Wilder[i - 1] * (window - 1) + data[col][i]) / window
    data[f"Wilder_{col}_{window}"] = Wilder


def VWMA(data, window, col="Close", weighted_col="Volume"):
    weighted_data = (data[col] * data[weighted_col]).rolling(window=window).sum()
    summed_weights = data[weighted_col].rolling(window=window).sum()
    data[f"VWMA_{col}_{window}"] = weighted_data / summed_weights


def WVAD(
    data,
    window,
    open_col="Open",
    high_col="High",
    low_col="Low",
    close_col="Close",
    volume_col="Volume",
):
    weighted_data = (
        (
            (data[open_col] - data[close_col])
            / (data[high_col] - data[low_col])
            * data[volume_col]
        )
        .rolling(window=window)
        .sum()
    )
    summed_weights = data[volume_col].rolling(window=window).sum()
    data[f"WVAD_{window}"] = weighted_data / summed_weights


def GARCH(data, col="Close"):
    returns = data[col].pct_change().dropna()
    model = arch_model(returns, vol="GARCH", p=1, q=1)
    model_fit = model.fit(disp="off")
    yhats = np.sqrt(model_fit.forecast(start=0).variance)
    data[f"GARCH_{col}"] = yhats


def relative(data, window=21, col="Volume"):
    data[f"Relative_{col}_{window}"] = data[col] / talib.SMA(data[col], window)
