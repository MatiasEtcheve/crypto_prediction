from pandas.core.arrays.sparse import dtype
import config
import csv
import numpy as np
import os.path
from binance.client import Client
from datetime import datetime, timedelta
from datetime import timedelta
from os import listdir, getcwd, remove
import json
import pandas as pd

# format = "%d-%m-%Y"


def tsConverter(ts, format='%d-%m-%Y'):
    ts = int(ts)
    if ts > 9999999999:
        ts /= 1000
    return datetime.utcfromtimestamp(ts).strftime(format)


def timeStampConverter(ts, format='%d-%m-%Y'):
    if isinstance(ts, list) or isinstance(ts, np.ndarray):
        if isinstance(ts[0], list) or isinstance(ts[0], np.ndarray):
            times = []
            for time in ts:
                times.append(tsConverter(time[0]))
            return times
        else:
            return tsConverter(ts[0], format=format)
    else:
        return tsConverter(ts, format=format)


def getFileNameCandles(symbol, interval, from_date, end_date, format='%d-%m-%Y'):
    return f"data/{symbol}_{interval}_{from_date.strftime('%d-%m-%Y')}_{end_date.strftime('%d-%m-%Y')}.csv"


def getFiles(path=None):
    if path is None:
        path = getcwd()
    list_files = listdir(path+"/data/")
    data = []
    for index, file in enumerate(list_files):
        key_list = ["symbol", "interval", "from_date", "end_date"]
        list = file.split(".")[0].split("_")
        zip_iterator = zip(key_list, list)
        data.append(dict(zip_iterator))
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['from_date'] = pd.to_datetime(df['from_date'], format='%d-%m-%Y')
    # df['from_date'] = df['from_date'].dt.strftime('%d-%m-%Y')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%d-%m-%Y')
    # df['end_date'] = df['end_date'].dt.strftime('%d-%m-%Y')
    return df


def getCorrespondingFiles(data, symbol, interval, from_date, end_date, format='%d-%m-%Y'):
    if isinstance(from_date, str):
        from_date = datetime.strptime(from_date, format)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, format)
    index_symbol = data["symbol"] == symbol
    index_interval = data["interval"] == interval
    index_from_date = data["from_date"] <= from_date
    index_end_date = data["end_date"] >= end_date
    data_selected = data[index_symbol & index_interval]
    data_inside = data_selected[(data["from_date"] > from_date) & (data["end_date"] < end_date)]
    return data_selected[index_from_date & index_end_date], data_selected[index_from_date], data_selected[index_end_date], data_inside


client = Client(config.API_KEY, config.API_SECRET)
info = client.get_all_tickers()
info.sort(key=lambda x: x["symbol"], reverse=True)
info = [i for i in info if "USDT" in i["symbol"] or "USDC" in i["symbol"]]


def get_candles(symbol, interval, from_date, end_date):
    candles = client.get_historical_klines(symbol,
                                           interval,
                                           from_date.strftime('%d %b, %Y'),
                                           end_date.strftime('%d %b, %Y'))
    return candles


def get_data(symbol, interval, from_date, end_date, verbose=True):
    data = getFiles()
    if end_date > datetime.today(): 
        end_date = datetime().date.today()
    if data.empty:
        candles = np.array(get_candles(symbol, interval, from_date, end_date), dtype=float)
        file_name = getFileNameCandles(symbol, interval, from_date, end_date)
    else:
        data_outside, data_before, data_after, data_inside = getCorrespondingFiles(data, symbol, interval, from_date, end_date)
        for i in range(len(data_inside.index)):
            row = data_inside.iloc[0]
            file_name = getFileNameCandles(row["symbol"], row["interval"], row["from_date"], row["end_date"])
            remove(file_name)
        if not data_outside.empty:
            row = data_outside.iloc[0]
            return getFileNameCandles(row["symbol"], row["interval"], row["from_date"], row["end_date"])
        else:
            if not data_before.empty:
                row = data_before.iloc[0]
                file_name_before = getFileNameCandles(row["symbol"], row["interval"], row["from_date"], row["end_date"])
                data_before = np.genfromtxt(file_name_before, delimiter=",")
                data_now = np.array(get_candles(symbol, interval, row["end_date"], end_date))
                if data_before.shape == (0, ): candles = data_now
                else: candles = np.concatenate((data_before, data_now), axis=0)
                candles = np.concatenate((data_before, data_now), axis=0)             
                candles = candles[np.argsort(candles[:, 0]), :]
                _, indices = np.unique(candles[:, 0], return_index=True)
                candles = candles[indices, :]
                file_name = getFileNameCandles(symbol, interval, row["from_date"], end_date)
                remove(file_name_before)

            elif not data_after.empty: 
                row = data_after.iloc[0]
                file_name_after = getFileNameCandles(row["symbol"], row["interval"], row["from_date"], row["end_date"])
                data_after = np.genfromtxt(file_name_after, delimiter=",")
                data_now = np.array(get_candles(symbol, interval, from_date, row["from_date"]), dtype=float)
                if data_now.shape == (0, ): candles = data_after
                else: candles = np.concatenate((data_after, data_now), axis=0)
                candles = candles[np.argsort(candles[:, 0]), :]
                _, indices = np.unique(candles[:, 0], return_index=True)
                candles = candles[indices, :]
                file_name = getFileNameCandles(symbol, interval, from_date, row["end_date"])
                remove(file_name_after)

            else:
                candles = np.array(get_candles(symbol, interval, from_date, end_date), dtype=float)
                file_name = getFileNameCandles(symbol, interval, from_date, end_date)
    file = open(file_name, "w", newline="")
    candlestick_writer = csv.writer(file, delimiter=',')
    for candle in candles:
        candle[0] = int(float(candle[0]))
        candlestick_writer.writerow(candle)

    if verbose:
        print(
            f"{len(candles)} candlesticks for {interval} intervals, between {timeStampConverter(candles[0])} to {timeStampConverter(candles[-1])}")
    file.close()
    return file_name


from_date = datetime(2006, 8, 4)
end_date = datetime(2021, 4, 30)
interval = client.KLINE_INTERVAL_12HOUR
symbol = "DOGEUSDT"
print(get_data(symbol, interval, from_date, end_date))

