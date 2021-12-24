from datetime import datetime, timedelta, date
from pathlib import Path
from time import time
import re
from typing import Annotated
import pandas as pd
import vectorbt as vbt
import os

import pytz

FORMAT = "%Y-%m-%d"
"""Expected datetime format"""
complete_time = {"d": "days"}
KLINE_COLUMNS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]
"""Kline columns of dataframe"""


def select_klines_from_file(beginning_date, ending_date, filename):
    """
    Selects klines from a csv file. These klines open at `beginning_date` and close at `ending_date`.

    Args:
        beginning_date (datetime): open time
        ending_date (datetime): close time
        filename (str): filename of the csv file to open

    Returns:
        pd.DataFrame: dataframe containing klines
    """
    klines = vbt.Data.load(filename)
    return klines.loc[beginning_date:ending_date]

def download_and_save_klines(symbol, interval, beginning_date, ending_date):
    """
    Downloads klines of `symbol` from `from_date` to `to_date`, at interval `interval`.

    Args:
        symbol (str): ticker to download eg `BTCUSDT`
        interval (str): interval of klines, eg `6h`
        beginning_date (datetime): open time
        ending_date (datetime): close time

    Returns:
        str: filename of csv file containing the klines
    """
    klines = vbt.BinanceData.download(symbol, start=beginning_date, end=ending_date, interval=interval)
    filename = (
        "candlesticks/"
        + "_".join(
            [
                "-".join(symbol),
                interval,
                beginning_date.strftime(FORMAT),
                ending_date.strftime(FORMAT),
            ]
        )
        + ".pickle"
    )
    klines.save(fname=filename)
    return filename


def select_data(symbol, interval, beginning_date, ending_date):
    """
    Selects klines of `symbol` from `from_date` to `to_date`, at interval `interval`.
    If the klines have already been downloaded, it fetches it in the csv file.
    Otherwise, it downloads the data from the Binance API.

    Args:
        symbol (str): ticker to download eg `BTCUSDT`
        interval (str): interval of klines, eg `6h`
        beginning_date (datetime): open time
        ending_date (datetime): close time

    Returns:
        pd.DataFrame: dataframe containing klines
    """
    p = (Path(__file__).resolve().parent / "candlesticks").glob("**/*")
    files = [x for x in p if x.is_file()]
    filenames = [x.stem.split("_") for x in files]
    df_files = pd.DataFrame(
        filenames, columns=["symbol", "interval", "start_date", "end_date"]
    )
    df_files[["start_date", "end_date"]] = df_files[["start_date", "end_date"]].apply(
        lambda x: pd.to_datetime(x, format=FORMAT).dt.tz_localize('UTC')
    )

    if isinstance(symbol, str):
        symbol = [symbol]
    symbol_to_string = "-".join(symbol)

    beginning_date = datetime(beginning_date.year, beginning_date.month, beginning_date.day, tzinfo=pytz.utc)
    ending_date = datetime(ending_date.year, ending_date.month, ending_date.day,  tzinfo=pytz.utc)

    later_file = df_files[
        (df_files["symbol"] == symbol_to_string) & (df_files["interval"] == interval) & (df_files["end_date"] > ending_date)
    ]
    sooner_file = df_files[
        (df_files["symbol"] == symbol_to_string) & (df_files["interval"] == interval) & (df_files["start_date"] < beginning_date)
    ]

    perfect_file = df_files[
        (df_files["symbol"] == symbol_to_string) & (df_files["interval"] == interval)
        & (
            df_files["start_date"]
            <= beginning_date
        )
        & (
            df_files["end_date"]
            >= ending_date
        )
    ]
    useless_file = df_files[
        (df_files["symbol"] == symbol_to_string) & (df_files["interval"] == interval)
        & (df_files["start_date"] >= beginning_date)
        & (df_files["end_date"] <= ending_date)
    ]

    if not perfect_file.empty:
        filename = files[perfect_file.index[0]]
        return select_klines_from_file(beginning_date, ending_date, filename)
    elif not sooner_file.empty or not later_file.empty:
        if not sooner_file.empty:
            beginning_date = min(beginning_date, sooner_file["start_date"].iloc[0])
            filename = files[sooner_file.index[0]]
            Path.unlink(filename)
        if not later_file.empty:
            ending_date = max(ending_date, later_file["end_date"].iloc[0])
            filename = files[later_file.index[0]]
            Path.unlink(filename)
    for index in useless_file.index:
        filename = files[index]
        Path.unlink(filename)
    new_filename = download_and_save_klines(
        symbol,
        interval,
        beginning_date,
        ending_date,
    )
    return select_klines_from_file(beginning_date, ending_date, new_filename)

if __name__ == '__main__':
    klines = select_data(["BTCUSDT", "ETHUSDT"], "30m", beginning_date=datetime.now()-timedelta(days=365), ending_date=datetime.now())
    klines.plot(column='Close', base=1).show()