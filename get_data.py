from datetime import datetime, timedelta, date
from pathlib import Path
from time import time
import re
from typing import Annotated
import pandas as pd

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
    data = pd.read_csv(filename, delimiter=",", header=0, index_col=0)
    data.index = pd.to_datetime(data.index)
    return data.loc[beginning_date:ending_date]


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
    from binance.client import Client

    api_key = os.getenv("BINANCE_API_KEY", None)
    api_secret = os.getenv("BINANCE_API_SECRET", None)
    client = Client(api_key, api_secret)
    print("downloading klines...", end="")
    klines = client.get_historical_klines(
        symbol,
        interval,
        beginning_date.strftime(FORMAT),
        ending_date.strftime(FORMAT),
    )
    print("done")
    data = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    filename = (
        "candlesticks/"
        + "_".join(
            [
                symbol,
                interval,
                beginning_date.strftime(FORMAT),
                ending_date.strftime(FORMAT),
            ]
        )
        + ".csv"
    )
    data["Open time"] = pd.to_datetime(data["Open time"], unit="ms")
    data["Close time"] = pd.to_datetime(data["Close time"], unit="ms")
    data.set_index("Open time", inplace=True, drop=True)
    data.to_csv(filename, sep=",", header=data.columns, index=True)
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
        filenames, columns=["ticker", "interval", "start_date", "end_date"]
    )
    df_files[["start_date", "end_date"]] = df_files[["start_date", "end_date"]].apply(
        lambda x: pd.to_datetime(x, format=FORMAT)
    )

    beginning_date = datetime(beginning_date.year, beginning_date.month, beginning_date.day)
    ending_date = datetime(ending_date.year, ending_date.month, ending_date.day)

    later_file = df_files[
        (df_files["ticker"] == symbol) & (df_files["interval"] == interval) & (df_files["end_date"] > ending_date)
    ]
    sooner_file = df_files[
        (df_files["ticker"] == symbol) & (df_files["interval"] == interval) & (df_files["start_date"] < beginning_date)
    ]

    perfect_file = df_files[
        (df_files["ticker"] == symbol) & (df_files["interval"] == interval)
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
        (df_files["ticker"] == symbol) & (df_files["interval"] == interval)
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
