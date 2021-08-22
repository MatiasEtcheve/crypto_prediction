import csv
from datetime import datetime
from pathlib import Path
import os
import re
import numpy as np
import pandas as pd
from binance.client import Client
from pprint import pprint
import config
import tools.dict_tools as dict_tools

client = Client(config.API_KEY, config.API_SECRET)

TICKERS = client.get_all_tickers()
SYMBOLS = [ticker["symbol"] for ticker in TICKERS]
FORMAT = "%d-%m-%Y"
# "%Y-%m-%dT%H:%M:%S.%f%z"
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


def select_klines_from_file(beginning_date, ending_date, filename):
    data = pd.read_csv(filename, delimiter=",", header=0, dtype=str)
    readable_datetime = pd.to_datetime(data["Open time"], unit="ms")
    return data[
        (readable_datetime >= beginning_date) & (ending_date >= readable_datetime)
    ]


def download_and_save_klines(symbol, interval, from_date, to_date):
    klines = client.get_historical_klines(
        symbol, interval, from_date.strftime(FORMAT), to_date.strftime(FORMAT),
    )
    data = pd.DataFrame(klines, columns=KLINE_COLUMNS)
    readable_datetime = pd.to_datetime(data["Open time"], unit="ms")
    beginning_date = readable_datetime.iloc[0]
    ending_date = readable_datetime.iloc[-1]
    filename = (
        "data/"
        + "_".join(
            [
                symbol,
                interval,
                max(from_date, beginning_date).strftime(FORMAT),
                min(to_date, ending_date).strftime(FORMAT),
            ]
        )
        + ".csv"
    )
    data.to_csv(filename, sep=",", header=KLINE_COLUMNS, index=False)
    return filename


def select_data(symbol, interval, beginning_date, ending_date):
    p = (Path(__file__).resolve().parent / "data").glob("**/*")
    files = [x for x in p if x.is_file()]
    for file in files:
        filename = file.name
        [current_symbol, current_interval, from_date, to_date] = filename.removesuffix(
            ".csv"
        ).split("_")
        from_date = datetime.strptime(from_date, FORMAT)
        to_date = datetime.strptime(to_date, FORMAT)
        if current_symbol != symbol or current_interval != interval:
            continue
        if from_date <= beginning_date or to_date <= ending_date:
            Path.unlink(file)
            new_filename = download_and_save_klines(
                symbol,
                interval,
                min(from_date, beginning_date),
                max(to_date, ending_date),
            )
            return select_klines_from_file(beginning_date, ending_date, new_filename)

        if from_date < beginning_date and ending_date < to_date:
            return select_klines_from_file(beginning_date, ending_date, file.stem)

    filename = download_and_save_klines(symbol, interval, beginning_date, ending_date)
    return select_klines_from_file(beginning_date, ending_date, filename)


if __name__ == "__main__":
    beginning_date = datetime.strptime("28-01-2018", FORMAT)
    ending_date = datetime.strptime("28-01-2021", FORMAT)
    select_data("DOGEUSDT", "1w", beginning_date, ending_date)
