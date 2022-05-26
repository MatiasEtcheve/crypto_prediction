from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
import pytz
import yfinance as yf

from query_datasets import blockchain, klines, sant, trends

FORMAT = "%d-%m-%Y"
"""Expected datetime format"""


def _save_type(
    data: pd.DataFrame,
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
    directory: Path = Path(__file__).resolve().parent,
    data_type: str = "unknown",
):
    filename = Path(directory) / "_".join(
        [
            "-".join(symbol) if isinstance(symbol, list) else symbol,
            data_type,
            interval,
            beginning_date.strftime(FORMAT),
            ending_date.strftime(FORMAT),
        ]
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    filename = str(filename) + ".csv"
    data.to_csv(filename)
    return filename


def fetch_and_save_type(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
    directory: Path = str(Path(__file__).resolve().parent),
    data_type: str = "unknown",
):
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
    klines = fetch_type(
        symbol,
        interval,
        beginning_date,
        ending_date,
        data_type=data_type,
    )
    filename = _save_type(
        klines,
        symbol,
        interval,
        beginning_date,
        ending_date,
        directory,
        data_type=data_type,
    )
    return filename


def select_type_from_file(
    beginning_date: datetime,
    ending_date: datetime,
    filename: Path,
    compute_metrics: Callable = lambda x: x,
):
    """
    Selects klines from a csv file. These klines open at `beginning_date` and close at `ending_date`.

    Args:
        beginning_date (datetime): open time
        ending_date (datetime): close time
        filename (str): filename of the csv file to open

    Returns:
        pd.DataFrame: dataframe containing klines
    """
    klines = pd.read_csv(filename)
    klines = klines.rename(columns={klines.columns[0]: "Datetime"})
    klines.loc[:, "Datetime"] = pd.to_datetime(klines["Datetime"], utc=True)
    klines = klines.set_index("Datetime", drop=True)
    klines = klines.loc[beginning_date:ending_date]
    return klines
    # return compute_metrics(klines)


def download_type(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
    compute_metrics: Callable = lambda x: x,
    directory: Path = Path(__file__).resolve().parent,
    data_type: str = "unknown",
):

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
    p = Path(directory).glob("*.csv")
    files = [x for x in p if x.is_file()]
    filenames = [x.stem.split("_") for x in files]
    df_files = pd.DataFrame(
        filenames, columns=["symbol", "data_type", "interval", "start_date", "end_date"]
    )
    df_files = df_files[df_files["data_type"] == data_type]
    df_files.loc[:, ["start_date", "end_date"]] = df_files[
        ["start_date", "end_date"]
    ].apply(lambda x: pd.to_datetime(x, format=FORMAT).dt.tz_localize("UTC"))

    beginning_date = beginning_date.replace(tzinfo=pytz.UTC)
    ending_date = ending_date.replace(tzinfo=pytz.UTC)
    symbol_to_string = "-".join(symbol) if isinstance(symbol, list) else symbol
    later_file = df_files[
        (df_files["symbol"] == symbol_to_string)
        & (df_files["interval"] == interval)
        & (df_files["end_date"] > ending_date)
    ]
    sooner_file = df_files[
        (df_files["symbol"] == symbol_to_string)
        & (df_files["interval"] == interval)
        & (df_files["start_date"] < beginning_date)
    ]

    perfect_file = df_files[
        (df_files["symbol"] == symbol_to_string)
        & (df_files["interval"] == interval)
        & (df_files["start_date"] <= beginning_date)
        & (df_files["end_date"] >= ending_date)
    ]
    useless_file = df_files[
        (df_files["symbol"] == symbol_to_string)
        & (df_files["interval"] == interval)
        & (df_files["start_date"] >= beginning_date)
        & (df_files["end_date"] <= ending_date)
    ]
    if not perfect_file.empty:
        filename = files[perfect_file.index[0]]
        return select_type_from_file(
            beginning_date, ending_date, filename, compute_metrics
        )

    new_beginning_date = beginning_date
    new_ending_date = ending_date
    if not sooner_file.empty or not later_file.empty:
        if not sooner_file.empty:
            new_beginning_date = min(
                new_beginning_date, sooner_file["start_date"].iloc[0]
            )
            filename = files[sooner_file.index[0]]
            Path.unlink(filename)
        if not later_file.empty:
            new_ending_date = max(new_ending_date, later_file["end_date"].iloc[0])
            filename = files[later_file.index[0]]
            Path.unlink(filename)
    for index in useless_file.index:
        filename = files[index]
        Path.unlink(filename)
    new_filename = fetch_and_save_type(
        symbol,
        interval,
        new_beginning_date,
        new_ending_date,
        directory,
        data_type=data_type,
    )
    return select_type_from_file(
        beginning_date, ending_date, new_filename, compute_metrics
    )


def fetch_type(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
    data_type: str = "unknown",
):
    if data_type == "klines":
        return fetch_klines(symbol, interval, beginning_date, ending_date)
    if data_type == "blockchain":
        return fetch_blockchain(symbol, interval, beginning_date, ending_date)
    if data_type == "trends":
        return fetch_trends(symbol, interval, beginning_date, ending_date)
    if data_type == "santiment":
        return fetch_santiment(symbol, interval, beginning_date, ending_date)
    else:
        raise NotImplementedError("")


fetch_klines = klines.fetch_klines
download_klines = partial(download_type, data_type="klines")

fetch_blockchain = blockchain.fetch_blockchain
download_blockchain = partial(download_type, data_type="blockchain")

fetch_trends = trends.fetch_trends
download_trends = partial(download_type, data_type="trends")

fetch_santiment = sant.fetch_santiment
download_santiment = partial(download_type, data_type="santiment")


if __name__ == "__main__":
    beginning_date = datetime(2015, 12, 12)
    ending_date = datetime(2017, 12, 25)
    _klines = download_klines(
        "BTC",
        "1d",
        beginning_date=beginning_date,
        ending_date=ending_date,
        directory="tmp/",
    )
    _klines = _klines.dropna(axis=0)
    print(_klines)
    print(len(_klines))

    _blockchain = download_blockchain(
        "BTC",
        "1d",
        beginning_date=beginning_date,
        ending_date=ending_date,
        directory="tmp/",
    )
    _blockchain = _blockchain.dropna(axis=0)
    print(_blockchain)
    print(len(_blockchain))

    _trends = download_trends(
        "BTC",
        "1d",
        beginning_date=beginning_date,
        ending_date=ending_date,
        directory="tmp/",
    )
    _trends = _trends.dropna(axis=0)
    print(_trends)
    print(len(_trends))

    _santiment = download_santiment(
        "BTC",
        "1d",
        beginning_date=beginning_date,
        ending_date=ending_date,
        directory="tmp/",
    )
    _santiment = _santiment.dropna(axis=0)
    print(_santiment)
    print(len(_santiment))
