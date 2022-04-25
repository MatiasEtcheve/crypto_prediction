from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytz
import yfinance as yf

# import vectorbt as vbt

# FORMAT = "%Y-%m-%d"
FORMAT = "%d-%m-%Y"
"""Expected datetime format"""


def select_klines_from_file(beginning_date, ending_date, filename, type="csv"):
    """
    Selects klines from a csv file. These klines open at `beginning_date` and close at `ending_date`.

    Args:
        beginning_date (datetime): open time
        ending_date (datetime): close time
        filename (str): filename of the csv file to open

    Returns:
        pd.DataFrame: dataframe containing klines
    """
    if type == "csv":
        klines = pd.read_csv(filename)
        klines = klines.rename(columns={klines.columns[0]: "Datetime"})
        klines["Datetime"] = pd.to_datetime(klines["Datetime"], utc=True)
        mask = (klines["Datetime"] >= beginning_date) & (
            klines["Datetime"] <= ending_date
        )
        return klines.loc[mask]
    elif type == "vbt":
        # klines = vbt.Data.load(filename)
        return klines.loc[beginning_date:ending_date]


def download_klines(
    symbol, interval, beginning_date, ending_date, compute_metrics=None, type="csv"
):
    if type == "vbt":
        pass
        # klines = vbt.BinanceData.download(
        #     [s + "USDT" for s in symbol],
        #     start=beginning_date,
        #     end=ending_date,
        #     interval=interval,
        # )
    elif type == "csv":
        assert (
            isinstance(symbol, str) or len(symbol) == 1
        ), f"Symbol can't be a list in a csv, but it is {symbol}"
        klines = yf.download(
            tickers=symbol + "-USD",
            start=beginning_date,
            end=ending_date,
            interval=interval,
            progress=True,
            show_errors=True,
        )
        try:
            klines.index = klines.index.tz_convert(pytz.UTC).rename("Datetime")
        except TypeError as e:
            klines.index = klines.index.tz_localize(pytz.UTC).rename("Datetime")
        except AttributeError as e:
            pass
        klines = klines.drop(
            labels=["Adj Close"],
            axis=1,
        )
        klines = klines.astype("float64")
        if True or klines.empty:
            from binance.client import Client

            client = Client()
            klines = client.get_historical_klines(
                symbol + "USDT",
                interval,
                str(beginning_date.timestamp() * 1000),
                str(ending_date.timestamp() * 1000),
            )
            klines = pd.DataFrame(
                klines,
                columns=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "close_time",
                    "qav",
                    "num_trades",
                    "taker_base_vol",
                    "taker_quote_vol",
                    "ignore",
                ],
            ).drop(
                labels=[
                    "close_time",
                    "qav",
                    "num_trades",
                    "taker_base_vol",
                    "taker_quote_vol",
                    "ignore",
                ],
                axis=1,
            )

            klines["Datetime"] = pd.to_datetime(
                klines["Datetime"], unit="ms"
            ).dt.tz_localize(pytz.UTC)
            klines = klines.set_index("Datetime")
            klines = klines.astype("float32")
        if compute_metrics is not None:
            klines = compute_metrics(klines)
    return klines


def save_klines(
    klines,
    symbol,
    interval,
    beginning_date,
    ending_date,
    directory=str(Path(__file__).resolve().parent),
    type="csv",
):

    filename = str(directory) + "_".join(
        [
            "-".join(symbol) if isinstance(symbol, list) else symbol,
            interval,
            beginning_date.strftime(FORMAT),
            ending_date.strftime(FORMAT),
        ]
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if type == "vbt":
        filename += ".pkl"
        klines.save(fname=filename)
    elif type == "csv":
        filename += ".csv"
        klines.to_csv(filename)
    return filename


def download_and_save_klines(
    symbol,
    interval,
    beginning_date,
    ending_date,
    compute_metrics=None,
    directory=str(Path(__file__).resolve().parent),
    type="csv",
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
    klines = download_klines(
        symbol,
        interval,
        beginning_date,
        ending_date,
        compute_metrics,
        type,
    )
    filename = save_klines(
        klines,
        symbol,
        interval,
        beginning_date,
        ending_date,
        directory,
        type,
    )
    return filename


def select_data(
    symbol,
    interval,
    beginning_date,
    ending_date,
    compute_metrics=None,
    directory=Path(__file__).resolve().parent,
    type="csv",
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
    if type == "csv":
        p = Path(directory).glob("**/*.csv")
    else:
        p = Path(directory).glob("**/*.pkl")
    files = [x for x in p if x.is_file()]
    filenames = [x.stem.split("_") for x in files]
    df_files = pd.DataFrame(
        filenames, columns=["symbol", "interval", "start_date", "end_date"]
    )
    df_files[["start_date", "end_date"]] = df_files[["start_date", "end_date"]].apply(
        lambda x: pd.to_datetime(x, format=FORMAT).dt.tz_localize("UTC")
    )

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
        return select_klines_from_file(beginning_date, ending_date, filename, type)
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
        symbol, interval, beginning_date, ending_date, compute_metrics, directory, type
    )
    return select_klines_from_file(beginning_date, ending_date, new_filename, type)


if __name__ == "__main__":
    # klines = select_data(
    #     "LTC",
    #     "1m",
    #     beginning_date=datetime(2022, 4, 5),
    #     ending_date=datetime(2022, 4, 6),
    #     directory="whateverthefuck/",
    #     type="csv",
    # )
    # klines = klines.dropna(axis=0)
    from binance.client import Client

    client = Client()
    klines = client.get_historical_klines(
        "LTCUSDT",
        "1m",
        str(datetime(2022, 4, 5).timestamp() * 1000),
        str(datetime(2022, 4, 6).timestamp() * 1000),
    )
    print(len(klines))
