from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import talib
import yfinance as yf

# FORMAT = "%Y-%m-%d"
FORMAT = "%d-%m-%Y"
"""Expected datetime format"""


def concatenate_indicators(data):
    differences = data.diff(periods=1, axis=0).rename(
        columns={
            "Open": "Opend",
            "High": "Highd",
            "Low": "Lowd",
            "Close": "Closed",
            "Volume": "Volumed",
        }
    )
    pct_changes = data.pct_change(periods=1, axis=0).rename(
        columns={
            "Open": "Openp",
            "High": "Highp",
            "Low": "Lowp",
            "Close": "Closep",
            "Volume": "Volumep",
        }
    )
    SMA5 = talib.SMA(data["Close"], timeperiod=5)
    SMA10 = talib.SMA(data["Close"], timeperiod=10)
    SMA20 = talib.SMA(data["Close"], timeperiod=20)
    EMA12 = talib.EMA(data["Close"], timeperiod=12)
    EMA26 = talib.EMA(data["Close"], timeperiod=26)
    MACD, MACDsign, _ = talib.MACD(
        data["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    ROC13 = talib.ROC(data["Close"], timeperiod=13)
    K15, D5 = talib.STOCHF(
        data["High"],
        data["Low"],
        data["Close"],
        fastk_period=15,
        fastd_period=5,
        fastd_matype=0,
    )
    BOLup, _, BOLlow = talib.BBANDS(
        data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
    )
    BOL = (data["Close"] - BOLlow) / (BOLup - BOLlow)
    MOM12 = talib.MOM(data["Close"], timeperiod=12)
    indicators = {
        "SMA5": SMA5,
        "SMA10": SMA10,
        "SMA20": SMA20,
        "EMA12": EMA12,
        "EMA26": EMA26,
        "MACD": MACD,
        "MACDsign": MACDsign,
        "ROC13": ROC13,
        "K15": K15,
        "D5": D5,
        "BOLup": BOLup,
        "BOLlow": BOLlow,
        "BOL": BOL,
        "MOM12": MOM12,
    }
    indicators_df = pd.concat(
        indicators.values(),
        axis=1,
        keys=indicators.keys(),
    )
    klines = pd.concat([data, differences, pct_changes, indicators_df], axis=1)
    return klines


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
    klines = pd.read_csv(filename)
    klines = klines.rename(columns={klines.columns[0]: "Datetime"})
    klines["Datetime"] = pd.to_datetime(klines["Datetime"], utc=True)
    mask = (klines["Datetime"] >= beginning_date) & (klines["Datetime"] <= ending_date)
    return klines.loc[mask]


def download_and_save_klines(
    symbol,
    interval,
    beginning_date,
    ending_date,
    directory=str(Path(__file__).resolve().parent),
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
    klines = yf.download(
        tickers=symbol + "-USD",
        start=beginning_date,
        end=ending_date,
        interval=interval,
        auto_adjust=True,
        progress=True,
        show_errors=False,
    )

    if klines.empty:
        from binance.client import Client

        client = Client()
        klines = client.get_historical_klines(
            symbol + "USDT",
            interval,
            beginning_date.strftime(FORMAT),
            ending_date.strftime(FORMAT),
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
        klines["Datetime"] = pd.to_datetime(klines["Datetime"], unit="ms")
        klines = klines.set_index("Datetime")
        klines = klines.astype("float64")
    filename = (
        directory
        + "/candlesticks/"
        + "_".join(
            [
                symbol,
                interval,
                beginning_date.strftime(FORMAT),
                ending_date.strftime(FORMAT),
            ]
        )
        + ".txt"
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    klines = concatenate_indicators(klines)
    klines.to_csv(filename)
    return filename


def select_data(
    symbol,
    interval,
    beginning_date,
    ending_date,
    directory=Path(__file__).resolve().parent,
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
    p = (Path(directory) / "candlesticks").glob("**/*")
    files = [x for x in p if x.is_file()]
    filenames = [x.stem.split("_") for x in files]
    df_files = pd.DataFrame(
        filenames, columns=["symbol", "interval", "start_date", "end_date"]
    )
    df_files[["start_date", "end_date"]] = df_files[["start_date", "end_date"]].apply(
        lambda x: pd.to_datetime(x, format=FORMAT).dt.tz_localize("UTC")
    )

    # if isinstance(symbol, str):
    #     symbol = [symbol]
    # symbol_to_string = "-".join(symbol)

    beginning_date = datetime(
        beginning_date.year, beginning_date.month, beginning_date.day, tzinfo=pytz.utc
    )
    ending_date = datetime(
        ending_date.year, ending_date.month, ending_date.day, tzinfo=pytz.utc
    )

    later_file = df_files[
        (df_files["symbol"] == symbol)
        & (df_files["interval"] == interval)
        & (df_files["end_date"] > ending_date)
    ]
    sooner_file = df_files[
        (df_files["symbol"] == symbol)
        & (df_files["interval"] == interval)
        & (df_files["start_date"] < beginning_date)
    ]

    perfect_file = df_files[
        (df_files["symbol"] == symbol)
        & (df_files["interval"] == interval)
        & (df_files["start_date"] <= beginning_date)
        & (df_files["end_date"] >= ending_date)
    ]
    useless_file = df_files[
        (df_files["symbol"] == symbol)
        & (df_files["interval"] == interval)
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
        directory,
    )
    return select_klines_from_file(beginning_date, ending_date, new_filename)


if __name__ == "__main__":
    klines = select_data(
        "BTC",
        "1d",
        beginning_date=datetime.now() - timedelta(days=49),
        ending_date=datetime.now(),
        directory="whateverthefuck/",
    )
