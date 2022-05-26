from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import pytz
from tools.dataframe import convert_to_timedelta

# FORMAT = "%Y-%m-%d"
FORMAT = "%d-%m-%Y"
"""Expected datetime format"""


def fetch_klines(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
):
    import yfinance as yf

    assert (
        isinstance(symbol, str) or len(symbol) == 1
    ), f"Symbol can't be a list in a csv, but it is {symbol}"
    klines = yf.download(
        tickers=symbol + "-USD",
        start=beginning_date + convert_to_timedelta(interval, ago=1),
        end=ending_date + convert_to_timedelta(interval, ago=1),
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
    if klines.empty:
        from binance.client import Client

        client = Client()
        klines = client.get_historical_klines(
            symbol + "USDT",
            interval,
            str(
                (beginning_date - convert_to_timedelta(interval, ago=1)).timestamp()
                * 1000
            ),
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
                # "num_trades",
                # "taker_base_vol",
                # "taker_quote_vol",
                "ignore",
            ],
            axis=1,
        )

        klines.loc[:, "Datetime"] = pd.to_datetime(
            klines["Datetime"], unit="ms"
        ).dt.tz_localize(pytz.UTC)
        klines = klines.set_index("Datetime")
        klines = klines.astype("float32")
    return klines
