from datetime import datetime

import pandas as pd
import pytrends
import pytz
from pytrends.dailydata import get_daily_data
from pytrends.request import TrendReq

format = "%Y-%m-%d"


def fetch_trends(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
):
    # assert interval == "1d"

    beginning_date = beginning_date.replace(tzinfo=pytz.UTC)
    ending_date = ending_date.replace(tzinfo=pytz.UTC)

    trends = get_daily_data(
        symbol,
        start_year=beginning_date.year,
        start_mon=beginning_date.month,
        stop_year=ending_date.year,
        stop_mon=ending_date.month,
        geo="",
        verbose=False,
    )
    trends.index = pd.to_datetime(trends.index).tz_localize(pytz.UTC)
    trends.index.names = ["Datetime"]
    trends = trends.astype("float32")
    return trends.loc[beginning_date:ending_date, symbol].to_frame(symbol)
