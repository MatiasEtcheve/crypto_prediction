import os
import sys
from calendar import month
from datetime import date, datetime, timedelta
from importlib import reload
from pathlib import Path
from pprint import pprint
from time import time

import get_data
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import quantstats as qs
import streamlit as st
from binance.client import Client

from past_asset import PastAsset, PastQuote


class PastPortfolio(object):
    def __init__(
        self,
        client,
        interval,
    ):
        self.client = client
        self.interval = interval

        (
            self.current_amounts,
            self.current_values,
        ) = self.get_current_portfolio_amount_value()

        self.tickers = ["ETH", "BTC", "BNB", "LTC", "XRP", "TRX"]

        self.earliest_date = self.compute_earliest_date()

        self.assets = {
            ticker: PastAsset(
                ticker,
                self.client,
                self.current_amounts[ticker],
                self.current_values[ticker],
                self.earliest_date,
                self.interval,
            )
            for ticker in self.tickers
        }

        self.trades = pd.concat(
            [asset.trades for asset in self.assets.values()],
            keys=self.tickers,
            axis=0,
        )
        self.orders = pd.concat(
            [asset.orders for asset in self.assets.values()],
            keys=self.tickers,
            axis=0,
        )
        self.quote = PastQuote(
            "BUSD",
            self.client,
            self.trades,
            self.current_amounts["BUSD"],
            self.current_values["BUSD"],
            self.earliest_date,
            self.interval,
        )
        common_dates = set.intersection(
            *[set(asset.klines.index) for asset in self.assets.values()],
            set(self.quote.klines.index),
        )

        klines = pd.concat(
            [
                asset.klines.loc[common_dates].sort_index()
                for asset in self.assets.values()
            ]
            + [self.quote.klines.loc[common_dates].sort_index()],
            keys=self.tickers + ["BUSD"],
            axis=0,
        )

        somme = klines.groupby(level=1).agg(
            amount=("amount", "sum"), value=("value", "sum")
        )
        self._klines = pd.concat(
            [
                pd.concat(
                    [somme],
                    keys=["SUM"],
                ),
                klines,
            ]
        )

        self._returns = self._klines.loc["SUM"]["value"].pct_change()

    def _cast_date(self, shitty_date):
        if isinstance(shitty_date, date):
            return datetime(shitty_date.year, shitty_date.month, shitty_date.day)
        return shitty_date.replace(tzinfo=pytz.UTC)

    def klines(self, beginning_date, ending_date=datetime.now()):
        if beginning_date is None:
            beginning_date = self.earliest_date

        beginning_date = self._cast_date(beginning_date)
        ending_date = self._cast_date(ending_date)

        self._klines = self._klines.sort_index()
        return self._klines.loc[pd.IndexSlice[:, beginning_date:ending_date], :]

    def returns(self, beginning_date, ending_date=datetime.now()):
        if beginning_date is None:
            beginning_date = self.earliest_date

        beginning_date = self._cast_date(beginning_date)
        ending_date = self._cast_date(ending_date)

        self._returns = self._returns.sort_index()
        return self._returns.loc[beginning_date:ending_date]

    def compute_earliest_date(self):
        trades = pd.concat(
            [
                pd.DataFrame.from_dict(
                    self.client.get_my_trades(symbol=ticker + "BUSD")
                ).apply(pd.to_numeric, errors="ignore")
                for ticker in self.tickers
            ]
        )
        trades["time"] = pd.to_datetime(trades["time"], unit="ms").dt.tz_localize(
            pytz.UTC
        )
        return trades["time"].min()

    def get_current_portfolio_amount_value(self):
        amount = {}
        value = {}
        for balance in self.client.get_account()["balances"]:
            if balance["asset"] in ["BUSD", "USDT"]:
                value[balance["asset"]] = float(balance["free"]) + float(
                    balance["locked"]
                )
            else:
                avg_price = float(
                    self.client.get_avg_price(symbol=balance["asset"] + "USDT")["price"]
                )
                value[balance["asset"]] = (
                    float(balance["free"]) + float(balance["locked"])
                ) * avg_price
            amount[balance["asset"]] = float(balance["free"]) + float(balance["locked"])
        return amount, value
