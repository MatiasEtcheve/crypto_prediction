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


class PastAsset(object):
    def __init__(
        self, ticker, client, current_amount, current_value, earliest_date, interval
    ):
        self.ticker = ticker
        self.client = client
        self.current_amount = current_amount
        self.current_value = current_value
        self.earliest_date = earliest_date
        self.interval = interval

        self.trades = self.get_all_trades()
        self.orders = self.get_all_orders()
        self.klines = self.compute_klines()

    def _get_all(self, limit, object_to_call):
        if object_to_call == "trades":
            function_to_call = self.client.get_my_trades
            id = "id"
        elif object_to_call == "orders":
            function_to_call = self.client.get_all_orders
            id = "orderId"
        else:
            raise ValueError("")

        last_trades = function_to_call(symbol=self.ticker + "BUSD", limit=limit)
        last_trade_ids = [trade[id] for trade in last_trades]

        current_trades = last_trades
        while len(current_trades) == limit:
            earliest_date = current_trades[0]["time"]
            current_trades = function_to_call(
                symbol=self.ticker + "BUSD", limit=limit, endTime=earliest_date
            )
            if len(current_trades) != 0:
                last_trades.extend(
                    [
                        trade
                        for trade in current_trades
                        if trade[id] not in last_trade_ids
                    ]
                )
                last_trade_ids += [trade[id] for trade in current_trades]
        trades = pd.DataFrame.from_dict(last_trades)
        trades = trades.apply(pd.to_numeric, errors="ignore")
        trades["time"] = pd.to_datetime(trades["time"], unit="ms").dt.tz_localize(
            pytz.UTC
        )
        return trades

    def get_all_trades(self, limit=1000):
        return self._get_all(limit, "trades")

    def get_all_orders(self, limit=1000):
        return self._get_all(limit, "orders")

    def is_file_empty(self, file_name):
        if not file_name.is_file():
            return True
        with open(file_name, "r") as read_obj:
            one_char = read_obj.read(1)
            if one_char == "\n":
                return True
        return False

    def compute_klines(self):
        if len(self.trades) == 0:
            return []
        klines = get_data.download_klines(
            symbol=self.ticker,
            interval=self.interval,
            beginning_date=self.earliest_date,
            ending_date=datetime.now().replace(tzinfo=pytz.utc),
        )

        self.trades = self.trades.set_index(keys="time", drop=False)
        self.trades["amountAdded"] = self.trades["qty"] * (
            2 * self.trades["isBuyer"] - 1
        )
        trades = self.trades.groupby(by=self.trades.index).agg(
            price=("price", "mean"), amountAdded=("amountAdded", "sum")
        )
        trades["trades"] = 1
        klines["price"] = klines["Open"]
        klines["amountAdded"] = 0

        klines = pd.concat((klines, trades))
        klines = klines.sort_index()

        self.initial_amount = self.current_amount - klines["amountAdded"].cumsum()[-1]
        self.amount_added = klines["amountAdded"].cumsum()[-1]
        klines["amount"] = self.initial_amount + klines["amountAdded"].cumsum()
        klines["value"] = klines["amount"] * klines["price"]
        klines["ticker"] = self.ticker

        klines = klines[np.isnan(klines["trades"])].drop(labels="trades", axis=1)
        return klines


class PastQuote(PastAsset):
    def __init__(
        self,
        ticker,
        client,
        trades,
        current_amount,
        current_value,
        earliest_date,
        interval,
    ):
        self.ticker = ticker
        self.client = client
        self.current_amount = current_amount
        self.current_value = current_value
        self.earliest_date = earliest_date
        self.interval = interval

        self.trades = trades
        self.klines = self.compute_klines()

    def get_last_trades(self):
        raise NotImplementedError

    def get_last_orders(self):
        raise NotImplementedError

    def compute_klines(self):
        if len(self.trades) == 0:
            return []

        klines = get_data.download_klines(
            symbol=self.ticker,
            interval=self.interval,
            beginning_date=self.earliest_date,
            ending_date=datetime.now().replace(tzinfo=pytz.utc),
        )

        trades = self.trades.copy()

        trades["amountAdded"] = -trades["quoteQty"] * (2 * self.trades["isBuyer"] - 1)
        trades = trades.groupby(level=1).agg(amountAdded=("amountAdded", "sum"))
        trades["trades"] = 1

        klines["price"] = klines["Open"]
        klines["amountAdded"] = 0

        klines = pd.concat((klines, trades))
        klines = klines.sort_index()

        self.initial_amount = self.current_amount - klines["amountAdded"].cumsum()[-1]
        klines["amount"] = self.initial_amount + klines["amountAdded"].cumsum()
        klines["value"] = klines["amount"] * klines["price"]
        klines["ticker"] = self.ticker
        klines = klines[np.isnan(klines["trades"])].drop(labels="trades", axis=1)
        return klines
