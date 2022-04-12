import sys
from calendar import month
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules
if IS_KAGGLE:
    repo_path = Path("../input/crypto-prediction")
elif IS_COLAB:
    from google.colab import drive

    drive.mount("/content/gdrive")
    repo_path = Path("/content/gdrive/MyDrive/crypto-prediction")
else:
    repo_path = Path("/home/matias/crypto-prediction")
sys.path.append(str(repo_path))

import os
from datetime import date, datetime, timedelta
from importlib import reload
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

qs.extend_pandas()

api_key = os.environ.get("TESTNET_API")
api_secret = os.environ.get("TESTNET_SECRET")
client = Client(api_key, api_secret, testnet=True)


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

        self.tickers = [
            "ETH", "BTC", "BNB", "LTC", "XRP", "TRX"
        ]

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
        for balance in client.get_account()["balances"]:
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


col1, col2 = st.columns(2)

with col1:
    period = st.radio(
        label="Select period",
        options=["Custom", "Last day", "Last week", "Last month", "Last year", "All"],
        index=0,
        key="period",
    )

with col2:
    interval = st.radio(
        label="Select interval",
        options=["1m", "5m", "30m", "1h", "6h", "12h", "1d"],
        index=6,
        key="interval",
    )

col1, col2 = st.columns(2)
with col1:
    if st.session_state["period"] == "Custom":
        beginning_date = st.date_input(
            label="Beginning date",
            value=date(2022, 4, 7),
            key="beginning_date",
        )

with col2:
    if st.session_state["period"] == "Custom":
        ending_date = st.date_input(
            label="Ending date",
            value=datetime.now(),
            key="ending_date",
        )
    else:
        ending_date = datetime.now()

if st.session_state["period"] == "Last day":
    beginning_date = ending_date - timedelta(days=1)
if st.session_state["period"] == "Last week":
    beginning_date = ending_date - timedelta(days=7)
if st.session_state["period"] == "Last month":
    beginning_date = ending_date - timedelta(days=31)
if st.session_state["period"] == "Last year":
    beginning_date = ending_date - timedelta(days=365)
if st.session_state["period"] == "All":
    beginning_date = None


@st.experimental_memo
def load_portfolio(interval):
    return PastPortfolio(client, interval)


pf = load_portfolio(st.session_state["interval"])
st.write("First trade at:", pf.trades.index.get_level_values(1).min())
st.write("Last trade at:", pf.trades.index.get_level_values(1).max())

klines = pf.klines(beginning_date, ending_date)
returns = pf.returns(beginning_date, ending_date)
returns.index = returns.index.tz_localize(None)

ticker = st.selectbox(
    label="Select ticker",
    options=["ALL", "RETURN"] + list(klines.index.get_level_values(0).unique()),
    key="ticker",
    index=1,
)

if st.session_state["ticker"] in list(klines.index.get_level_values(0).unique()):
    fig = px.line(klines.loc[st.session_state["ticker"]], y="value")
elif st.session_state["ticker"] == "RETURN":
    fig = px.line(returns)
elif st.session_state["ticker"] == "SUM":
    fig = px.line(
        klines.drop("SUM"),
        x=klines.drop("SUM").index.get_level_values(1),
        y="value",
        color=klines.drop("SUM").index.get_level_values(0),
    )
st.plotly_chart(fig)


st.write(returns.plot_monthly_heatmap(show=False))
st.pyplot(returns.plot_drawdown(show=False))
st.pyplot(returns.plot_snapshot(show=False))
