import os
import pickle
import sys
from calendar import month
from datetime import date, datetime, timedelta
from importlib import reload
from pathlib import Path
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import quantstats as qs
import streamlit as st
from binance.client import Client
from datapoints import assets, portfolios
from datapoints.portfolios import PastPortfolio
from plotly.subplots import make_subplots
from yfinance import Tickers

qs.extend_pandas()
testnet = True
st.set_page_config(layout="wide")

reload_cache = st.button(label="Reload cache", on_click=st.experimental_memo.clear)

network = st.radio(
    label="Select network",
    options=["Binance", "Testnet"],
    index=0,
)

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
            value=date(2022, 4, 8),
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
def load_portfolio(network, interval):
    if network == "Testnet":
        api_key = os.environ.get("TESTNET_API")
        api_secret = os.environ.get("TESTNET_SECRET")
        tickers = ["BNB", "BTC", "ETH", "LTC", "TRX", "XRP"]
    elif network == "Binance":
        api_key = os.environ.get("BINANCE_WATCH_API")
        api_secret = os.environ.get("BINANCE_WATCH_SECRET")
        tickers = [
            "ETH",
            "BTC",
            "BNB",
            "LTC",
            "XRP",
            "TRX",
            "ALGO",
            "DOT",
            "TRX",
            "EOS",
            "BNB",
            "SNX",
            "OMG",
            "ETH",
            "LUNA",
            "ADA",
            "AAVE",
            "SOL",
            "DOGE",
            "AVAX",
        ]
    else:
        raise NotImplementedError("")
    client = Client(api_key, api_secret, testnet=network == "Testnet")
    config = {"interval": interval}
    return PastPortfolio.from_tickers(
        client,
        config,
        tickers,
        beginning_date,
        ending_date,
    )


pf = load_portfolio(network, st.session_state["interval"])
pf.trades["returns"] = (pf.trades["sold_price"] - pf.trades["price"]) / pf.trades[
    "price"
]

if not pf.trades.empty:
    st.write(
        "Last trades at:",
        pf.trades.loc[
            pd.DataFrame.from_records(pf.trades.index).groupby(0)[1].last().items()
        ]
        .loc[
            :,
            [
                "time",
                "orderId",
                "price",
                "qty",
                "commission",
                "isBuyer",
                "amountAdded",
            ],
        ]
        .sort_index(),
    )

# st.write(
#     "Best trade:",
#     pf.trades.loc[pf.trades["returns"].idxmax()].loc[
#         :, ["price", "time", "sold_price", "sold_time"]
#     ],
# )

if not pf.orders.empty:
    st.write(
        "Last orders at:",
        pf.orders.loc[
            pd.DataFrame.from_records(pf.orders.index).groupby(0)[1].last().items()
        ]
        .loc[
            :,
            ["time", "orderId", "price", "origQty", "executedQty", "status", "side"],
        ]
        .sort_index(),
    )

st.write(
    "Balances",
    pf.current_values,
)
st.write("Initial balance", pf.initial_amounts)
klines = pf.klines
returns = pf.returns
orders = pf.orders.loc[:, beginning_date:, :]
returns.index = returns.index.tz_localize(None)

ticker = st.selectbox(
    label="Select ticker",
    options=["ALL", "RETURN"] + list(klines.index.get_level_values(0).unique()),
    key="ticker",
    index=1,
)

if st.session_state["ticker"] == "SUM":
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(
            x=klines.loc[st.session_state["ticker"]].index,
            y=klines.loc[st.session_state["ticker"]]["value"],
            name="value",
        ),
    )

elif (
    st.session_state["ticker"] in list(klines.index.get_level_values(0).unique())
    and st.session_state["ticker"] != "SUM"
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if st.session_state["ticker"] in orders.index.get_level_values(0):
        sell_mask = orders.loc[st.session_state["ticker"]]["side"] == "SELL"
        sell_orders = orders.loc[st.session_state["ticker"]].loc[sell_mask]

        buy_mask = orders.loc[st.session_state["ticker"]]["side"] == "BUY"
        buy_orders = orders.loc[st.session_state["ticker"]].loc[buy_mask]

        size = 12
        fig.add_trace(
            go.Scatter(
                x=sell_orders["time"],
                y=sell_orders["price"],
                name="sell orders",
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=size,
                ),
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=buy_orders["time"],
                y=buy_orders["price"],
                name="buy orders",
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=size,
                ),
            ),
        )
    fig.add_trace(
        go.Scatter(
            x=klines.loc[st.session_state["ticker"]].index,
            y=klines.loc[st.session_state["ticker"]]["value"],
            name="value",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=pf.assets[st.session_state["ticker"]].df.index,
            y=pf.assets[st.session_state["ticker"]].df["Open"],
            name="Open price",
        ),
        secondary_y=False,
    )

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
st.pyplot(returns.plot_distribution(show=False))

st.pyplot(returns.plot_histogram(resample="D", show=False))


st.title("Log")

with open(
    Path(__file__).resolve().parents[1] / "live_trading" / "first_strategy" / "log.txt"
) as log_file:
    lines = log_file.readlines()
    st.write("\n".join(lines[-50:-1:]))
