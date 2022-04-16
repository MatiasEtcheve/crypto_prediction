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
from past_portfolio import PastPortfolio

qs.extend_pandas()
testnet = True


network = st.radio(
    label="Select network",
    options=["Binance", "Testnet"],
    index=1,
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
def load_portfolio(network, interval):
    if network == "Testnet":
        api_key = os.environ.get("TESTNET_API")
        api_secret = os.environ.get("TESTNET_SECRET")
    elif network == "Binance":
        api_key = os.environ.get("BINANCE_WATCH_API")
        api_secret = os.environ.get("BINANCE_WATCH_SECRET")
    client = Client(api_key, api_secret, testnet=network == "Testnet")
    return PastPortfolio(client, interval)


pf = load_portfolio(network, st.session_state["interval"])
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