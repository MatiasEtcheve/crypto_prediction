import json
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz
import requests


def fetch_blockchain(
    symbol: Optional[str],
    interval: Optional[str],
    beginning_date: datetime,
    ending_date: Optional[datetime] = None,
):

    # assert symbol == "BTC" and interval == "1d"
    charts = [
        "avg-block-size",
        "hash-rate",
        "n-transactions",
        "trade-volume",  # The total USD value of trading volume on major bitcoin exchanges
        "market-cap",  # The total USD value of bitcoin in circulation.
        "median-confirmation-time",  # The median time for a transaction with miner fees to be included in a mined block and added to the public ledger.
        "difficulty",  # A relative measure of how difficult it is to mine a new block for the blockchain.
        "cost-per-transaction",  # A chart showing miners revenue divided by the number of transactions.
        "n-unique-addresses",  # The total number of unique addresses used on the blockchain.
        "miners-revenue",  # Total value in USD of coinbase block rewards and transaction fees paid to miners.
        # "transaction-fees-usd",  # Average transaction fees in USD per transaction. NOT WORKING
        "transaction-fees-usd",  # The total USD value of all transaction fees paid to miners. This does not include coinbase block rewards.
    ]
    list_df = []
    for chart in charts:
        r = requests.get(
            f"https://api.blockchain.info/charts/{chart}?start={beginning_date.strftime('%Y-%m-%d')}&timespan=30years&format=json&sampled=false"
        )
        df = (
            pd.DataFrame(json.loads(r._content.decode("utf-8"))["values"])
            .rename(columns={"x": "Datetime", "y": chart})
            .set_index(
                "Datetime",
                drop=True,
            )
        )
        df.index = pd.to_datetime(df.index, unit="s").tz_localize(pytz.UTC)
        df = df.groupby(df.index.date)[chart].mean()
        df.index = pd.to_datetime(df.index).tz_localize(pytz.UTC)
        list_df.append(df)
    blockchain = pd.concat(list_df, axis=1).astype("float32")
    blockchain["transaction-fees-usd"] = (
        blockchain["transaction-fees-usd"] / blockchain["n-transactions"]
    )
    return blockchain
