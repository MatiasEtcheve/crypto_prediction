import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import pytz
from binance import BinanceSocketManager
from binance.client import Client
from binance.enums import *


class Asset(object):
    def __init__(
        self,
        ticker: str,
        df: pd.DataFrame,
        precision_step: float,
        precision_tick: float,
        max_amount: float,
        interval: str = "1d",
        compute_metrics: Callable = lambda x: x,
        initial_amount: float = 0,
    ):
        self.ticker = ticker
        self.df = df
        self.precision_step = precision_step
        self.precision_tick = precision_tick
        self.max_amount = max_amount
        self.interval = interval
        self.compute_metrics = compute_metrics
        self.initial_amount = initial_amount

        self.trades = []
        self.orders = []
        self._trade_ids = set()
        self._order_ids = set()

    def isempty(self):
        return len(self.df) == 0

    def predict_from(self, rf):
        return rf.predict(self.df.to_numpy()[:, 7:-1])

    def predict_last_from(self, rf):
        return rf.predict(self.df.to_numpy()[-1, 7:-1].reshape(1, -1))[0]

    def predict_proba_last_from(self, rf):
        return rf.predict_proba(self.df.to_numpy()[-1, 7:-1].reshape(1, -1))[0][1]

    def append_kline(
        self, values: List[float], datetime: Union[datetime, List[datetime]]
    ):
        if not isinstance(datetime, list) or not isinstance(datetime, np.ndarray):
            datetime = [datetime]
        df = pd.DataFrame(
            values, index=datetime, columns=["Open", "High", "Low", "Close", "Volume"]
        )
        df.index = (
            pd.to_datetime(df.index, unit="ms").tz_localize(pytz.UTC).rename("Datetime")
        )
        if df.index[0] not in self.df.index:
            self.df = pd.concat([self.df, df])
            self.df = self.df.sort_index()
            self.df = self.compute_metrics(self.df)

    async def kline_listener(self, bm: BinanceSocketManager, interval: str = "1d"):
        async with bm.kline_socket(
            symbol=self.ticker + "BUSD", interval=interval
        ) as stream:
            while True:
                kline = (await stream.recv())["k"]
                if kline["x"]:
                    self.append_kline(
                        values=np.array(
                            [
                                float(kline["o"]),
                                float(kline["h"]),
                                float(kline["l"]),
                                float(kline["c"]),
                                float(kline["v"]),
                            ]
                        ).reshape(1, -1),
                        datetime=kline["t"],
                    )
                    return

    def cancel_orders(self, client: Client):
        open_orders = client.get_open_orders(symbol=self.ticker + "BUSD")
        for order in open_orders:
            client.cancel_order(
                symbol=self.ticker + "BUSD",
                orderId=order["orderId"],
                timestamp=True,
            )

    def update_trades(
        self,
        client: Client,
        starting_date: Optional[datetime] = None,
        limit: Optional[int] = 20,
    ):
        last_trades = client.get_my_trades(
            symbol=self.ticker + "BUSD",
            limit=limit,
            startTime=None
            if starting_date is None
            else int(starting_date.timestamp() * 1000),
        )
        for last_trade in last_trades:
            if last_trade["id"] not in self._trade_ids:
                self.trades.append(last_trade)
                self._trade_ids.add(last_trade["id"])

    def update_orders(
        self,
        client: Client,
        starting_date: Optional[datetime] = None,
        limit: Optional[int] = 20,
    ):
        last_orders = client.get_all_orders(
            symbol=self.ticker + "BUSD",
            limit=limit,
            startTime=None
            if starting_date is None
            else int(starting_date.timestamp() * 1000),
        )
        for last_order in last_orders:
            if last_order["orderId"] not in self._order_ids:
                self.orders.append(last_order)
                self._order_ids.add(last_order["orderId"])

    def save_orders_history(self, path: Optional[Path] = None):
        path = path / "orders.csv"
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
        orders_df = pd.DataFrame(self.orders)
        if path.is_file():
            previous_orders_df = pd.read_csv(path)
            all_orders_df = pd.concat([previous_orders_df, orders_df])
        else:
            all_orders_df = orders_df
        if not all_orders_df.empty:
            mask = all_orders_df["orderId"].duplicated()
            all_orders_df[~mask].to_csv(path, index=False)

    def save_trades_history(self, path: Optional[Path] = None):
        path = path / "trades.csv"
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
        trades_df = pd.DataFrame(self.trades)
        if path.is_file():
            previous_trades_df = pd.read_csv(path)
            all_trades_df = pd.concat([previous_trades_df, trades_df])
        else:
            all_trades_df = trades_df
        if not all_trades_df.empty:
            mask = all_trades_df["id"].duplicated()
            all_trades_df[~mask].to_csv(path, index=False)

    def save_history(self, path: Optional[Path] = None):
        self.save_orders_history(path)
        self.save_trades_history(path)
