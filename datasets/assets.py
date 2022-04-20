import logging
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional, Union

import get_data
import numpy as np
import pandas as pd
import pytz
from binance import BinanceSocketManager
from binance.client import Client
from binance.enums import *

from datasets.filters import *


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = 0
        console = logging.FileHandler("log.txt")
        logger.addHandler(console)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        console.setFormatter(formatter)
    return logger


logger = get_logger(__name__)


class TrainAsset(object):
    def __init__(
        self,
        ticker: str,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        interval: str = "1d",
        compute_metrics: Callable = lambda x: x,
    ):
        self.ticker = ticker
        self.df = df
        self.labels = labels
        self.interval = interval
        self.compute_metrics = compute_metrics

        assert len(self.labels) == len(self.df)

    @property
    def isempty(self):
        return len(self.df) == 0

    @property
    def features(self):
        return self.df.to_numpy()[:, 6:]

    def predict_from(self, rf):
        return rf.predict(self.features)

    def predict_proba_from(self, rf):
        return rf.predict_proba(self.features)[:, 1]

    def predict_last_from(self, rf):
        return rf.predict(self.features[-1].reshape(1, -1))[0]

    def predict_proba_last_from(self, rf):
        return rf.predict_proba(self.features[-1].reshape(1, -1))[0][1]

    def append_kline(
        self,
        values: List[float],
        datetime: Union[datetime, List[datetime]],
        overwrite: bool = False,
    ):
        if not isinstance(datetime, list) or not isinstance(datetime, np.ndarray):
            datetime = [datetime]
        df = pd.DataFrame(
            values, index=datetime, columns=["Open", "High", "Low", "Close", "Volume"]
        )
        df.index = (
            pd.to_datetime(df.index, unit="ms").tz_localize(pytz.UTC).rename("Datetime")
        )
        if overwrite or df.index[0] not in self.df.index:
            self.df = pd.concat([self.df, df])
            self.df = self.df[~self.df.index.duplicated(keep="last")]
            self.df = self.df.sort_index()
            self.df = self.compute_metrics(self.df)

            self.labels = self.df.pop("Direction")


class LiveAsset(TrainAsset):
    def __init__(
        self,
        ticker: str,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        client: Client,
        interval: str = "1d",
        compute_metrics: Callable = lambda x: x,
    ):
        super().__init__(ticker, df, labels, interval, compute_metrics)

        self.client = client
        (
            self.max_amount,
            self.precision_step,
            self.precision_tick,
        ) = self._retrieve_precisions()

        self.trades = []
        self.orders = []
        self._trade_ids = set()
        self._order_ids = set()

    def _get_current_amount_value(self):
        balance = self.client.get_asset_balance(asset=self.ticker)
        return float(balance["free"]) + float(balance["locked"])

    def _retrieve_precisions(self):
        info = self.client.get_symbol_info(self.ticker + "BUSD")
        for filter in info["filters"]:
            if filter["filterType"] == "LOT_SIZE":
                step_size = float(filter["stepSize"])
                max_amount = float(filter["maxQty"])
            if filter["filterType"] == "PRICE_FILTER":
                tick_size = float(filter["tickSize"])
        return (
            max_amount,
            int(round(-np.log10(step_size), 0)),
            int(round(-np.log10(tick_size), 0)),
        )

    def validate_order(self, amount: float, price: float):
        ticker = self.ticker + "BUSD"
        filters = self.client.get_symbol_info(ticker)["filters"]
        for filter in filters:
            if filter["filterType"] == "PRICE_FILTER":
                if float(filter["minPrice"]) > price:
                    raise LowPriceFilterException
                if float(filter["maxPrice"]) < price:
                    raise HighPriceFilterException
            if filter["filterType"] == "PERCENT_PRICE":
                avg_price = float(self.client.get_avg_price(symbol=ticker)["price"])
                if price > float(filter["multiplierUp"]) * avg_price:
                    raise HighPercentPriceException
                if price < float(filter["multiplierDown"]) * avg_price:
                    raise LowPercentPriceException
            if filter["filterType"] == "LOT_SIZE":
                if amount > float(filter["maxQty"]):
                    raise HighSizeException(max_amount=float(filter["maxQty"]))
                if amount < float(filter["minQty"]):
                    raise LowSizeException
            if filter["filterType"] == "MIN_NOTIONAL":
                if filter["applyToMarket"] and price * amount < float(
                    filter["minNotional"]
                ):
                    raise MinNotionalException(
                        price=price,
                        amount=amount,
                        min_notional=filter["minNotional"],
                        notional=price * amount,
                    )

    def place_buy_order(self, origAmount: float, origPrice: float):
        amount = np.true_divide(
            np.floor(origAmount * 10 ** self.precision_step),
            10 ** self.precision_step,
        )
        price = np.true_divide(
            np.floor(origPrice * 10 ** self.precision_tick),
            10 ** self.precision_tick,
        )
        self.validate_order(amount, price)
        amount = "{val:.{precision}f}".format(val=amount, precision=self.precision_step)
        price = "{val:.{precision}f}".format(val=price, precision=self.precision_tick)
        return self.client.order_limit_buy(
            symbol=self.ticker + "BUSD", quantity=amount, price=price
        )

    def place_sell_order(self, origAmount: float, origPrice: float):
        amount = np.true_divide(
            np.floor(origAmount * 10 ** self.precision_step),
            10 ** self.precision_step,
        )
        price = np.true_divide(
            np.ceil(origPrice * 10 ** self.precision_tick),
            10 ** self.precision_tick,
        )
        self.validate_order(amount, price)
        amount = "{val:.{precision}f}".format(val=amount, precision=self.precision_step)
        price = "{val:.{precision}f}".format(val=price, precision=self.precision_tick)
        return self.client.order_limit_sell(
            symbol=self.ticker + "BUSD", quantity=amount, price=price
        )

    def place_order(
        self,
        origPrice: float,
        allocated_money: float = None,
        origAmount: float = None,
        side: str = "buy",
    ):
        if side == "buy":
            assert (
                allocated_money is not None and origPrice is not None
            ), "You must specify the allocated money and the price when you buy."
            assert origAmount is None, "You can't specify the amount when you buy."
        else:
            assert (
                origAmount is not None and origPrice is not None
            ), "You must specify the amount and the price when you sell."
            assert (
                allocated_money is None
            ), "You can't specify the allocated money when you sell."

        if side == "buy":
            return self.place_buy_order(
                min(allocated_money / origPrice, self.max_amount),
                origPrice,
            )

        if side == "sell":
            return self.place_sell_order(
                origAmount,
                origPrice,
            )

    def sell_single_asset(self):
        balance = float(self.client.get_asset_balance(asset=self.ticker)["free"])
        if balance > 0:
            try:
                nb_complete_orders = int(balance // float(self.max_amount))
                amounts = [float(self.max_amount) for i in range(nb_complete_orders)]
                amounts += [balance - sum(amounts)]
                orders = []
                for amount in amounts:
                    orders.append(
                        self.place_order(
                            origPrice=self.df["Open"].iloc[-1],
                            origAmount=amount,
                            side="sell",
                        )
                    )
            except MinNotionalException as e:
                logger.error(
                    f"Can't sell {self.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
                )
            else:
                return orders
        return None

    def buy_single_asset(self, money: float):
        try:
            orders = [
                self.place_order(
                    origPrice=self.df["Open"].iloc[-1],
                    allocated_money=money,
                    side="buy",
                )
            ]
        except MinNotionalException as e:
            logger.error(
                f"Can't buy {self.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
            )
        except LowSizeException as e:
            logger.error(f"Can't buy {self.ticker} dust:\nLow size error")
        else:
            return orders
        return None

    def swap_single_ticker(self, direction: np.ndarray, money: float):
        if direction and money > 0:
            return self.buy_single_asset(money)
        if not direction:
            return self.sell_single_asset()
        return None

    def cancel_orders(self):
        open_orders = self.client.get_open_orders(symbol=self.ticker + "BUSD")
        for order in open_orders:
            self.client.cancel_order(
                symbol=self.ticker + "BUSD",
                orderId=order["orderId"],
                timestamp=True,
            )

    def update_trades(
        self,
        starting_date: Optional[datetime] = None,
        limit: Optional[int] = 20,
    ):
        last_trades = self.client.get_my_trades(
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
        starting_date: Optional[datetime] = None,
        limit: Optional[int] = 20,
    ):
        last_orders = self.client.get_all_orders(
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
                        overwrite=True,
                    )
                    self.append_kline(
                        values=np.array(
                            [
                                float(kline["c"]),
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
                        ).reshape(1, -1),
                        datetime=kline["T"] + 1,
                        overwrite=False,
                    )
                    return


class PastAsset(LiveAsset):
    def __init__(
        self,
        ticker: str,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        client: Client,
        interval: str = "1d",
        compute_metrics: Callable = lambda x: x,
    ):
        super(LiveAsset, self).__init__(ticker, df, labels, interval, compute_metrics)
        self.client = client

        self.current_amount = self._get_current_amount_value()
        self.trades = self.get_all_trades()
        self.orders = self.get_all_orders()
        self.compute_klines()

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

        self.trades = self.trades.set_index(keys="time", drop=False)
        self.trades["amountAdded"] = self.trades["qty"] * (
            2 * self.trades["isBuyer"] - 1
        )
        trades = self.trades.groupby(by=self.trades.index).agg(
            price=("price", "mean"), amountAdded=("amountAdded", "sum")
        )
        trades["trades"] = 1
        self.df["price"] = self.df["Open"]
        self.df["amountAdded"] = 0

        self.klines = pd.concat((self.df, trades))
        self.klines = self.klines.sort_index()

        self.initial_amount = (
            self.current_amount - self.klines["amountAdded"].cumsum()[-1]
        )
        self.amount_added = self.klines["amountAdded"].cumsum()[-1]
        self.klines["amount"] = (
            self.initial_amount + self.klines["amountAdded"].cumsum()
        )
        self.klines["value"] = self.klines["amount"] * self.klines["price"]

        self.klines = self.klines[np.isnan(self.klines["trades"])].drop(
            labels=["trades", "price"], axis=1
        )
        self.df = self.df.drop(labels=["price", "amountAdded"], axis=1)


class PastQuote(PastAsset):
    def __init__(
        self,
        ticker: str,
        df: pd.DataFrame,
        labels: pd.DataFrame,
        client: Client,
        trades,
        interval: str = "1d",
        compute_metrics: Callable = lambda x: x,
    ):
        super(LiveAsset, self).__init__(ticker, df, labels, interval, compute_metrics)
        self.client = client

        self.current_amount = self._get_current_amount_value()
        self.trades = trades
        self.compute_klines()

    def get_last_trades(self):
        raise NotImplementedError

    def get_last_orders(self):
        raise NotImplementedError

    def compute_klines(self):
        if len(self.trades) == 0:
            return []

        self.trades["amountAdded"] = -self.trades["quoteQty"] * (
            2 * self.trades["isBuyer"] - 1
        )

        trades = self.trades.groupby(level=1).agg(amountAdded=("amountAdded", "sum"))
        trades["trades"] = 1
        self.df["price"] = self.df["Open"]
        self.df["amountAdded"] = 0

        self.klines = pd.concat((self.df, trades))
        self.klines = self.klines.sort_index()

        self.initial_amount = (
            self.current_amount - self.klines["amountAdded"].cumsum()[-1]
        )
        self.amount_added = self.klines["amountAdded"].cumsum()[-1]
        self.klines["amount"] = (
            self.initial_amount + self.klines["amountAdded"].cumsum()
        )
        self.klines["value"] = self.klines["amount"] * self.klines["price"]

        self.klines = self.klines[np.isnan(self.klines["trades"])].drop(
            labels=["trades", "price"], axis=1
        )
        self.df = self.df.drop(labels=["price", "amountAdded"], axis=1)
