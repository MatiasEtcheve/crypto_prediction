from datetime import datetime, timedelta
from pathlib import Path
from pprint import pprint
from typing import List, Optional

import get_data
import numpy as np
import pandas as pd
import pytz
from binance.client import Client
from binance.enums import *

from .asset import Asset as Asset
from .filters import (
    HighPercentPriceException,
    HighPriceFilterException,
    HighSizeException,
    LowPercentPriceException,
    LowPriceFilterException,
    LowSizeException,
    MinNotionalException,
)


class Portfolio(object):
    def __init__(
        self,
        client: Client,
        config,
        path: Optional[Path] = Path().resolve().parent / "live_trading",
        tickers: Optional[List[str]] = None,
        assets: Optional[List[Asset]] = None,
    ):
        self.client = client
        self.config = config
        self.path = path

        if tickers is not None:
            self.init_from_tickers(tickers)

        if assets is not None:
            self.init_from_assets(assets)

        self.save_balances()

    def _concatenate_indicators(self, data: pd.DataFrame):
        m = list(range(1, 21)) + list(range(40, 241, 20))
        for mi in [0] + m:
            data[f"ir_{mi}"] = data["Close"].shift(mi) / data["Open"].shift(mi) - 1
        for mi in m:
            data[f"cr_{mi}"] = data["Close"].shift(1) / data["Close"].shift(mi + 1) - 1
        for mi in m:
            data[f"or_{mi}"] = data["Open"] / data["Close"].shift(mi) - 1
        data["Direction"] = data["Close"] > data["Open"]
        return data

    def _get_data(
        self,
        ticker: str,
        beginning_date: datetime,
        ending_date: datetime,
        interval: str,
    ):
        try:
            data = get_data.download_klines(
                ticker,
                interval,
                beginning_date=beginning_date,
                ending_date=ending_date,
                compute_metrics=self._concatenate_indicators,
            )
            return data
        except Exception as e:
            print(e)
            print(f"Ticker {ticker} could not be downloaded")
            return []

    def _retrieve_precisions(self, ticker):
        info = self.client.get_symbol_info(ticker + "BUSD")
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

    def init_from_tickers(self, tickers: List[str]):
        self.assets = {}
        for ticker in tickers:
            df = self._get_data(
                ticker=ticker,
                beginning_date=datetime.now().replace(tzinfo=pytz.utc)
                - timedelta(days=400),
                ending_date=datetime.now().replace(tzinfo=pytz.utc),
                interval=self.config["interval"],
            )
            if len(df) > 0:
                # df = df.set_index("Datetime")
                df = df.replace(
                    to_replace=[np.inf, -np.inf, np.float64("inf"), -np.float64("inf")],
                    value=0,
                )

                max_amount, precision_step, precision_tick = self._retrieve_precisions(
                    ticker
                )

                asset = Asset(
                    ticker=ticker,
                    interval=self.config["interval"],
                    df=df,
                    max_amount=max_amount,
                    precision_step=precision_step,
                    precision_tick=precision_tick,
                    compute_metrics=self._concatenate_indicators,
                    initial_amount=0,
                )
                if not asset.isempty():
                    self.assets[ticker] = asset
        self.tickers = list(self.assets.keys())

    def init_from_assets(self, list_assets: List[Asset]):
        self.assets = {asset.ticker: asset for asset in list_assets}
        self.tickers = list(self.assets.keys())

    def validate_order(self, asset: Asset, amount: float, price: float):
        ticker = asset.ticker + "BUSD"
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

    def place_buy_order(self, asset: Asset, origAmount: float, origPrice: float):
        amount = np.true_divide(
            np.floor(origAmount * 10 ** asset.precision_step),
            10 ** asset.precision_step,
        )
        price = np.true_divide(
            np.floor(origPrice * 10 ** asset.precision_tick),
            10 ** asset.precision_tick,
        )
        self.validate_order(asset, amount, price)
        amount = "{val:.{precision}f}".format(
            val=amount, precision=asset.precision_step
        )
        price = "{val:.{precision}f}".format(val=price, precision=asset.precision_tick)
        return self.client.order_limit_buy(
            symbol=asset.ticker + "BUSD", quantity=amount, price=price
        )

    def place_sell_order(self, asset: Asset, origAmount: float, origPrice: float):
        amount = np.true_divide(
            np.floor(origAmount * 10 ** asset.precision_step),
            10 ** asset.precision_step,
        )
        price = np.true_divide(
            np.ceil(origPrice * 10 ** asset.precision_tick),
            10 ** asset.precision_tick,
        )
        self.validate_order(asset, amount, price)
        amount = "{val:.{precision}f}".format(
            val=amount, precision=asset.precision_step
        )
        price = "{val:.{precision}f}".format(val=price, precision=asset.precision_tick)
        return self.client.order_limit_sell(
            symbol=asset.ticker + "BUSD", quantity=amount, price=price
        )

    def cancel_orders(self):
        for asset in self.assets.values():
            asset.cancel_orders(self.client)

    def place_order(
        self,
        asset: Asset,
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
                asset,
                min(allocated_money / origPrice, asset.max_amount),
                origPrice,
            )

        if side == "sell":
            return self.place_sell_order(
                asset,
                origAmount,
                origPrice,
            )

    def sell_single_asset(self, asset: Asset):
        balance = float(self.client.get_asset_balance(asset=asset.ticker)["free"])
        if balance - asset.initial_amount > 0:
            try:
                nb_complete_orders = int(
                    (balance - asset.initial_amount) // float(asset.max_amount)
                )
                amounts = [float(asset.max_amount) for i in range(nb_complete_orders)]
                amounts += [balance - asset.initial_amount - sum(amounts)]
                orders = []
                for amount in amounts:
                    orders.append(
                        self.place_order(
                            asset=asset,
                            origPrice=asset.df["Close"].iloc[-1],
                            origAmount=amount,
                            side="sell",
                        )
                    )
            except MinNotionalException as e:
                print(f"Can't sell {asset.ticker} dust:")
                print(f"\tAmount: {e.amount}\tPrice: {e.price}")
                print(f"\tMin notional: {e.min_notional}\tNotional: {e.notional}")
            else:
                return orders
        return None

    def buy_single_asset(self, asset: Asset, money: float):
        try:
            orders = [
                self.place_order(
                    asset=asset,
                    origPrice=asset.df["Close"].iloc[-1],
                    allocated_money=money,
                    side="buy",
                )
            ]
        except MinNotionalException as e:
            print(f"Can't buy {asset.ticker} dust:")
        except LowSizeException as e:
            print(f"Can't buy {asset.ticker} dust:")
        else:
            return orders
        return None

    def swap_single_ticker(self, direction: np.ndarray, money: float, asset: Asset):
        if direction and money > 0:
            return self.buy_single_asset(asset, money)
        if not direction:
            return self.sell_single_asset(asset)
        return None

    def save_history(self, path: Optional[Path] = None):
        if path is None:
            path = self.path
        for asset in self.assets.values():
            asset.save_history(path / asset.ticker)

    def save_balances(self, path: Optional[Path] = None):
        if path is None:
            path = self.path / "balances.csv"
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
        account = self.client.get_account()
        balances = pd.DataFrame.from_dict(
            [
                balance
                for balance in account["balances"]
                if balance["asset"] in self.tickers + ["BUSD", "USDT"]
            ]
        )
        balances["time"] = datetime.now().timestamp() * 1000
        balances["amount"] = balances["free"].astype(float) + balances["locked"].astype(
            float
        )
        balances["amount"] = balances["free"].astype(float) + balances["locked"].astype(
            float
        )

        def get_price(row):
            if row["asset"] in ["BUSD", "USDT"]:
                return 1
            else:
                avg_price = float(
                    self.client.get_avg_price(symbol=row["asset"] + "USDT")["price"]
                )
                return avg_price

        balances["price"] = balances.apply(func=get_price, axis=1)
        balances["values"] = balances["price"] * balances["amount"]

        if path.is_file():
            previous_balances = pd.read_csv(path)
            all_balances_df = pd.concat([previous_balances, balances])
        else:
            all_balances_df = balances
        all_balances_df.to_csv(path, index=False)
