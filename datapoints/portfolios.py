import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytz
from binance.client import Client
from random_forests import utils

from datapoints.assets import LiveAsset, PastAsset, PastQuote


def _cast_date(shitty_date):
    if isinstance(shitty_date, date):
        return datetime(shitty_date.year, shitty_date.month, shitty_date.day)
    return shitty_date.replace(tzinfo=pytz.UTC)


class LivePortfolio(object):
    def __init__(
        self,
        client: Client,
        config,
        assets: Optional[List[LiveAsset]] = None,
        path: Optional[Path] = Path().resolve().parent / "live_trading",
    ):
        self.client = client
        self.config = config
        self.path = path

        self.assets = {asset.ticker: asset for asset in assets}
        self.tickers = list(self.assets.keys())

    def cancel_orders(self):
        for asset in self.assets.values():
            asset.cancel_orders()

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

    def _compute_inital_balance(self):
        amount = []
        for asset in self.assets.values():
            amount.append({"ticker": asset.ticker, "amount": asset.initial_amount})
        return (
            pd.DataFrame(amount)
            .set_index(keys="ticker", drop=True)
            .sort_values("amount", axis=0, ascending=False)
        )

    def _compute_current_balance(self):
        return _compute_current_balance(self.client)

    @classmethod
    def from_tickers(cls, client, config, tickers, path):
        assets = []

        time_equivalence = {"d": "days", "h": "hours", "m": "minutes"}
        interval_value, interval_base = re.findall("\d+|\D+", config["interval"])
        interval_value = 400 * int(interval_value)
        interval_base = time_equivalence[interval_base]

        for ticker in tickers:
            print(f"Building {ticker} asset:")
            asset = utils.create_asset(
                ticker,
                config["interval"],
                beginning_date=datetime.now()
                - timedelta(**{interval_base: interval_value}),
                ending_date=datetime.now(),
                compute_metrics=utils._concatenate_indicators,
            )
            assets.append(
                LiveAsset(
                    asset.ticker,
                    asset.df.iloc[:-1],
                    asset.labels.iloc[:-1],
                    client,
                    config["interval"],
                    utils._concatenate_indicators,
                )
            )
        return cls(client, config, assets=assets, path=path)


class PastPortfolio(LivePortfolio):
    def __init__(
        self,
        client,
        config,
        assets,
    ):
        super().__init__(client, config, assets)

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

        earliest_date = min(
            [asset.klines.index.min() for asset in self.assets.values()]
        )
        latest_date = max([asset.klines.index.max() for asset in self.assets.values()])
        quote = utils.create_asset(
            "BUSD",
            self.config["interval"],
            earliest_date,
            latest_date,
            utils._concatenate_indicators,
        )
        self.quote = PastQuote(
            ticker=quote.ticker,
            df=quote.df,
            labels=quote.labels,
            client=self.client,
            trades=self.trades,
            interval=quote.interval,
            compute_metrics=quote.compute_metrics,
        )

        self._compute_klines()
        self.current_values = self._compute_current_balance()
        self.initial_amounts = self._compute_inital_balance()

    def _compute_klines(self):
        common_dates = set.intersection(
            *[set(asset.klines.index) for asset in self.assets.values()],
            set(self.quote.klines.index),
        )
        klines = pd.concat(
            [asset.klines.loc[common_dates] for asset in self.assets.values()]
            + [self.quote.klines.loc[common_dates]],
            keys=self.tickers + ["BUSD"],
            axis=0,
        )
        somme = klines.groupby(level=1).agg(
            amount=("amount", "sum"), value=("value", "sum")
        )
        self.klines = pd.concat(
            [
                pd.concat(
                    [somme],
                    keys=["SUM"],
                ),
                pd.concat(
                    [asset.klines for asset in self.assets.values()]
                    + [self.quote.klines],
                    keys=self.tickers + ["BUSD"],
                    axis=0,
                ),
            ]
        )
        print(self.klines.loc["SUM"]["value"])
        self.returns = self.klines.loc["SUM"]["value"].pct_change()

    # @classmethod
    # def from_tickers(cls, client, config, asset_tickers, beginning_date, ending_date):
    #     assets = []
    #     for ticker in asset_tickers:
    #         print(f"Building {ticker} asset:")
    #         asset = utils.create_asset(
    #             ticker,
    #             config["interval"],
    #             beginning_date=_cast_date(beginning_date),
    #             ending_date=_cast_date(ending_date),
    #             compute_metrics=utils._concatenate_indicators,
    #         )
    #         assets.append(
    #             PastAsset(
    #                 asset.ticker,
    #                 asset.df,
    #                 asset.labels,
    #                 client,
    #                 config["interval"],
    #                 utils._concatenate_indicators,
    #             )
    #         )
    #     return cls(client, config, assets=assets)


def _compute_current_balance(client):
    balances = []
    for balance in client.get_account()["balances"]:
        ticker = balance["asset"]
        if balance["asset"].startswith("LD"):
            ticker = balance["asset"][2:]

        current_amount = float(balance["free"]) + float(balance["locked"])
        if current_amount > 0:

            if ticker in ["BUSD", "USDT"]:
                avg_price = 1
            elif ticker == "BTC":
                avg_price = float(client.get_avg_price(symbol=ticker + "USDT")["price"])

            else:
                avg_price = float(
                    client.get_avg_price(symbol=ticker + "BTC")["price"]
                ) * float(client.get_avg_price(symbol="BTCUSDT")["price"])
            current_value = current_amount * avg_price
        else:
            current_value = 0
        balances.append(
            {"ticker": ticker, "amount": current_amount, "value": current_value}
        )
    return (
        pd.DataFrame(balances)
        .set_index(keys="ticker", drop=True)
        .sort_values("value", axis=0, ascending=False)
    )
