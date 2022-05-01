import asyncio
import logging
import os
import pickle
import time
from calendar import c
from datetime import datetime, timedelta
from distutils.log import ERROR
from importlib import reload
from pathlib import Path
from pprint import pprint
from tkinter.messagebox import INFO

import datasets.assets as assets
import datasets.portfolios as portfolios
import get_data
import numpy as np
import wandb
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.enums import *
from tools import inspect_code, plotting, training, wandb_api

saving_path = Path(__file__).resolve().parent
root_path = Path(__file__).resolve().parent / "tmp"
starting_date = datetime.now()

wandb_api.login()
api = wandb.Api()
run_name = "1gqaeid3"
run = api.run(f"matiasetcheverry/crypto-prediction/{run_name}")
config = run.config
model = run.file("rf.pkl")
model = model.download(root=root_path / run.name, replace=True)
with open(model.name, "rb") as file:
    rf = pickle.load(file)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 0
        console = logging.FileHandler("log.txt")
        logger.addHandler(console)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        console.setFormatter(formatter)
    return logger


logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
# ERROR
# WARNING
# INFO
# DEBUG


class ClassificationPortfolio(portfolios.LivePortfolio):
    def _compute_nb_asset_with_allocated_money_below_threshold(
        self, allocated_money, threshold
    ):
        return np.sum([1 for m in allocated_money.values() if m < threshold])

    def _get_top_n_assets(self, probabilities, n):
        if n >= len(list(probabilities.keys())):
            n = len(list(probabilities.keys()))
        sorted_probabilities = {
            ticker: p
            for ticker, p in sorted(
                probabilities.items(), key=lambda item: item[1], reverse=True
            )
        }
        return {
            ticker: sorted_probabilities[ticker]
            for ticker in list(sorted_probabilities.keys())[:n]
        }

    def _allocate_money(self, probabilities, money):
        sum_probabilities = np.sum(list(probabilities.values()))
        allocated_money = {
            ticker: probabilities[ticker] * money / sum_probabilities
            for ticker in probabilities.keys()
        }
        return allocated_money

    def compute_amount_money(self, money: float, probabilities: np.ndarray):
        allocated_money = {ticker: 0 for ticker in probabilities.keys()}
        positive_probabilities = {k: p for k, p in probabilities.items() if p > 0.5}
        n = len(list(probabilities.keys()))

        current_probabilities = self._get_top_n_assets(positive_probabilities, n)
        current_allocated_money = self._allocate_money(current_probabilities, money)
        while (
            self._compute_nb_asset_with_allocated_money_below_threshold(
                current_allocated_money, 10
            )
            > 0
            and n > 1
        ):
            n -= 1
            current_probabilities = self._get_top_n_assets(positive_probabilities, n)
            current_allocated_money = self._allocate_money(current_probabilities, money)

        allocated_money.update(current_allocated_money)
        return allocated_money

    def compute_probabilities(self) -> np.ndarray:
        probabilities = {}
        for asset in self.assets.values():
            try:
                p = asset.predict_proba_last_from(rf)
            except ValueError as e:
                logger.error(f"Problem predicting on {asset.ticker}: {e}")
                p = 0
            probabilities[asset.ticker] = p
        return probabilities

    def trade(self, money: int):
        # os.system("/bin/bash -c 'ntpdate pool.ntp.org'")
        self.cancel_orders()
        probabilities = self.compute_probabilities()
        allocated_money = self.compute_amount_money(
            money,
            probabilities,
        )
        order_ids = []
        for asset in self.assets.values():
            probability = probabilities[asset.ticker]
            direction = probability > 0.5
            money = allocated_money[asset.ticker]
            logger.info(
                f"prediction: ticker: {asset.ticker}, probability: {probability}, side: {direction}, money: {money}"
            )
            print(
                f"prediction: ticker: {asset.ticker}, probability: {probability}, side: {direction}, money: {money}"
            )

        for asset in self.assets.values():
            probability = probabilities[asset.ticker]
            direction = probability > 0.5
            money = allocated_money[asset.ticker]
            orders = asset.swap_single_ticker(direction, money)
            if orders is not None:
                for order in orders:
                    logger.info(
                        f"order: {order['symbol']}, side: {order['side']}, proba: {probability}, money allocated: {float(order['origQty'])*float(order['price'])}"
                    )
                    print(
                        f"order: {order['symbol']}, side: {order['side']}, proba: {probability}, money allocated: {float(order['origQty'])*float(order['price'])}"
                    )
                    order_ids.append(
                        {"symbol": order["symbol"], "orderId": order["orderId"]}
                    )
        return order_ids


api_key = os.environ.get("BINANCE_API")
api_secret = os.environ.get("BINANCE_SECRET")
client = Client(api_key, api_secret, testnet=False)
pf = ClassificationPortfolio.from_tickers(
    client,
    config,
    [
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
    ],
    saving_path / "live_trading",
)


async def kline_listener(aclient):
    bm = BinanceSocketManager(aclient)
    input_coroutines = [
        asset.kline_listener(bm, interval="1d") for asset in list(pf.assets.values())
    ]
    await asyncio.gather(*input_coroutines, return_exceptions=True)
    logger.info("klines gathered")
    pf.trade(float(pf.client.get_asset_balance(asset="BUSD")["free"]))
    logger.info("trades ordered")

    for asset in pf.assets.values():
        asset.update_trades(starting_date=starting_date, limit=20)
        asset.update_orders(starting_date=starting_date, limit=20)
        asset.save_history(saving_path / "live_trading" / asset.ticker)


async def main():
    aclient = await AsyncClient.create()
    while True:
        await kline_listener(aclient)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
