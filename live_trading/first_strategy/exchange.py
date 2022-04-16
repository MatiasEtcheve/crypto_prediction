import asyncio
import os
import pickle
from datetime import datetime, timedelta
from importlib import reload
from pathlib import Path
from pprint import pprint

import numpy as np
import wandb
from binance import AsyncClient, BinanceSocketManager
from binance.client import Client
from binance.enums import *
from tools import inspect_code, plotting, training, wandb_api

from live_trading import portfolio

saving_path = Path(__file__).resolve().parent
root_path = Path(__file__).resolve().parent / "tmp"
starting_date = datetime.now()

wandb_api.login()
api = wandb.Api()
run_name = "1fullt5y"
run = api.run(f"matiasetcheverry/crypto-prediction/{run_name}")
config = run.config
model = run.file("rf.pkl")
model = model.download(root=root_path / run.name, replace=True)
with open(model.name, "rb") as file:
    rf = pickle.load(file)


class ClassificationPortfolio(portfolio.Portfolio):
    def compute_amount_money(
        self, money: float, directions: np.ndarray, probabilities: np.ndarray
    ):
        # preallocated_money = [
        #     float(self.client.get_asset_balance(asset=asset.ticker)["free"])
        #     - asset.initial_amount
        #     for asset in self.assets.values()
        # ]
        allocated_money = np.zeros_like(probabilities)
        # mask = directions * (np.array(preallocated_money) == 0)
        mask = directions
        allocated_money[mask] = (
            probabilities[mask] * money / np.sum(probabilities[mask])
        )
        return allocated_money

    def compute_probabilities(self) -> np.ndarray:
        probabilities = []
        for asset in self.assets.values():
            try:
                probabilities.append(asset.predict_proba_last_from(rf))
            except ValueError as e:
                print(f"Problem predicting on {asset.ticker}")
                print(f"\t{e}")
                probabilities.append(0)
        return np.array(probabilities)

    def trade(self, money: int):
        os.system("/bin/bash -c 'ntpdate pool.ntp.org'")
        self.cancel_orders()
        probabilities = self.compute_probabilities()
        directions = probabilities > 0.5
        allocated_money = self.compute_amount_money(
            money,
            directions,
            probabilities,
        )
        order_ids = []
        for index, asset in enumerate(self.assets.values()):
            probability = probabilities[index]
            direction = directions[index]
            money = allocated_money[index]
            orders = self.swap_single_ticker(direction, money, asset)
            if orders is not None:
                for order in orders:
                    print("New order")
                    print(f"\tTICKER: {order['symbol']}")
                    print(f"\tSide: {order['side']}")
                    print(f"\tProba going up: {probability}")
                    print(f"\tAmount: {order['origQty']}")
                    print(f"\tPrice: {order['price']}")
                    print(
                        f"\tMoney allocated: {float(order['origQty'])*float(order['price'])}"
                    )
                    print(f"\tOrder id: {order['orderId']}")
                    print(f"\tTransaction time: {order['transactTime']}")
                    order_ids.append(
                        {"symbol": order["symbol"], "orderId": order["orderId"]}
                    )
        return order_ids


api_key = os.environ.get("TESTNET_API")
api_secret = os.environ.get("TESTNET_SECRET")
client = Client(api_key, api_secret, testnet=True)
pf = ClassificationPortfolio(
    client,
    config,
    saving_path / "live_trading",
    ["BNB", "BTC", "ETH", "LTC", "TRX", "XRP"],
)


async def kline_listener(aclient):
    bm = BinanceSocketManager(aclient)
    input_coroutines = [
        asset.kline_listener(bm, interval="5m") for asset in list(pf.assets.values())
    ]
    await asyncio.gather(*input_coroutines, return_exceptions=True)
    print("Finally gathered at:", datetime.now())
    print("Initially traded at:", datetime.now())
    pf.trade(float(pf.client.get_asset_balance(asset="BUSD")["free"]))
    print("Finally traded at:", datetime.now())

    for asset in pf.assets.values():
        asset.update_trades(client, starting_date=starting_date, limit=20)
        asset.update_orders(client, starting_date=starting_date, limit=20)
        asset.save_history(saving_path / "live_trading" / asset.ticker)


async def main():
    aclient = await AsyncClient.create()
    while True:
        await kline_listener(aclient)
        print("------------------------------------------")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
