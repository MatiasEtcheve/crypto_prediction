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
from binance.exceptions import *
from datasets.filters import *
from tools import inspect_code, plotting, training, wandb_api

saving_path = Path().resolve().parent
root_path = Path().resolve().parent / "tmp"
starting_date = datetime.now()

config = {}


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


config["interval"] = "1d"


class VolatilityPortfolio(portfolios.LivePortfolio):
    def __init__(self, client, config, assets, path):
        super().__init__(client, config, assets, path)

        self.last_trade_prices = {
            asset.ticker: float(
                self.client.get_avg_price(symbol=asset.ticker + "BUSD")["price"]
            )
            for asset in self.assets.values()
        }
        self.floor_price = {
            ticker: 0.7 * price for ticker, price in self.last_trade_prices.items()
        }
        self.sell_orders = {}
        self.buy_orders = {}
        self.cancel_orders()
        for asset in self.assets.values():
            self.place_orders(asset)

    def place_sell_order(self, asset):
        try:
            amount = asset._get_current_amount()

            origPrice = self.last_trade_prices[asset.ticker] * (1 + 1.5 / 100)
            origAmount = 1 / 100 * amount
            order = asset.place_sell_order(
                origPrice=origPrice,
                origAmount=origAmount,
            )
            log_info = f"order: {order['symbol']}, side: {order['side']}, current_price: {self.last_trade_prices[asset.ticker]}, new_price: {origPrice}, quantity: {origAmount}, money allocated: {float(order['origQty'])*float(order['price'])}"
            asset.logger.info(log_info)
            print(log_info)
            return order
        except MinNotionalException as e:
            asset.logger.error(
                f"Can't sell {asset.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
            )
            print(
                f"Can't sell {asset.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
            )
        except Exception as e:
            asset.logger.error(f"{type(e)}: {str(e)}")
        return None

    def place_buy_order(self, asset):
        try:
            amount = asset._get_current_amount()

            available_quote_amount = float(
                self.client.get_asset_balance(asset="BUSD")["free"]
            )
            origPrice = self.last_trade_prices[asset.ticker] * (1 - 1.5 / 100)
            origAmount = min(
                1.0125 / 100 * amount,
                available_quote_amount / origPrice,
            )
            if origPrice > self.floor_price[asset.ticker]:
                order = asset.place_buy_order(
                    origPrice=origPrice,
                    origAmount=origAmount,
                )
                log_info = f"order: {order['symbol']}, side: {order['side']}, current_price: {self.last_trade_prices[asset.ticker]}, new_price: {origPrice}, quantity: {origAmount}, money allocated: {float(order['origQty'])*float(order['price'])}"
                asset.logger.info(log_info)
                print(log_info)
                return order
        except MinNotionalException as e:
            asset.logger.error(
                f"Can't buy {asset.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
            )
            print(
                f"Can't buy {asset.ticker} dust:\n\tAmount: {e.amount}\tPrice: {e.price}\n\tMin notional: {e.min_notional}\tNotional: {e.notional}"
            )
        except LowSizeException as e:
            asset.logger.error(f"Can't buy {asset.ticker} dust:\nLow size error")
        except Exception as e:
            asset.logger.error(f"{type(e)}: {str(e)}")
        return None

    def place_orders(self, asset):
        order = self.place_sell_order(asset)
        self.sell_orders[asset.ticker] = order
        order = self.place_buy_order(asset)
        self.buy_orders[asset.ticker] = order

    def trade(self):
        for asset in self.assets.values():
            buy_order = self.buy_orders[asset.ticker]
            if buy_order is not None and self.client.get_order(
                symbol=buy_order["symbol"],
                orderId=buy_order["orderId"],
            )["status"] in [
                ORDER_STATUS_PARTIALLY_FILLED,
                ORDER_STATUS_FILLED,
            ]:
                self.last_trade_prices[asset.ticker] = buy_order["price"]
                asset.cancel_orders()
                self.place_orders(asset)

            sell_order = self.client.get_order(
                symbol=self.sell_orders[asset.ticker]["symbol"],
                orderId=self.sell_orders[asset.ticker]["orderId"],
            )
            if sell_order["status"] in [
                ORDER_STATUS_PARTIALLY_FILLED,
                ORDER_STATUS_FILLED,
            ]:
                self.last_trade_prices[asset.ticker] = sell_order["price"]
                asset.cancel_orders()
                self.place_orders(asset)


api_key = os.environ.get("TESTNET_API")
api_secret = os.environ.get("TESTNET_SECRET")
client = Client(api_key, api_secret, testnet=True)
pf = VolatilityPortfolio.from_tickers(
    client,
    config,
    ["BNB", "BTC", "ETH", "LTC", "TRX", "XRP"],
    saving_path / "live_trading",
)
while True:
    pf.trade()
