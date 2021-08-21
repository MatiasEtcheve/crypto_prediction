import csv
from datetime import datetime
from os import listdir, getcwd, remove

import numpy as np
import pandas as pd
from binance.client import Client
from pprint import pprint
import config
import tools.dict_tools as dict_tools

client = Client(config.API_KEY, config.API_SECRET)

TICKERS = client.get_all_tickers()
SYMBOLS = [ticker["symbol"] for ticker in tickers]

if __name__ == "__main__":
    # klines = client.get_historical_klines("BNBBTC", Client.KLINE_INTERVAL_12HOUR, "17 June, 2020")
    client.get_avg_price(symbol='BNBBTC')
    print(len(SYMBOLS))


