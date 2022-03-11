import os
from pprint import pprint

import ccxt

exchange = ccxt.binance(
    {
        "options": {
            "adjustForTimeDifference": True,
        },
        "enableRateLimit": True,
        "apiKey": os.getenv("BINANCE_API_KEY", None),
        "secret": os.getenv("BINANCE_API_SECRET", None),
    }
)
if exchange.has["fetchMyTrades"]:
    a = exchange.fetch_my_trades(symbol="ETHUSDT", since=None, limit=None, params={})[0]
    pprint(a)
    print(a["amount"] * a["price"], a["cost"], a["cost"] * 0.001 / a["price"])
    print(a["fee"])
