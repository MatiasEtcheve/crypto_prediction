import get_data
import backtrader as bt
import json
from io import StringIO
import strategy.test
from pprint import pprint
from strategy import moving_average
from datetime import timedelta, datetime
from time import time
from broker import Cerebro
from time import time

class PandasData(bt.feed.DataBase):
    params = (
        ("nocase", True),
        ("datetime", None),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", 7),
    )


from_date = datetime.now() - timedelta(days=365)
end_date = datetime.now() - timedelta(days=10)
interval = "12h"
symbol = "ETHUSDT"
# cerebro = Cerebro(symbol, interval, from_date, end_date, strategy.test.MACrossover, plot_rules="bars")


cerebro = Cerebro(
    symbol,
    interval,
    from_date,
    end_date,
    moving_average.MAChannel,
    plot_rules="equity bars indicators",
)
results = cerebro.run()
cerebro.save()
cerebro.plot(
    volume=False,
    style="candle",
    barup="green",
)
# print(cerebro)
