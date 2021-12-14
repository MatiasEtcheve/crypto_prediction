import math
import backtrader as bt

# Don't Trade Moving Average Crossovers - Trade This Instead
# https://www.youtube.com/watch?v=T62NrcrgeWc

BEST_PARAMS = [
    {
        "period": 85,
        "consecutive_period": 63,
    },
    {
        "period": 115,
        "consecutive_period": 86,
    },
    {
        "period": 185,
        "consecutive_period": 5,
    },
    {
        "period": 195,
        "consecutive_period": 5,
    },
    {
        "period": 205,
        "consecutive_period": 5,
    },
    {
        "period": 215,
        "consecutive_period": 5,
    },
    {
        "period": 385,
        "consecutive_period": 100,
    },
]


class MACrossover(bt.Strategy):
    def __init__(self):
        sma_long = bt.indicators.MovingAverageSimple(self.data.close, period=200)
        sma_short = bt.indicators.MovingAverageSimple(self.data.close, period=50)
        self.cross = bt.indicators.CrossOver(sma_short, sma_long)
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if self.cross[0] >= 1 and not self.position:
            self.buy()
        if self.cross[0] <= -1 and self.position:
            self.sell()


class MAChannel(bt.Strategy):
    params = (("period", 95), ("consecutive_period", 95 // 4))

    def __init__(self):
        sma_high = bt.indicators.MovingAverageSimple(
            self.data.high, period=self.p.period
        )
        sma_low = bt.indicators.MovingAverageSimple(self.data.low, period=self.p.period)
        self.higher = bt.Or(self.data.open >= sma_high, self.data.close >= sma_high)
        self.lower = bt.Or(self.data.open <= sma_low, self.data.close <= sma_low)
        bt.LinePlotterIndicator(self.higher, name="higher")
        bt.LinePlotterIndicator(self.lower, name="lower")
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if (
            not self.position
            and math.fsum(self.higher.get(size=self.p.consecutive_period))
            == self.p.consecutive_period
        ):
            self.buy()
        if (
            self.position
            and math.fsum(self.lower.get(size=self.p.consecutive_period))
            == self.p.consecutive_period
        ):
            self.sell()
>>>>>>> cac7099 (backtrader)
