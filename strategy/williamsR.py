from .generic import GenericStrategy
import backtrader as bt
import indicators

# Williams R indicator & MACD - Best Indicators for 2021?
# https://www.youtube.com/watch?v=OtTQWlGFMhU&t=207s

# BEST_PARAMS = [
#     {
#         "period": 85,
#         "consecutive_period": 63,
#     },
#     {
#         "period": 115,
#         "consecutive_period": 86,
#     },
#     {
#         "period": 185,
#         "consecutive_period": 5,
#     },
#     {
#         "period": 195,
#         "consecutive_period": 5,
#     },
#     {
#         "period": 205,
#         "consecutive_period": 5,
#     },
#     {
#         "period": 215,
#         "consecutive_period": 5,
#     },
#     {
#         "period": 385,
#         "consecutive_period": 100,
#     },
# ]


class WilliamsR(bt.Strategy):
    params = (("period", 5),)

    def __init__(self):
        self.williamsR = bt.indicators.WilliamsR(self.data, period=self.p.period)
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if not self.position and self.williamsR[0] <= -90:
            self.buy()
        if self.position and self.williamsR[0] >= -20:
            self.sell()


class WilliamsRMACDHisto(GenericStrategy):
    params = dict(
        williamsR_period=5,
        smoothing_period=5,
        consecutive_period=5,
        fast_period=12,
        slow_period=26,
        trail=False,
        verbose=0,
    )

    def __init__(self):
        self.williamsR = bt.indicators.WilliamsR(
            self.data, period=self.p.williamsR_period
        )
        self.histo = indicators.MACD(
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            smoothing_period=self.p.smoothing_period,
        ).lines.smooth_histo
        self.high = bt.And(
            *[
                self.histo(-i) < self.histo(-i - 1)
                for i in range(self.p.consecutive_period)
            ]
        )
        bt.LinePlotterIndicator(self.high, name="high")
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if not self.position and self.williamsR[0] <= -90 and self.high:
            self.order_buy = self.buy()
        if self.position and not self.high:
            self.close()


class WilliamsRMACD(GenericStrategy):
    params = dict(
        williamsR_period=5,
        smoothing_period=5,
        consecutive_period=5,
        fast_period=12,
        slow_period=26,
        stop_loss=0.05,
        trail=False,
        verbose=0,
    )

    def __init__(self):
        self.williamsR = bt.indicators.WilliamsR(
            self.data, period=self.p.williamsR_period
        )
        self.macd = indicators.MACD(
            period_me1=self.p.fast_period,
            period_me2=self.p.slow_period,
            smoothing_period=self.p.smoothing_period,
        ).lines.smooth_macd
        self.high = bt.And(
            *[
                self.macd(-i) < self.macd(-i - 1)
                for i in range(self.p.consecutive_period)
            ]
        )
        bt.LinePlotterIndicator(self.high, name="high")
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if not self.position and self.williamsR[0] <= -90 and self.high:
            self.order_buy = self.buy()
        if self.position and not self.high:
            self.close()
