from .generic import GenericStrategy
import backtrader as bt
import indicators

# Trix
# https://www.youtube.com/watch?v=uE04UROWkjs&list=LL&index=15


class TRIXStoRSI(GenericStrategy):
    params = dict(
        trix_period=15,
        sig_period=9,
        stoch_period=14,
        stop_loss=0.01,
        trail=False,
        verbose=0,
        trend=True
    )

    def __init__(self):
        self.trend = bt.indicators.MovingAverageSimple(self.data.close, period=200)
        self.trix = indicators.Trix(self.data, period=self.p.trix_period, sigperiod=self.p.sig_period)
        self.stoch_rsi = indicators.StochRSI(self.data, period=self.p.stoch_period)
        self.trix_above_mean = self.trix.trix > self.trix.signal
        # bt.LinePlotterIndicator(self.truc, name="trix")
        self.setsizer(bt.sizers.PercentSizer(percents=99))

    def next(self):
        if not self.position and self.trix_above_mean and self.stoch_rsi <= 0.8:
            if self.p.trend and self.data > self.trend:
                self.order_buy = self.buy()
        if self.position and not self.trix_above_mean and self.stoch_rsi >= 0.2:
            self.close()
        # if self.position and not self.high:
        #     self.close()
