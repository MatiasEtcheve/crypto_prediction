import backtrader as bt
import backtrader.indicators as btind


class Trix(bt.Indicator):
    lines = ("trix", "signal")
    params = (("period", 15), ("_rocperiod", 1), ("sigperiod", 9),)

    plotinfo = dict(plothlines=[0.0])

    def _plotlabel(self):
        plabels = [self.p.period]
        plabels += [self.p._rocperiod]
        return plabels

    def __init__(self):
        ema1 = btind.ExponentialMovingAverage(self.data, period=self.p.period)
        ema2 = btind.ExponentialMovingAverage(ema1, period=self.p.period)
        ema3 = btind.ExponentialMovingAverage(ema2, period=self.p.period)

        # 1 period Percentage Rate of Change
        self.lines.trix = 100.0 * (ema3 / ema3(-self.p._rocperiod) - 1.0)
        self.l.signal = btind.SimpleMovingAverage(self.lines.trix, period=self.p.sigperiod)
        super(Trix, self).__init__()
