import backtrader as bt

class MACD(bt.Indicator):
    lines = ("smooth_macd", "signal", "smooth_histo")
    params = (
        ("smoothing_period", 5),
        ("period_me1", 12),
        ("period_me2", 26),
        ("period_signal", 9),
    )
    plotinfo = dict(plothlines=[0.0])
    plotlines = dict(
        smooth_macd=dict(_method="bar", alpha=0.50, width=1.0),
        signal=dict(ls="--"),
        smooth_histo=dict(_method="bar", alpha=0.50, width=1.0),
    )

    def __init__(self):
        super(MACD, self).__init__()
        me1 = bt.indicators.ExponentialMovingAverage(
            self.data, period=self.p.period_me1
        )
        me2 = bt.indicators.ExponentialMovingAverage(
            self.data, period=self.p.period_me2
        )
        macd = me1 - me2
        self.lines.smooth_macd = bt.indicators.MovingAverageSimple(
            macd, period=self.p.smoothing_period
        )
        self.lines.signal = bt.indicators.ExponentialMovingAverage(
            macd, period=self.p.period_signal
        )
        self.lines.smooth_histo = bt.indicators.MovingAverageSimple(
            macd - self.lines.signal
        )
