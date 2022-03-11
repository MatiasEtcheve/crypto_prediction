import backtrader as bt

class StochRSI(bt.Indicator):
    lines = ('stochrsi',)
    params = dict(
        period=14,  # to apply to RSI
        pperiod=None,  # if passed apply to HighestN/LowestN, else "period"
    )

    def __init__(self):
        rsi = bt.ind.RSI(self.data, period=self.p.period)

        pperiod = self.p.pperiod or self.p.period
        maxrsi = bt.ind.Highest(rsi, period=3)
        minrsi = bt.ind.Lowest(rsi, period=3)

        self.l.stochrsi = (rsi - minrsi) / (maxrsi - minrsi + 1e-6)