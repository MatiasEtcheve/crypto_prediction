from get_data import *
import backtrader as bt

class RSIStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = bt.talib.RSI(self.data, period=14)
    
    def next(self):
        if self.rsi < 40 and not self.position:
            self.buy(size=0.1)
        if self.rsi > 70 and self.position:
            self.close()

cerebro = bt.Cerebro()

from_date = datetime.now() - timedelta(days=8*170)
end_date = datetime.now()
interval = client.KLINE_INTERVAL_12HOUR
symbol = "DOGEUSDT"
file_name = get_data(symbol, interval, from_date, end_date)

data = bt.feeds.GenericCSVData(dataname=file_name, dtformat=1, compression=12*60, timeframe=bt.TimeFrame.Minutes)
cerebro.adddata(data)
cerebro.addstrategy(RSIStrategy)
print(cerebro.broker.getvalue())
cerebro.run()
print(cerebro.broker.getvalue())
cerebro.plot()