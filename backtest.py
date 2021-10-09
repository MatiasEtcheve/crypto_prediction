import get_data
import backtrader as bt
from datetime import timedelta, datetime

class PandasData(bt.feed.DataBase):
    '''
    Uses a Pandas DataFrame as the feed source, using indices into column
    names (which can be "numeric")

    This means that all parameters related to lines must have numeric
    values as indices into the tuples

    Params:

      - ``nocase`` (default *True*) case insensitive match of column names

    Note:

      - The ``dataname`` parameter is a Pandas DataFrame

      - Values possible for datetime

        - None: the index contains the datetime
        - -1: no index, autodetect column
        - >= 0 or string: specific colum identifier

      - For other lines parameters

        - None: column not present
        - -1: autodetect
        - >= 0 or string: specific colum identifier
    '''

    params = (
        ('nocase', True),

        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', 0),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', 6),
    )


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
interval = "12h"
symbol = "DOGEUSDT"
df = get_data.select_data(symbol, interval, from_date, end_date)
print(df.dtypes)
print(df)
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.addstrategy(RSIStrategy)
print(cerebro.broker.getvalue())
cerebro.run()
print(cerebro.broker.getvalue())
cerebro.plot()