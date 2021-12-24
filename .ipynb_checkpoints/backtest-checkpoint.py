from datetime import datetime, timedelta
import vectorbt as vbt
from broker import Cerebro
from strategy import trix
from get_data import select_data
from talib import TRIX as tr

vbt.MA.run().ma.ma_crossed_above

from_date = datetime.now() - timedelta(days=365)
# from_date = datetime(2021,9, 18) - timedelta(minutes=30*200)
end_date = datetime.now()
# end_date = datetime(2021,9, 23)
interval = "30m"
symbol = "BTCUSDT"
price = select_data(symbol, interval, from_date, end_date)
trix = vbt.talib("TRIX", timeperiod=5)
print(trix)