 from datetime import datetime, timedelta
import backtrader as bt
from broker import Cerebro
from strategy import trix

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
# from_date = datetime(2021,9, 18) - timedelta(minutes=30*200)
end_date = datetime.now()
# end_date = datetime(2021,9, 23)
interval = "30m"
symbol = "BTCUSDT"
for stop_loss in [0.02, 0.05, 0.15, 0.30]:
    for trix_period in [15, 25, 66, 100, 150]:
        for stoch_period in [15, 25, 66, 100, 150]:
            print(f"STOP LOSS: {stop_loss}")
            print(f"TRIX PERIOD: {trix_period}")
            print(f"STOCH PERIOD: {stoch_period}")
            cerebro = Cerebro(
                symbol,
                interval,
                from_date,
                end_date,
                trix.TRIXStoRSI,
                stop_loss=stop_loss,
                trix_period=trix_period,
                stoch_period=stoch_period,
                trend=True,
                plot_rules="equity bars trades buysell",
                tradehistory=True,
                # verbose=1,
            )
            results = cerebro.run()
            cerebro.save()
            print(cerebro)
            # cerebro.plot(
            #     volume=False,
            #     # style="candle",
            #     # barup="green",
            #     # start=datetime(2021, 2, 13),
            #     # end=datetime(2021, 2, 26),
            # )
            # raise ValueError
            print("=================================")
