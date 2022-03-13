from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime, timedelta, time, date  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt


def create_time(hour=0, minute=0, second=0):
    total_seconds = second + minute * 60 + hour * 3600
    if abs(total_seconds) > 24 * 60 * 60:
        raise ValueError("Can not create time from this parameter, exceed 24hours")
    hours = (total_seconds // 60) // 60 % 24
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return time(second=seconds, minute=minutes, hour=hours)


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (("opening_time", 5), ("maperiod", 6))

    def log(self, txt, dt=None, doprint=False):
        """Logging function fot this strategy"""
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt}, {txt}")

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )

        self.above_sma = self.sma > self.data.lines.open

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    "BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    "SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f"
                    % (order.executed.price, order.executed.value, order.executed.comm)
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log("Close, %.2f" % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if (
                create_time(minute=self.p.opening_time - 2 * 60)
                <= self.data.datetime.time()
                <= create_time(minute=self.p.opening_time)
                and not self.above_sma[0]
            ):
                self.log("BUY CREATE, %.2f" % self.dataclose[0])
                self.order = self.buy()

        else:

            if (
                create_time(minute=self.p.opening_time)
                <= self.data.datetime.time()
                <= create_time(minute=self.p.opening_time + 2 * 60)
                and self.above_sma[0]
            ):
                self.log("SELL CREATE, %.2f" % self.dataclose[0])
                self.order = self.sell()

    def stop(self):
        self.log(f"Opening time: {create_time(minute=self.p.opening_time)}, {self.broker.getvalue()}", doprint=True)

class MACrossover(bt.Strategy):
    def __init__(self):
        sma_long = bt.indicators.MovingAverageSimple(self.data, period=200)
        sma_short = bt.indicators.MovingAverageSimple(self.data, period=50)
        self.cross = bt.indicators.CrossOver(sma_short, sma_long)
        self.setsizer(bt.sizers.PercentSizer(percents=50))

    def next(self):
        if self.cross[0] >= 1 and not self.position:
            self.buy()
        if self.cross[0] <= -1 and self.position:
            self.sell()

class Hodl(bt.Strategy):
    def __init__(self):
        print(len(self.data))

    def next(self):
        # print(bool(self.position))
        if not self.position and len(self)<400:
            self.buy(size=0.001)
            # print(self.position, bool(self.position))
            # raise ValueError
        if len(self) > 400 and self.position:
            self.sell(size=0.001)
