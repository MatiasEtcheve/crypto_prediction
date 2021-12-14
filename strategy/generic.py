import math
import os.path
import sys
from datetime import date, datetime, time, timedelta

import backtrader as bt

class GenericStrategy(bt.Strategy):
    params = dict(stop_loss=0.02, trail=False, verbose=1)

    def start(self):
        self.order_buy = None
        self.order_sell = None
        self.sell_on_stop = 0
        self.sell_on_signal = 0

    def log(self, txt, min_verbose):
        if self.p.verbose >= min_verbose:
            print(txt)

    def notify_order(self, order):
        # order_type = [
        #     "Market",
        #     "Close",
        #     "Limit",
        #     "Stop",
        #     "StopLimit",
        #     "StopTrail",
        #     "StopTrailLimit",
        #     "Historical",
        # ]
        if not order.status == order.Completed:
            return  # discard any other notification

        if order.issell():  # we left the market
            self.cancel(self.order_sell)
            # self.order_sell = None
            if order.exectype in [3, 4, 5, 6]:  # Hit stop loss
                self.sell_on_stop += 1
                self.log(
                    f"STOP LOSS@price: {order.executed.price} at {self.datetime.time()}",
                    1,
                )
            else:
                self.sell_on_signal += 1
                self.log(
                    f"SELL@price: {order.executed.price} at {self.datetime.time()}", 1
                )
            return

        if order.isbuy():  # we enter the market
            self.log(f"BUY@price: {order.executed.price} at {self.datetime.time()}", 1)
            # self.order_buy = None
            # self.cancel(self.order_sell)
            if self.p.stop_loss:
                if not self.p.trail:
                    stop_price = order.executed.price * (1.0 - self.p.stop_loss)
                    self.order_sell = self.sell(exectype=bt.Order.Stop, price=stop_price)
                else:
                    self.order_sell = self.sell(
                        exectype=bt.Order.StopTrail, trailamount=self.p.trail
                    )

    def notify_trade(self, trade):
        profitPercAbs = trade.history[-1].event.order.executed.price / trade.price
        profitPercAbs = 1 - profitPercAbs if trade.history[-1].event.order.executed.size > 0 else profitPercAbs - 1
        self.log(f"TRADE PROFIT %: {profitPercAbs*100}", 1)

    def stop(self):
        self.log(f"STOP LOSS HIT: {self.sell_on_stop}", 1)
        self.log(f"SIGNAL HIT: {self.sell_on_signal}", 1)