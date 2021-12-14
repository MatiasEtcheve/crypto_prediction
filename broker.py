from collections import OrderedDict
from pprint import pprint
import matplotlib.pyplot as plt
import backtrader as bt
from pathlib import Path
from pprint import pprint
import json
from datetime import datetime
import numpy as np
import total as total
from backtrader.utils import AutoOrderedDict

import get_data

FORMAT = "%Y-%m-%d"
"""Expected datetime format"""

timeframes = {
    "12h": {"timeframe": bt.TimeFrame.Minutes, "compression": 720},
    "15m": {"timeframe": bt.TimeFrame.Minutes, "compression": 15},
    "1d": {"timeframe": bt.TimeFrame.Days, "compression": 1},
    "1h": {"timeframe": bt.TimeFrame.Minutes, "compression": 60},
    "1m": {"timeframe": bt.TimeFrame.Minutes, "compression": 1},
    "1M": {"timeframe": bt.TimeFrame.Months, "compression": 1},
    "1w": {"timeframe": bt.TimeFrame.Weeks, "compression": 1},
    "2h": {"timeframe": bt.TimeFrame.Minutes, "compression": 120},
    "30m": {"timeframe": bt.TimeFrame.Minutes, "compression": 30},
    "3d": {"timeframe": bt.TimeFrame.Days, "compression": 3},
    "3m": {"timeframe": bt.TimeFrame.Minutes, "compression": 3},
    "4h": {"timeframe": bt.TimeFrame.Minutes, "compression": 240},
    "5m": {"timeframe": bt.TimeFrame.Minutes, "compression": 5},
    "6h": {"timeframe": bt.TimeFrame.Minutes, "compression": 360},
    "8h": {"timeframe": bt.TimeFrame.Minutes, "compression": 480},
}


class Cerebro(bt.Cerebro):
    def __init__(
        self,
        symbol,
        interval,
        from_date,
        end_date,
        strategy,
        commission=0.001,
        plot_rules="equity",
        **kwargs,
    ):
        super().__init__()
        self.p.stdstats = "indicator" in plot_rules
        self.broker.setcash(100)
        self.from_date = from_date
        self.end_date = end_date
        self.interval = interval
        self.symbol = symbol
        self.strategies = (
            [strategy] if isinstance(strategy, bt.strategy.MetaStrategy) else strategy
        )
        self.plot_rules = plot_rules.lower()
        self.commission = commission
        df = get_data.select_data(symbol, interval, from_date, end_date)
        data = bt.feeds.PandasData(
            dataname=df,
            name=symbol + interval,
            **timeframes[interval],
        )
        data.plotinfo.plot = "bars" in self.plot_rules
        self.adddata(data)
        self.broker.setcommission(commission=self.commission)
        for strategy in self.strategies:
            self.addstrategy(strategy, **kwargs)

        self.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")
        self.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.addanalyzer(
            bt.analyzers.SharpeRatio, _name="sharpe", **timeframes[interval]
        )
        self.addanalyzer(bt.analyzers.SQN, _name="sqn")

        self.addobserver(bt.observers.Broker, plot="broker" in self.plot_rules)
        self.addobserver(bt.observers.Value, plot="equity" in self.plot_rules)
        self.addobserver(bt.observers.Trades, plot="trades" in self.plot_rules)
        self.addobserver(bt.observers.BuySell, plot="buysell" in self.plot_rules)

        self.addwriter(bt.WriterStringIO, csv=True)

    @property
    def get_analyzers(self):
        return dict(self.runstrats[0][0].analyzers.getitems())

    def _sqn2rating(self, sqn_score):
        """Converts sqn_score score to human readable rating"""
        if sqn_score < 1.6:
            return "Poor"
        elif sqn_score < 1.9:
            return "Below average"
        elif sqn_score < 2.4:
            return "Average"
        elif sqn_score < 2.9:
            return "Good"
        elif sqn_score < 5.0:
            return "Excellent"
        elif sqn_score < 6.9:
            return "Superb"
        else:
            return "Holy Grail"

    @property
    def kpi(self):
        dt = self.runstrats[0][0].data._dataname.index
        buy_n_hold = (
            self.runstrats[0][0].data.close[0]
            / self.runstrats[0][0].data.close[-len(self.runstrats[0][0].data) + 1]
            * self.broker.startingcash
        )
        trade_analysis = self.get_analyzers["trade"].get_analysis()
        try:
            exposure = trade_analysis.len.total / len(dt)
        except KeyError:
            exposure = None
        rpl = trade_analysis.pnl.net.total
        total_return = rpl / self.broker.startingcash
        total_number_trades = trade_analysis.total.total
        trades_closed = trade_analysis.total.closed
        bt_period = dt[-1] - dt[0]
        bt_period_days = bt_period.days
        drawdown = self.get_analyzers["drawdown"].get_analysis()
        sharpe_ratio = self.get_analyzers["sharpe"].get_analysis()["sharperatio"]
        sqn_score = self.get_analyzers["sqn"].get_analysis()["sqn"]
        return OrderedDict(
            {
                "initial cash": self.broker.startingcash,
                "ending value": self.broker.getvalue(),
                "net profit": self.broker.getvalue() - self.broker.startingcash,
                "net profit % +": (self.broker.getvalue() - self.broker.startingcash)
                / self.broker.startingcash,
                "rpl": rpl,
                "exposure": exposure,
                "buy and hold": buy_n_hold,
                "result_won_trades": trade_analysis.won.pnl.total,
                "result_lost_trades": trade_analysis.lost.pnl.total,
                "profit_factor": -1
                * trade_analysis.won.pnl.total
                / (trade_analysis.lost.pnl.total - 1e-12),
                "rpl_per_trade": rpl / trades_closed,
                "total_return": 100 * total_return,
                "annual_return": (
                    100 * (1 + total_return) ** (365.25 / bt_period_days) - 100
                ),
                "max_money_drawdown": drawdown["max"]["moneydown"],
                "max_pct_drawdown": drawdown["max"]["drawdown"],
                # trades
                "total_number_trades": total_number_trades,
                "trades_closed": trades_closed,
                "pct_winning": 100 * trade_analysis.won.total / trades_closed,
                "pct_losing": 100 * trade_analysis.lost.total / trades_closed,
                "avg_money_winning": trade_analysis.won.pnl.average,
                "avg_money_losing": trade_analysis.lost.pnl.average,
                "best_winning_trade": trade_analysis.won.pnl.max,
                "worst_losing_trade": trade_analysis.lost.pnl.max,
                #  performance
                "sharpe_ratio": sharpe_ratio,
                "sqn_score": sqn_score,
                "sqn_human": self._sqn2rating(sqn_score),
            }
        )

    def __repr__(self):
        res = ""
        for label, info in self.kpi.items():
            res += "\033[1m{:>19}\033[0m  ".format(label)
            res += "{}\n".format(info)
        return res

    def save(self, filename=None, folder=None):
        cerebroinfo = {}
        datainfos = {}

        for i, data in enumerate(self.datas):
            datainfos['Data%d' % i] = data.getwriterinfo()

        cerebroinfo['datas'] = datainfos

        stratinfos = dict()
        for strat in self.runstrats[0]:
            stname = strat.__class__.__name__
            stratinfos[stname] = strat.getwriterinfo()

        cerebroinfo['Strategies'] = stratinfos

        if filename is None:
            beginning_date = datetime(self.from_date.year, self.from_date.month, self.from_date.day)
            ending_date = datetime(self.end_date.year, self.end_date.month, self.end_date.day)
            filename = [self.symbol, self.interval, beginning_date.strftime(FORMAT),ending_date.strftime(FORMAT)]
            for stname in stratinfos.keys():
                filename.append(stname)
                for param_name, param_value in stratinfos[stname]["Params"].items():
                    filename.append(param_name)
                    filename.append(param_value)
            filename = "_".join([str(i) for i in filename]) + ".csv"

        writer_string_io = self.runwriters[0]
        dataframe = writer_string_io.out.getvalue().split("=" * writer_string_io.p.seplen)[1].strip()
        # print(dataframe.split("\n")[0].split(","))
        cerebroinfo['dataframe'] = dataframe

        if folder is None:
            save_folder = Path() / "data" / list(stratinfos.keys())[0]
        else:
            save_folder = Path() / "data" / Path(folder)

        save_folder.mkdir(parents=True, exist_ok=True)
        with open(save_folder / filename, "w") as outfile:
            json.dump(cerebroinfo, outfile)
