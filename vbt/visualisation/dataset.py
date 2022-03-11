import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

observer_columns = {
    "Broker": ["broker", "broken_len", "broker_cash", "broker_value"],
    "Value": ["value", "value_len", "value_value"],
    "Trades - Net Profit/Loss": [
        "trades",
        "trades_len",
        "trades_pnlplus",
        "trades_pnlminus",
    ],
    "BuySell": ["buysell", "buysell_len", "buysell_buy", "buysell_sell"],
}

class Datapoint(object):
    def __init__(self, data_file_path):
        file = open(data_file_path)
        self.data = json.loads(file.read())

        self.name = data_file_path.stem
        self.strategies = list(self.data["Strategies"].keys())
        self.params = self.data["Strategies"][self.strategies[0]]["Params"]
        self.analyzers = self.data["Strategies"][self.strategies[0]]["Analyzers"]

        self._kpi = None
        self._dataframe = None

        self.return_kpi = True

    def build_dataframe(self):
        for stname in self.strategies:
            observer_columns[stname] = [stname, stname + "_len", stname + "_datetime"]

        self._dataframe = pd.read_csv(StringIO(self.data["dataframe"]), sep=",")
        for index_col, col in enumerate(self._dataframe.columns):
            for i, col_overriding_name in enumerate(observer_columns.get(col, [])):
                self._dataframe.rename(
                    columns={
                        self._dataframe.columns[index_col + i]: col_overriding_name
                    },
                    inplace=True,
                )

    def build_kpi(self):
        equity = self.dataframe["broker_value"].to_numpy(copy=True)
        return_rate = equity[1:] / equity[:-1]
        return_rate = np.where(return_rate<0, 1, return_rate)
        log_return = np.log(return_rate)

        value_analyzer = self.analyzers["Value"]
        drawdown_analyzer = self.analyzers["drawdown"]["Analysis"]
        sharpe_analyzer = self.analyzers["sharpe"]["Analysis"]
        sqn_analyzer = self.analyzers["sqn"]["Analysis"]
        trade_analyzer = self.analyzers["trade"]["Analysis"]
        profit_factor = (
                trade_analyzer.get("won", {}).get("total", 0) / trade_analyzer.get("total", {}).get("total", 1)
            ) if trade_analyzer.get("total", {}).get("total", 1) != 0 else 0
        self._kpi = {
            "exposure": trade_analyzer.get("len", {}).get("total", 0) / len(self.dataframe),
            "volatility": np.std(log_return[log_return != 1]),
            "sqn_ratio": sqn_analyzer["sqn"],
            "profit_factor": profit_factor,
            "max_drawdown": drawdown_analyzer.get("max", {}).get("drawdown", 0) / 100,
            "net profit %": value_analyzer["End"] / value_analyzer["Begin"],
        }

    @property
    def kpi(self):
        if self.return_kpi:
            if self._kpi is None:
                self.build_kpi()
            return self._kpi
        else:
            try:
                return self._normalized_kpi
            except AttributeError:
                print("You need to normalize first boi")
            return self.kpi

    @property
    def dataframe(self):
        if self._dataframe is None:
            self.build_dataframe()
        return self._dataframe

    def __repr__(self):
        return self.name


@dataclass
class Point:
    name: str = ""
    value: float = 0


class Dataset:
    def isin(self, CONTAINS: Union[str, List[str]], word: str) -> bool:
        if isinstance(CONTAINS, bool):
            return CONTAINS
        if isinstance(CONTAINS, str):
            CONTAINS = list(CONTAINS)
        for c in CONTAINS:
            if c not in word:
                return False
        return True

    def __init__(self, data, CONTAINS=True, NOT_CONTAINS=False):
        if isinstance(data, list) and all(isinstance(x, Datapoint) for x in data):
            self.datapoints = data
        elif isinstance(data, Path):
            data_file_paths = list(data.glob("*.csv"))
            self.datapoints = [
                Datapoint(data_file_path)
                for data_file_path in data_file_paths
                if self.isin(CONTAINS, str(data_file_path))
                and not self.isin(NOT_CONTAINS, str(data_file_path))
            ]
        else:
            raise ValueError("wrong constructor for dataset")

        self._order = {}
        self.return_kpi = True

    def compute_order(self):
        for index, datapoint in enumerate(self.datapoints):
            for kpi_name, kpi_value in datapoint.kpi.items():
                self._order[kpi_name] = self._order.get(kpi_name, []) + [
                    Point(name=datapoint.name, value=kpi_value)
                ]
        for kpi_name, order in self._order.items():
            self._order[kpi_name] = sorted(
                self._order[kpi_name],
                key=lambda x: x.value,
            )

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Dataset([self[ii] for ii in range(*key.indices(len(self)))])
        elif isinstance(key, int):
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            return self.datapoints[key]
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        return self.datapoints.__iter__()

    def __len__(self):
        return len(self.datapoints)

    def __add__(self, other):
        return Dataset(self.datapoints+other.datapoints)

    @property
    def order(self):
        if not self._order:
            self.compute_order()
        return self._order

    def return_kpi(self, x: bool):
        for index, datapoint in enumerate(self.datapoints):
            datapoint.return_kpi = x

    @property
    def kpi(self):
        return [datapoint.kpi for datapoint in self]

    def normalize_kpis(
        self,
        keys_to_normalize=["net profit %", "volatility", "sqn_ratio"],
    ):
        for index, datapoint in enumerate(self.datapoints):
            normalized_kpi = datapoint.kpi.copy()
            for kpi_name in sorted(normalized_kpi.keys()):
                if kpi_name in keys_to_normalize:
                    kpi_value = normalized_kpi.pop(kpi_name)
                    try:
                        max_val = self.order[kpi_name][-1].value
                        min_val = self.order[kpi_name][0].value
                        kpi_value = (kpi_value - min_val) / (max_val - min_val)
                    except ZeroDivisionError:
                        kpi_value = 0
                    normalized_kpi[kpi_name] = kpi_value
            self.datapoints[index]._normalized_kpi = normalized_kpi
            datapoint.return_kpi = False

    def sort(self, by=None, reverse=True):
        if by is None:
            by = list(self.kpi[0].keys())[0]
        self.datapoints = sorted(
            self.datapoints, key=lambda x: x.kpi[by], reverse=reverse
        )