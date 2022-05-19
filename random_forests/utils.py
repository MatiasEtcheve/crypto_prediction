from pathlib import Path

import get_data
import numpy as np
import pandas as pd
from datasets import assets


class DataModule:
    def __init__(
        self,
        config,
        compute_metrics=None,
        inputs=None,
        save_klines=True,
    ):
        self.config = config
        self.compute_metrics = compute_metrics
        self.inputs = inputs
        self.save_klines = save_klines

    def _init_train_val_data(self, train_datapoints):
        test_ratio = self.config["train_val_test_split"][-1]
        try:
            features = np.concatenate([dp.features for dp in train_datapoints], axis=0)
            labels = np.concatenate([dp.labels for dp in train_datapoints], axis=0)

            p = np.random.permutation(len(features))
            features, labels = features[p], labels[p]
            train_size = (
                int(
                    self.config["train_val_test_split"][0]
                    / (1 - test_ratio)
                    * len(features)
                )
                if test_ratio != 0
                else self.config["train_val_test_split"][0]
            )
            train_dataset = (features[:train_size], labels[:train_size])
            val_dataset = (features[train_size:], labels[train_size:])
        except ValueError as e:
            train_dataset = (np.array([]), np.array([]))
            val_dataset = (np.array([]), np.array([]))
        return train_dataset, val_dataset

    def setup(self):
        train_val_ratio = 1 - self.config["train_val_test_split"][-1]
        self.test_datapoints = []
        train_datapoints = []
        for input in self.inputs:
            dp = create_asset(
                **input,
                interval=self.config["interval"],
                compute_metrics=self.compute_metrics,
                save_klines=self.save_klines,
            )
            if dp == []:
                continue
            dp.df = dp.df.dropna()
            dp.labels = dp.labels.dropna()

            common_index = dp.df.index.intersection(dp.labels.index)
            dp.df = dp.df.loc[common_index]
            dp.labels = dp.labels.loc[common_index].astype("bool")

            train_val_size = int(len(dp.df) * train_val_ratio)
            test_dp = assets.TrainAsset(
                ticker=input["ticker"],
                df=dp.df[train_val_size:],
                labels=dp.labels[train_val_size:],
                interval=self.config["interval"],
                compute_metrics=self.compute_metrics,
            )
            train_dp = assets.TrainAsset(
                ticker=input["ticker"],
                df=dp.df[:train_val_size],
                labels=dp.labels[:train_val_size],
                interval=self.config["interval"],
                compute_metrics=self.compute_metrics,
            )
            if not test_dp.isempty:
                self.test_datapoints.append(test_dp)
            if not train_dp.isempty:
                train_datapoints.append(train_dp)

        self.train_dataset, self.val_dataset = self._init_train_val_data(
            train_datapoints
        )


def _concatenate_indicators(data, percentage=0.1):
    m = list(range(1, 21)) + list(range(40, 241, 20))
    for mi in [0] + m:
        data[f"ir_{mi}"] = data["Close"].shift(mi) / data["Open"].shift(mi) - 1
    for mi in m:
        data[f"cr_{mi}"] = data["Close"].shift(1) / data["Close"].shift(mi + 1) - 1
    for mi in m:
        data[f"or_{mi}"] = data["Open"] / data["Close"].shift(mi) - 1
    # positive_mask = data["Close"] > data["Open"] * (1 + percentage)
    # following_day_mask = positive_mask.shift(1, fill_value=False)
    # data["Direction"] = False
    # data["Direction"].loc[following_day_mask] = (
    #     data[following_day_mask]["Close"] > data[following_day_mask]["Open"]
    # )
    data["Direction"] = data["Close"] > data["Open"]
    return data


def create_asset(
    ticker,
    interval,
    beginning_date,
    ending_date,
    compute_metrics=lambda x: x,
    save_klines=False,
):
    if save_klines:
        data = get_data.select_data(
            ticker,
            interval,
            beginning_date=beginning_date,
            ending_date=ending_date,
            compute_metrics=compute_metrics,
            directory=Path(__file__).resolve().parent / "tmp" / "klines",
        )
    else:
        data = get_data.download_data(
            ticker,
            interval,
            beginning_date=beginning_date,
            ending_date=ending_date,
            compute_metrics=compute_metrics,
        )
    data = data.replace(
        to_replace=[np.inf, -np.inf, float("inf"), float("inf")],
        value=0,
    )
    labels = data.pop("Direction")

    return assets.TrainAsset(
        ticker=ticker,
        df=data,
        labels=labels,
        interval=interval,
        compute_metrics=compute_metrics,
    )
