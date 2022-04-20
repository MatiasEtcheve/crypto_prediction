import get_data
import numpy as np
import pandas as pd
from datasets import assets


class DataModule:
    def __init__(
        self,
        config,
        compute_metrics=None,
        csv_file=None,
        inputs=None,
    ):
        super().__init__()
        self.config = config
        self.compute_metrics = compute_metrics

        if csv_file is not None and inputs is None:
            df = pd.read_csv(csv_file, delimiter=";")
            df["beginning_date"] = pd.to_datetime(
                df.pop("beginning_date"), dayfirst=True, utc=True
            )
            df["ending_date"] = pd.to_datetime(
                df.pop("ending_date"), dayfirst=True, utc=True
            )
            self.inputs = df.to_dict("records")
        elif inputs is not None:
            self.inputs = inputs
        else:
            raise NotImplementedError("")

    def _get_data(
        self,
        ticker=None,
        beginning_date=None,
        ending_date=None,
        interval="12h",
    ):
        try:
            data = get_data.download_klines(
                ticker,
                interval,
                beginning_date=beginning_date,
                ending_date=ending_date,
                compute_metrics=self.compute_metrics,
            )
            return data
        except Exception as e:
            print(f"Ticker {ticker} could not be downloaded")
            print(f"\t{e}")
            return []

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
                if test_ratio != 1
                else 0
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
            )
            if dp == []:
                continue
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


def _concatenate_indicators(data):
    m = list(range(1, 21)) + list(range(40, 241, 20))
    for mi in [0] + m:
        data[f"ir_{mi}"] = data["Close"].shift(mi) / data["Open"].shift(mi) - 1
    for mi in m:
        data[f"cr_{mi}"] = data["Close"].shift(1) / data["Close"].shift(mi + 1) - 1
    for mi in m:
        data[f"or_{mi}"] = data["Open"] / data["Close"].shift(mi) - 1
    data["Direction"] = data["Close"] > data["Open"]
    return data


def create_asset(ticker, interval, beginning_date, ending_date, compute_metrics):
    # try:
    data = get_data.download_klines(
        ticker,
        interval,
        beginning_date=beginning_date,
        ending_date=ending_date,
        compute_metrics=compute_metrics,
    )
    # except Exception as e:
    #     print(f"Ticker {ticker} could not be downloaded")
    #     print(f"\t{e}")
    #     return []
    # else:
    # data = data.dropna(axis=0)
    data = data.replace(
        to_replace=[np.inf, -np.inf, np.float64("inf"), -np.float64("inf")],
        value=0,
    )
    labels = data.pop("Direction").to_numpy()

    return assets.TrainAsset(
        ticker=ticker,
        df=data,
        labels=labels,
        interval=interval,
        compute_metrics=compute_metrics,
    )
