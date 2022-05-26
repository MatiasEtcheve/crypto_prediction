from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tools import inspect_code


class ScriptCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, dirpath="./", datamodule=None):
        self.dirpath = Path(dirpath)
        self.datamodule = datamodule

    def on_train_begin(self, logs=None):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        with open(Path(self.dirpath) / "model_script.txt", "w") as file:
            model_script = inspect_code.get_class_code(type(self.model))
            file.write(model_script)

        for attr in ["generator", "discriminator"]:
            if hasattr(self.model, attr):
                with open(Path(self.dirpath) / f"{attr}_script.txt", "w") as file:
                    generator_script = inspect_code.get_class_code(
                        type(getattr(self.model, attr))
                    )
                    file.write(generator_script)

        if self.datamodule is not None:
            filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
            with open(filename_datamodule, "w") as file:
                file.write(inspect_code.get_class_code(type(self.datamodule)))


def precision_recall_accuracy_metrics(
    predictions: Union[pd.DataFrame, np.ndarray],
    targets: Union[pd.DataFrame, np.ndarray],
) -> Tuple[float, float, float]:
    """Computes precision, recall and accuracy metrics

    Args:
        predictions (Union[pd.DataFrame, np.ndarray]): predictions, y_hat
        targets (Union[pd.DataFrame, np.ndarray]): targets, y

    Returns:
        Tuple[float, float, float]: tuple of precision, recall, accuracy metrics
    """
    if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
        predictions = predictions.to_numpy()
    if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
        targets = targets.to_numpy()
    recall = recall_score(
        targets.reshape(-1, 1), predictions.reshape(-1, 1), zero_division=0
    )
    precision = precision_score(
        targets.reshape(-1, 1), predictions.reshape(-1, 1), zero_division=0
    )
    accuracy = accuracy_score(
        targets.reshape(-1, 1),
        predictions.reshape(-1, 1),
    )
    return (
        precision,
        recall,
        accuracy,
    )
