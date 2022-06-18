from functools import partial
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from tensorflow.keras.metrics import Accuracy as _accuracy
from tensorflow.keras.metrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from tensorflow.keras.metrics import Precision as _precision
from tensorflow.keras.metrics import Recall as _recall
from tensorflow.keras.metrics import RootMeanSquaredError


def MaxError(y_true, y_pred):
    return tf.math.reduce_max(tf.math.abs(y_true - y_pred))


@tf.function
def Precision(y_true, y_pred):
    if y_true.dtype.is_integer:
        return _precision(y_true, y_pred)
    else:
        y_true_bool = tf.math.greater(tf.squeeze(y_true[1:]), tf.squeeze(y_true[:-1]))
        y_pred_bool = tf.math.greater(tf.squeeze(y_pred[1:]), tf.squeeze(y_pred[:-1]))
        y_true_bool.set_shape([None])
        y_pred_bool.set_shape([None])
        tp = tf.math.reduce_sum(tf.cast(y_true_bool[y_pred_bool], tf.float32))
        return tp / tf.math.reduce_sum(tf.cast(y_pred_bool, tf.float32))


@tf.function
def Recall(y_true, y_pred):
    if y_true.dtype.is_integer:
        return _recall(y_true, y_pred)
    else:
        y_true_bool = tf.math.greater(tf.squeeze(y_true[1:]), tf.squeeze(y_true[:-1]))
        y_pred_bool = tf.math.greater(tf.squeeze(y_pred[1:]), tf.squeeze(y_pred[:-1]))
        y_true_bool.set_shape([None])
        y_pred_bool.set_shape([None])
        tp = tf.math.reduce_sum(tf.cast(y_pred_bool[y_true_bool], tf.float32))
        fn = tf.math.reduce_sum(1 - tf.cast(y_pred_bool[y_true_bool], tf.float32))
        return tp / (tp + fn)


def mae(tf_metric=False):
    if tf_metric:
        return MeanAbsoluteError(name="mae")
    return mean_absolute_error


def mse(tf_metric=False):
    if tf_metric:
        return MeanSquaredError(name="mse")
    return mean_squared_error


def rmse(tf_metric=False):
    if tf_metric:
        return RootMeanSquaredError(name="rmse")
    return partial(mean_squared_error, squared=False)


def mape(tf_metric=False):
    if tf_metric:
        return MeanAbsolutePercentageError(name="mape")
    return mean_absolute_percentage_error


def me(tf_metric=False):
    if tf_metric:
        return MaxError
    return max_error


def precision(tf_metric=False):
    if tf_metric:
        return Precision
    return precision_score


def recall(tf_metric=False):
    if tf_metric:
        return Recall
    return recall_score


tf_classification_metrics = [
    precision(tf_metric=True),
    recall(tf_metric=True),
]

tf_regression_metrics = [
    mae(tf_metric=True),
    mse(tf_metric=True),
    rmse(tf_metric=True),
    mape(tf_metric=True),
    me(tf_metric=True),
    precision(tf_metric=True),
    recall(tf_metric=True),
]


def regression_metrics(
    targets: Union[pd.DataFrame, np.ndarray],
    pred: Union[pd.DataFrame, np.ndarray],
    metrics: List[str] = ["mae", "mse", "rmse", "me", "mape", "classification"],
) -> Dict[str, float]:
    if isinstance(pred, pd.DataFrame) or isinstance(pred, pd.Series):
        pred = pred.to_numpy()
    if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
        targets = targets.to_numpy()

    _metrics = {}
    for metric_name in metrics:
        if metric_name == "classification":
            classification_pred = pred[:-1] < pred[1:]
            classification_targets = targets[:-1] < targets[1:]
            _metrics.update(
                classification_metrics(
                    targets=classification_targets,
                    pred=classification_pred,
                )
            )
        else:
            _metrics[metric_name] = globals()[metric_name]()(targets, pred)
    return _metrics


def classification_metrics(
    targets: Union[pd.DataFrame, np.ndarray],
    pred: Union[pd.DataFrame, np.ndarray],
    metrics: List[str] = [
        "precision",
        "recall",
    ],
) -> Dict[str, float]:
    if isinstance(pred, pd.DataFrame) or isinstance(pred, pd.Series):
        pred = pred.to_numpy()
    if isinstance(targets, pd.DataFrame) or isinstance(targets, pd.Series):
        targets = targets.to_numpy()
    _metrics = {}
    for metric_name in metrics:
        _metrics[metric_name] = globals()[metric_name]()(targets, pred)
    return _metrics
