import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots

import metrics as _metrics
from tools import dataframe


def _init_plot(base_datapoints, max_plots):
    subplot_height = 300
    fig = make_subplots(
        rows=max_plots,
        cols=1,
        subplot_titles=[dp.ticker for dp in base_datapoints.values()],
        horizontal_spacing=0.0001,
        vertical_spacing=0.003,
        shared_xaxes=True,
    )
    fig.update_layout(
        height=subplot_height * max_plots,
        width=1000,
        margin=dict(l=10, r=20, t=30, b=10),
        legend_tracegroupgap=subplot_height,
    )
    return fig


def regression_plot(base_datapoints, max_plots=None):
    if max_plots is None:
        max_plots = len(base_datapoints.keys())

    fig = _init_plot(base_datapoints, max_plots)

    for index, dp in enumerate(base_datapoints.values()):
        if index >= max_plots:
            break
        predictions = dp.predictions
        labels = dp.labels

        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=labels,
                line=dict(color="black", width=1),
                name=f"{dp.ticker} close",
                legendgroup=str(index + 1),
            ),
            row=index + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=np.squeeze(predictions),
                line=dict(color="red", width=1),
                name=f"{dp.ticker} predictions",
                legendgroup=str(index + 1),
            ),
            row=index + 1,
            col=1,
        )

    return fig


def beginning_end_predictions(
    predictions,
    df,
    interval,
    lag,
    nan_df=True,
):
    idx_positive_predictions = predictions[predictions].index
    while (
        idx_positive_predictions[-1] + dataframe.convert_to_timedelta(interval, lag)
        > df.index[-1]
    ):
        idx_positive_predictions = idx_positive_predictions[:-1]
    beginning_predictions_close = df.loc[idx_positive_predictions]["Close"].to_frame()
    ending_predictions_close = df.loc[
        idx_positive_predictions + dataframe.convert_to_timedelta(interval, lag)
    ]["Close"].to_frame()
    if nan_df:
        nan_df = pd.DataFrame(
            [np.nan] * len(ending_predictions_close),
            index=ending_predictions_close.index,
            columns=["Close"],
        )
        return beginning_predictions_close, ending_predictions_close, nan_df
    return beginning_predictions_close, ending_predictions_close


def mask_segment(beginning_predictions_close, ending_predictions_close, nan_df, mask):
    temp_beginning_predictions = beginning_predictions_close[mask]
    temp_beginning_predictions.loc[:, "num_positive_pred"] = np.arange(
        len(temp_beginning_predictions)
    )
    temp_ending_predictions = ending_predictions_close[mask]
    temp_ending_predictions.loc[:, "num_positive_pred"] = (
        np.arange(len(temp_ending_predictions)) + 0.3
    )
    temp_nan_df = nan_df[mask]
    temp_nan_df.loc[:, "num_positive_pred"] = np.arange(len(temp_nan_df)) + 0.6

    return pd.concat(
        [temp_beginning_predictions, temp_ending_predictions, temp_nan_df], axis=0
    ).sort_values(by="num_positive_pred", axis=0)


def prediction_segments(predictions, df, interval, lag):
    (
        beginning_predictions_close,
        ending_predictions_close,
        nan_df,
    ) = beginning_end_predictions(predictions, df, interval, lag, nan_df=True)

    is_success = (
        ending_predictions_close["Close"].to_numpy()
        > beginning_predictions_close["Close"].to_numpy()
    )

    positive_segment = mask_segment(
        beginning_predictions_close, ending_predictions_close, nan_df, is_success
    )
    negative_segment = mask_segment(
        beginning_predictions_close, ending_predictions_close, nan_df, ~is_success
    )

    return positive_segment, negative_segment


def classification_plot(
    base_datapoints, interval, lag, threshold=None, metrics=False, max_plots=None
):
    if metrics:
        current_metrics = []
    if max_plots is None:
        max_plots = len(base_datapoints.keys())

    fig = _init_plot(base_datapoints, max_plots)

    for index, (ticker, dp) in enumerate(base_datapoints.items()):
        if index >= max_plots:
            break

        labels = dp.labels
        df = dp.df
        if threshold is not None and "probabilities" in dp.predictions.columns:
            predictions = dp.predictions["probabilities"]
        else:
            predictions = dp.predictions["predictions"].astype(bool)
        if metrics:
            current_metrics.append(_metrics.classification_metrics(labels, predictions))

        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=df.loc[labels.index]["Close"],
                line=dict(color="black", width=1),
                name=f"{dp.ticker} close",
                legendgroup=str(index + 1),
            ),
            row=index + 1,
            col=1,
        )

        positive_segment, negative_segment = prediction_segments(
            predictions,
            df,
            interval,
            lag,
        )

        for color, segment in zip(
            ["green", "red"], [positive_segment, negative_segment]
        ):
            fig.add_trace(
                go.Scatter(
                    x=segment.index,
                    y=segment["Close"],
                    line=dict(color=color, width=2),
                    name=f"{color} predictions",
                    legendgroup=str(index + 1),
                ),
                row=index + 1,
                col=1,
            )
    if metrics:
        return current_metrics, fig
    return fig


import math


def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range=None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
):

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = tf.stack(tensor, dim=0)

    if len(tensor.shape) == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if len(tensor.shape) == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = tf.concat((tensor, tensor, tensor), 0)
        tensor = tf.expand_dims(tensor, 0)

    if len(tensor.shape) == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = tf.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tf.identity(tensor)  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(
                value_range, tuple
            ), "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img = tf.clip_by_value(img, low, high)
            img = (img - low) / max(high - low, 1e-5)
            print(tf.math.reduce_min(img), tf.math.reduce_max(img))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(tf.math.reduce_min(t)), float(tf.math.reduce_max(t)))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tf.squeeze(tensor)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = tf.fill(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value
    ).numpy()
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[
                :,
                y * height + padding : (y + 1) * height,
                x * width + padding : (x + 1) * width,
            ] = tensor[k]
            k = k + 1
    return grid
