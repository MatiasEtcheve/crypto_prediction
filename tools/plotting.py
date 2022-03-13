import math
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from black import out


def plot_hist(
    targets: Union[pd.DataFrame, np.ndarray],
    predictions: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    nb_hist_per_line: int = 2,
    columns: Optional[List[str]] = None,
    targets_kwargs={},
    predictions_kwargs={},
):
    if isinstance(targets, pd.DataFrame):
        if columns is None:
            columns = targets.columns
        targets = targets.copy().to_numpy()

    if isinstance(predictions, pd.DataFrame):
        if columns is None:
            columns = predictions.columns
        predictions = predictions.copy().to_numpy()

    nb_features = targets.shape[1]
    height = int(math.ceil(nb_features / nb_hist_per_line))
    fig, axs = plt.subplots(
        height,
        nb_hist_per_line,
        figsize=(
            6 * nb_hist_per_line,
            6 * height,
        ),
    )
    for i in range(nb_features):
        sns.histplot(
            data=targets[:, i],
            ax=axs[i // nb_hist_per_line, i % nb_hist_per_line],
            label="target",
            color="blue",
            kde=False,
            **targets_kwargs,
        )
        if predictions is not None:
            sns.histplot(
                data=predictions[:, i],
                ax=axs[i // nb_hist_per_line, i % nb_hist_per_line],
                label="prediction",
                color="orange",
                kde=False,
                **predictions_kwargs,
            )
            axs[i // nb_hist_per_line, i % nb_hist_per_line].legend()
        if columns is not None:
            axs[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {columns[i]}"
            )
    return fig, axs


def plot_kde(
    data: List[Union[pd.DataFrame, np.ndarray]],
    nb_hist_per_line: int = 2,
    columns: Optional[List[str]] = None,
    labels=["targets", "predictions"],
):
    if isinstance(data, pd.DataFrame):
        if columns is None:
            columns = data.columns
        data = [data.copy().to_numpy()]

    if isinstance(data, list):
        for index, d in enumerate(data):
            if isinstance(d, pd.DataFrame):
                if columns is None:
                    columns = d.columns
                data[index] = d.copy().to_numpy()

    nb_features = data[0].shape[1]
    height = int(math.ceil(nb_features / nb_hist_per_line))
    fig, axs = plt.subplots(
        height,
        nb_hist_per_line,
        figsize=(
            6 * nb_hist_per_line,
            6 * height,
        ),
    )
    for i in range(nb_features):
        for index, d in enumerate(data):
            sns.kdeplot(
                data=d[:, i],
                shade=True,
                ax=axs[i // nb_hist_per_line, i % nb_hist_per_line],
                label=labels[index],
            )

            axs[i // nb_hist_per_line, i % nb_hist_per_line].legend()
        if columns is not None:
            axs[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {columns[i]}"
            )
    return fig, axs


def plot_pairplot(
    targets: Union[pd.DataFrame, np.ndarray],
    predictions: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    columns: Optional[List[str]] = None,
):
    if isinstance(targets, np.ndarray):
        targets_df = pd.DataFrame(targets, columns=columns)
    else:
        targets_df = targets.copy()
    if predictions is not None:
        if isinstance(predictions, np.ndarray):
            predictions_df = pd.DataFrame(predictions, columns=columns)
        else:
            predictions_df = predictions.copy()
        predictions_df["type"] = "predictions"
        targets_df["type"] = "targets"
        outputs = pd.concat([predictions_df, targets_df], ignore_index=True)
    else:
        outputs = targets_df
    sns_plot = sns.pairplot(
        data=outputs,
        diag_kind="kde",
        hue="type",
    )
    return sns_plot


def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(15, 10))
    img = ax.matshow(df.corr())
    ax.set_xticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_xticklabels(
        df.select_dtypes(["number"]).columns,
        fontsize=14,
        rotation=90,
    )
    ax.set_yticks(range(df.select_dtypes(["number"]).shape[1]))
    ax.set_yticklabels(
        df.select_dtypes(["number"]).columns,
        fontsize=14,
    )
    cb = plt.colorbar(img, ax=ax)
    cb.ax.tick_params(labelsize=14)
    ax.set_title("Correlation Matrix", fontsize=16)
    return fig, ax
