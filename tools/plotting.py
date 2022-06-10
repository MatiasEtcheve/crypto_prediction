import math
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_hist(
    data: List[Union[pd.DataFrame, np.ndarray]],
    nb_hist_per_line: int = 2,
    columns: Optional[List[str]] = None,
    labels: Optional[List[str]] = ["targets", "predictions"],
) -> Tuple[Figure, Axes]:
    """Plots histograms of `data`

    Example::
        targets = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        predictions = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        fig, axs = plot_hist([targets, predictions], nb_hist_per_line=3)

    Args:
        data (List[Union[pd.DataFrame, np.ndarray]]): List of data to plot.
            This data can be a `pd.DataFrame`, where each column will be a feature, and a row is an observation.
            This data can be a `np.ndarray`, where each column will a feature, and a row will be an observation.
            Each `pd.DataFrame` / `np.ndarray` in the list must have the same  shape, with the same columns names.
            The plot of each `pd.DataFrame` / `np.ndarray` will be overlapped.
        nb_hist_per_line (int, optional): Number of histograms to plot per row. Defaults to 2.
        columns (Optional[List[str]], optional): Feature's names.
            If `columns` is `None` and at least one array in `data` is a `pd.DataFrame`, the column names will be inferred from this array.
            If `columns` is not `None`, it must be a list of size the number of features in each array in `data`.
            Defaults to None.
        labels (Optional[List[str]], optional): Labels of each in array in `data\. Defaults to ["targets", "predictions"].

    Returns:
        Tuple[Figure, Axes]: figure and axes containing the plots
    """

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
    if len(data) < 10:
        colors = ["orange", "blue", "magenta", "green", "cyan", "pink"]
    else:
        colors = np.random.rand(len(data), 3)
    for i in range(nb_features):
        for index, d in enumerate(data):
            sns.histplot(
                data=d[:, i],
                ax=axs[i // nb_hist_per_line, i % nb_hist_per_line],
                label=labels[index],
                color=colors[index],
            )

            axs[i // nb_hist_per_line, i % nb_hist_per_line].legend()
        if columns is not None:
            axs[i // nb_hist_per_line, i % nb_hist_per_line].set_title(
                f"Histogram of {columns[i]}"
            )

    for row in axs:
        for ax in row:
            if not ax.collections:
                ax.set_visible(False)
            index += 1

    return fig, axs


def plot_kde(
    data: List[Union[pd.DataFrame, np.ndarray]],
    nb_hist_per_line: int = 2,
    columns: Optional[List[str]] = None,
    labels: Optional[List[str]] = ["targets", "predictions"],
) -> Tuple[Figure, Axes]:
    """Plots kernel density estimations of `data`

    Example::
        targets = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        predictions = pd.DataFrame(...,
            columns=["feature_A", "feature_B", "feature_C"],
            index=["observation_1", "observation_2", "observation_3"]
        )
        fig, axs = plot_kde([targets, predictions], nb_hist_per_line=3)

    Args:
        data (List[Union[pd.DataFrame, np.ndarray]]): List of data to plot.
            This data can be a `pd.DataFrame`, where each column will be a feature, and a row is an observation.
            This data can be a `np.ndarray`, where each column will a feature, and a row will be an observation.
            Each `pd.DataFrame` / `np.ndarray` in the list must have the same  shape, with the same columns names.
            The plot of each `pd.DataFrame` / `np.ndarray` will be overlapped.
        nb_hist_per_line (int, optional): Number of histograms to plot per row. Defaults to 2.
        columns (Optional[List[str]], optional): Feature's names.
            If `columns` is `None` and at least one array in `data` is a `pd.DataFrame`, the column names will be inferred from this array.
            If `columns` is not `None`, it must be a list of size the number of features in each array in `data`.
            Defaults to None.
        labels (Optional[List[str]], optional): Labels of each in array in `data\. Defaults to ["targets", "predictions"].

    Returns:
        Tuple[Figure, Axes]: figure and axes containing the plots
    """
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
            axs[i // nb_hist_per_line, i % nb_hist_per_line] = sns.kdeplot(
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

    for row in axs:
        for ax in row:
            if not ax.collections:
                ax.set_visible(False)
            index += 1

    return fig, axs


def plot_correlation(df: pd.DataFrame) -> Tuple[Figure, Axes]:
    """Plots the correlation between all the features in a dataframe.

    Args:
        df (pd.DataFrame): dataframe where the columns are the features and the rows are the observations.

    Returns:
        Tuple[Figure, Axes]: figure and axes containing the plots
    """
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
