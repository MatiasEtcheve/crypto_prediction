import json
from dataclasses import dataclass
from io import StringIO
from itertools import cycle
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualisation.dataset import Dataset


def display_pdf(dataset: Dataset, index: int):
    datapoint = dataset[index]
    sorted_trades = np.sort(
        np.concatenate(
            (
                datapoint.dataframe.trades_pnlplus.dropna().unique(),
                datapoint.dataframe.trades_pnlminus.dropna().unique(),
            )
        )
    )
    return go.Histogram(
        x=sorted_trades,
        name="",
        hovertemplate="<i>Value</i>: %{y}<br><i>Range</i>: %{x}",
        showlegend=False,
    )


def display_equity(dataset: Dataset, index: int, start=0, end=-1):
    datapoint = dataset[index]
    if np.isnan(start) and np.isnan(end):
        start = 0
        end = 0
    if np.isnan(start):
        start = 0
    if np.isnan(end):
        end = len(datapoint.dataframe)
    start = max(start, 0)
    end = min(end, len(datapoint.dataframe))
    return go.Scatter(
        x=datapoint.dataframe.datetime.iloc[start:end],
        y=datapoint.dataframe.broker_value.iloc[start:end],
        name="",
        showlegend=False,
    )


def display_best_trade(dataset: Dataset, index: int):
    pad = 25
    dataframe = dataset[index].dataframe
    index_sell = dataframe.trades_pnlplus.idxmax()
    try:
        index_buy = dataframe.loc[:index_sell].buysell_buy.dropna().index[-1]
    except IndexError:
        index_buy = index_sell + 1
    return display_equity(
        dataset,
        index,
        index_buy - pad,
        index_sell + pad,
    )


def display_worst_trade(dataset: Dataset, index: int):
    pad = 25
    dataframe = dataset[index].dataframe
    index_sell = dataframe.trades_pnlminus.idxmin()
    try:
        index_buy = dataframe.loc[:index_sell].buysell_buy.dropna().index[-1]
    except IndexError:
        index_buy = index_sell + 1
    return display_equity(
        dataset,
        index,
        index_buy - pad,
        index_sell + pad,
    )


def display_average_trade(dataset: Dataset, index: int):
    pad = 25
    mean = dataset[index].analyzers["trade"]["Analysis"]["pnl"]["net"]["average"]
    dataframe = dataset[index].dataframe
    column = "trades_pnlplus" if mean > 0 else "trades_pnlminus"
    index_sell = (dataframe[column].dropna() - mean).abs().argsort().index[0]
    try:
        index_buy = dataframe.loc[:index_sell].buysell_buy.dropna().index[-1]
    except IndexError:
        index_buy = index_sell + 1
    return display_equity(
        dataset,
        index,
        index_buy - pad,
        index_sell + pad,
    )


def display_kpis(dataset: Dataset, index: int):
    datapoint = dataset[index]
    kpi_labels = list(datapoint.kpi.keys())
    kpi_values = [datapoint.kpi[x] for x in kpi_labels]
    for index_label, label in enumerate(kpi_labels):
        sorted_names = [point.name for point in dataset.order[label]]
        kpi_labels[
            index_label
        ] += f"_{len(sorted_names)-sorted_names.index(datapoint.name)}/{len(sorted_names)}"
    return go.Scatterpolar(
        r=kpi_values,
        theta=kpi_labels,
        fill="toself",
        name="",
        hovertemplate="<i>θ</i>: %{theta}<br><i>r</i>: %{r}<br>",
        mode="lines+text",
        showlegend=False,
    )


def display(
    dataset: Dataset,
    display_functions: List[Callable] = [
        display_equity,
        display_pdf,
        display_kpis,
    ],
):
    specs = []
    for func in display_functions:
        if func == display_kpis:
            specs.append({"type": "polar"})
        else:
            specs.append({})

    specs = [list(x) for x in np.reshape(np.array(specs), (1, len(display_functions)))]

    palette = cycle(plotly.colors.qualitative.Dark24)
    figs = {}
    for index, datapoint in enumerate(dataset):
        color = next(palette)

        fig = make_subplots(
            rows=1,
            cols=len(display_functions),
            specs=specs,
            # vertical_spacing=0.3,
        )
        for index_plot, func in enumerate(display_functions):
            trace = func(dataset, index)
            fig.add_trace(trace, row=1, col=index_plot + 1)
        fig.update_layout(
            margin_t=0,
            margin_b=10,
            legend=dict(orientation="h"),
            height=200,
            width=1200,
        )
        fig.update_traces(nbinsx=20, selector=dict(type="histogram"))
        fig.update_traces(marker_color=color)
        fig.update_polars(radialaxis_range=[0, 1])
        fig.update_annotations(
            yshift=35,
            xshift=-70,
            xanchor="left",
            yanchor="top",
        )
        figs[datapoint.name] = fig
    return figs
