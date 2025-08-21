"""Plots logged during training"""
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import pylab
import torch
from ocf_data_sampler.numpy_sample.common_types import TensorBatch

import wandb


def wandb_line_plot(
    x: Sequence[float], 
    y: Sequence[float], 
    xlabel: str, 
    ylabel: str, 
    title: str | None = None
) -> wandb.plot.CustomChart:
    """Make a wandb line plot"""
    data = [[xi, yi] for (xi, yi) in zip(x, y)]
    table = wandb.Table(data=data, columns=[xlabel, ylabel])
    return wandb.plot.line(table, xlabel, ylabel, title=title)


def plot_sample_forecasts(
    batch: TensorBatch,
    y_hat: torch.Tensor,
    quantiles: list[float] | None,
    key_to_plot: str,
) -> plt.Figure:
    """Plot a batch of data and the forecast from that batch"""

    y = batch[key_to_plot].cpu().numpy()
    y_hat = y_hat.cpu().numpy()    
    ids = batch[f"{key_to_plot}_id"].cpu().numpy().squeeze()
    times_utc = pd.to_datetime(
        batch[f"{key_to_plot}_time_utc"].cpu().numpy().squeeze().astype("datetime64[ns]")
    )
    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()[:batch_size]):

        ax.plot(times_utc[i], y[i], marker=".", color="k", label=r"$y$")

        if quantiles is None:
            ax.plot(
                times_utc[i][-len(y_hat[i]) :],
                y_hat[i], 
                marker=".", 
                color="r", 
                label=r"$\hat{y}$",
            )
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    times_utc[i][-len(y_hat[i]) :],
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"ID: {ids[i]} | {times_utc[i][0].date()}", fontsize="small")

        xticks = [t for t in times_utc[i] if t.minute == 0][::2]
        ax.set_xticks(ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90)
        ax.grid()

    axes[0, 0].legend(loc="best")

    if batch_size<16:
        for ax in axes.ravel()[batch_size:]:
            ax.axis("off")
    
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    title =  f"Normed {key_to_plot.upper()} output"
    
    plt.suptitle(title)
    plt.tight_layout()

    return fig
