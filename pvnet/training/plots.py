"""Plots logged during training"""
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import pylab
import torch
import wandb
from ocf_data_sampler.numpy_sample.common_types import TensorBatch


def wandb_line_plot(
    x: Sequence[float], 
    y: Sequence[float], 
    xlabel: str, 
    ylabel: str, 
    title: str | None = None,
    add_identity_line: bool = False,
) -> wandb.plot.CustomChart:
    """Make a wandb line plot"""
    # Main series data
    data = [[xi, yi, "Data"] for xi, yi in zip(x, y)]
    
    # Add identity line endpoints if requested
    if add_identity_line:
        min_val, max_val = min(x), max(x)
        data.append([min_val, min_val, "x=y"])
        data.append([max_val, max_val, "x=y"])

    table = wandb.Table(data=data, columns=[xlabel, ylabel, "Series"])

    # stroke=None creates a clean single line; stroke="Series" creates multi-line legend
    stroke_col = "Series" if add_identity_line else None

    return wandb.plot.line(
        table=table, 
        x=xlabel, 
        y=ylabel, 
        stroke=stroke_col, 
        title=title
    )

def wandb_line_plot_custom(
    x: Sequence[float], 
    y: Sequence[float], 
    xlabel: str, 
    ylabel: str, 
    title: str | None = None):
    return wandb_line_plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, 
                           title=title, add_identity_line=True)
   

def plot_sample_forecasts(
    batch: TensorBatch,
    y_hat: torch.Tensor,
    quantiles: list[float] | None,
    key_to_plot: str,
) -> plt.Figure:
    """Plot a batch of data and the forecast from that batch"""

    y = batch[key_to_plot].cpu().numpy()
    y_hat = y_hat.cpu().numpy()
    forecast_length = y_hat.shape[1]
    ids = batch["location_id"].cpu().numpy().squeeze()
    times_utc = pd.to_datetime(
        batch["time_utc"].cpu().numpy().squeeze().astype("datetime64[ns]")
    )
    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()[:batch_size]):

        # Crop to the forecast window only
        forecast_times = times_utc[i][-forecast_length:]
        y_no_history = y[i][-forecast_length:]

        ax.plot(forecast_times, y_no_history, marker=".", color="k", label=r"$y$")

        if quantiles is None:
            ax.plot(
                forecast_times,
                y_hat[i],
                marker=".",
                color="r",
                label=r"$\hat{y}$",
            )
        else:
            cm = pylab.get_cmap("twilight")
            for nq, q in enumerate(quantiles):
                ax.plot(
                    forecast_times,
                    y_hat[i, :, nq],
                    color=cm(q),
                    label=r"$\hat{y}$" + f"({q})",
                    alpha=0.7,
                )

        ax.set_title(f"ID: {ids[i]} | {forecast_times[0].date()}", fontsize="small")

        xticks = [t for t in forecast_times if t.minute == 0][::2]
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
