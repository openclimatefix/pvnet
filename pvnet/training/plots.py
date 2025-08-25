"""Plots logged during training"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
import wandb
from ocf_data_sampler.numpy_sample.common_types import TensorBatch


def wandb_line_plot(
    x: Sequence[float],
    y: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> wandb.plot.CustomChart:
    """Make a wandb line plot"""
    data = [[xi, yi] for (xi, yi) in zip(x, y)]
    table = wandb.Table(data=data, columns=[xlabel, ylabel])
    return wandb.plot.line(table, xlabel, ylabel, title=title)


def plot_sample_forecasts(
    batch: TensorBatch,
    y_hat: torch.Tensor,
    model,
    key_to_plot: str,
) -> plt.Figure:
    """Plot a batch of data and the forecast from that batch"""

    y = batch[key_to_plot].cpu()
    y_hat = y_hat.cpu()
    ids = batch[f"{key_to_plot}_id"].cpu().numpy().squeeze()
    times_utc = pd.to_datetime(
        batch[f"{key_to_plot}_time_utc"]
        .cpu()
        .numpy()
        .squeeze()
        .astype("datetime64[ns]")
    )
    batch_size = y.shape[0]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i, ax in enumerate(axes.ravel()[:batch_size]):
        # Get the forecast-only part of the ground truth and time
        y_true_forecast = y[i, -model.forecast_len :]
        times_forecast = times_utc[i, -model.forecast_len :]

        # Plot ground truth
        ax.plot(
            times_forecast,
            y_true_forecast,
            marker=".",
            color="k",
            label=r"True Value ($y$)",
        )

        if model.use_gmm:
            mus, sigmas, pis = model._parse_gmm_params(y_hat[i : i + 1])
            mus, sigmas, pis = mus.squeeze(0), sigmas.squeeze(0), pis.squeeze(0)

            mixture_mean = torch.sum(pis * mus, dim=-1)
            mixture_variance = torch.sum(
                pis * (mus.pow(2) + sigmas.pow(2)), dim=-1
            ) - mixture_mean.pow(2)
            mixture_std = torch.sqrt(mixture_variance.clamp(min=1e-6))

            ax.plot(
                times_forecast,
                mixture_mean.numpy(),
                marker=".",
                color="red",
                label=r"Mixture Mean",
            )

            lower_bound = mixture_mean - 1.645 * mixture_std
            upper_bound = mixture_mean + 1.645 * mixture_std
            ax.fill_between(
                times_forecast,
                lower_bound.numpy(),
                upper_bound.numpy(),
                color="red",
                alpha=0.2,
                label="90% Confidence",
            )

        elif model.use_quantile_regression:
            y_hat_i = y_hat[i]
            quantiles = model.output_quantiles
            median_idx = quantiles.index(0.5)

            ax.plot(
                times_forecast,
                y_hat_i[:, median_idx],
                marker=".",
                color="blue",
                label=r"Median",
            )

            num_quantiles = len(quantiles)
            for j in range(num_quantiles // 2):
                l_q, u_q = quantiles[j], quantiles[num_quantiles - 1 - j]
                ax.fill_between(
                    times_forecast,
                    y_hat_i[:, j],
                    y_hat_i[:, num_quantiles - 1 - j],
                    alpha=0.2,
                    color="blue",
                    label=f"{l_q*100:.0f}-{u_q*100:.0f}%",
                )
        else:
            ax.plot(
                times_forecast,
                y_hat[i],
                marker=".",
                color="green",
                label=r"Point Forecast",
            )

        ax.set_title(f"ID: {ids[i]} | {times_utc[i][0].date()}", fontsize="small")
        xticks = [t for t in pd.to_datetime(times_forecast) if t.minute == 0][::2]
        ax.set_xticks(
            ticks=xticks, labels=[f"{t.hour:02}" for t in xticks], rotation=90
        )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4)

    if batch_size < 16:
        for ax in axes.ravel()[batch_size:]:
            ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (hour of day)")

    title = f"Normalized {key_to_plot.upper()} Power"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
