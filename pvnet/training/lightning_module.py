"""Pytorch lightning module for training PVNet models"""

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
import xarray as xr
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from ocf_data_sampler.torch_datasets.sample.base import copy_batch_to_device
from omegaconf import DictConfig

from pvnet.data.base_datamodule import collate_fn
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
from pvnet.training.plots import plot_sample_forecasts, wandb_line_plot
from pvnet.utils import validate_batch_against_config


class PVNetLightningModule(pl.LightningModule):
    """Lightning module for training PVNet models"""

    def __init__(
            self,
            model: BaseModel,
            optimizer: AbstractOptimizer,
            model_config: DictConfig,
            save_all_validation_results: bool = False,
        ):
            """Lightning module for training PVNet models

            Args:
                model: The PVNet model
                optimizer: Optimizer
                model_config: The model configuration.
                save_all_validation_results: Whether to save all the validation predictions to wandb
            """
            super().__init__()

            # This single line saves all constructor arguments to self.hparams
            # Ignore model and optimizer as they are complex objects
            self.save_hyperparameters(ignore=["model", "optimizer"])
            self.model = model
            self._optimizer = optimizer
            self.lr = None

    def transfer_batch_to_device(
        self, 
        batch: TensorBatch, 
        device: torch.device, 
        dataloader_idx: int,
    ) -> dict:
        """Method to move custom batches to a given device"""
        return copy_batch_to_device(batch, device)

    def _calculate_quantile_loss(self, y_quantiles: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate quantile loss.

        Note:
            Implementation copied from:
                https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting
                /metrics/quantile.html#QuantileLoss.loss

        Args:
            y_quantiles: Quantile prediction of network
            y: Target values

        Returns:
            Quantile loss
        """
        losses = []
        for i, q in enumerate(self.model.output_quantiles):
            errors = y - y_quantiles[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        losses = 2 * torch.cat(losses, dim=2)

        return losses.mean()
    
    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self.model)

    def _calculate_common_losses(
        self, 
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses common to train, and val"""

        losses = {}

        if self.model.use_quantile_regression:
            losses["quantile_loss"] = self._calculate_quantile_loss(y_hat, y)
            y_hat = self.model._quantiles_to_prediction(y_hat)

        losses.update({"MSE":  F.mse_loss(y_hat, y), "MAE": F.l1_loss(y_hat, y)})

        return losses
    
    def training_step(self, batch: TensorBatch, batch_idx: int) -> torch.Tensor:
        """Run training step"""
        y_hat = self.model(batch)

        # Batch is adapted in the model forward method, but needs to be adapted here too
        batch = self.model._adapt_batch(batch)

        y = batch[self.model._target_key][:, -self.model.forecast_len :]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self.log_dict(losses, on_step=True, on_epoch=True)

        if self.model.use_quantile_regression:
            opt_target = losses["quantile_loss/train"]
        else:
            opt_target = losses["MAE/train"]
        return opt_target
    
    def _calculate_val_losses(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate additional losses only run in validation"""

        losses = {}

        if self.model.use_quantile_regression:
            metric_name = "val_fraction_below/fraction_below_{:.2f}_quantile"
            # Add fraction below each quantile for calibration
            for i, quantile in enumerate(self.model.output_quantiles):
                below_quant = y <= y_hat[..., i]
                # Mask values small values, which are dominated by night
                mask = y >= 0.01
                losses[metric_name.format(quantile)] = below_quant[mask].float().mean()

        return losses

    def _calculate_step_metrics(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor, 
    ) -> tuple[np.array, np.array]:
        """Calculate the MAE and MSE at each forecast step"""

        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0).cpu().numpy()
        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0).cpu().numpy()
       
        return mae_each_step, mse_each_step
    
    def _store_val_predictions(self, batch: TensorBatch, y_hat: torch.Tensor) -> None:
        """Internally store the validation predictions"""
        
        taregt_key = self.model._target_key

        y = batch[taregt_key][:, -self.model.forecast_len :].cpu().numpy()
        y_hat = y_hat.cpu().numpy() 
        ids = batch[f"{taregt_key}_id"].cpu().numpy()
        init_times_utc = pd.to_datetime(
            batch[f"{taregt_key}_time_utc"][:, self.model.history_len+1]
            .cpu().numpy().astype("datetime64[ns]")
        )

        if self.model.use_quantile_regression:
            p_levels = self.model.output_quantiles
        else:
            p_levels = [0.5]
            y_hat = y_hat[..., None]

        ds_preds_batch = xr.Dataset(
            data_vars=dict(
                y_hat=(["sample_num", "forecast_step",  "p_level"], y_hat),
                y=(["sample_num", "forecast_step"], y),
            ),
            coords=dict(
                ids=("sample_num", ids),
                init_times_utc=("sample_num", init_times_utc),
                p_level=p_levels,
            ),
        )
        self.all_val_results.append(ds_preds_batch)

    def on_validation_epoch_start(self):
        """Run at start of val period"""
        # Set up stores which we will fill during validation
        self.all_val_results: list[xr.Dataset] = []
        self._val_horizon_maes: list[np.array] = []
        if self.current_epoch==0:
            self._val_persistence_horizon_maes: list[np.array] = []
        
        # Plot some sample forecasts
        val_dataset = self.trainer.val_dataloaders.dataset

        plots_per_figure = 16
        num_figures = 2

        for plot_num in range(num_figures):
            idxs = np.arange(plots_per_figure) + plot_num * plots_per_figure
            idxs = idxs[idxs<len(val_dataset)]

            if len(idxs)==0:
                continue

            batch = collate_fn([val_dataset[i] for i in idxs])
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)

            if self.current_epoch == 0 and plot_num == 0:
                validate_batch_against_config(
                    batch=batch, 
                    model_config=self.hparams.model_config
                )

            with torch.no_grad():
                y_hat = self.model(batch)
            
            # Batch is adapted in the model forward method, but needs to be adapted here too
            batch = self.model._adapt_batch(batch)
            
            fig = plot_sample_forecasts(
                batch,
                y_hat,
                quantiles=self.model.output_quantiles,
                key_to_plot=self.model._target_key,
            )

            plot_name = f"val_forecast_samples/sample_set_{plot_num}"

            self.logger.experiment.log({plot_name: wandb.Image(fig)})

            plt.close(fig)

    def validation_step(self, batch: TensorBatch, batch_idx: int) -> None:
        """Run validation step"""

        y_hat = self.model(batch)
        # Batch is adapted in the model forward method, but needs to be adapted here too
        batch = self.model._adapt_batch(batch)

        # Internally store the val predictions
        self._store_val_predictions(batch, y_hat)

        y = batch[self.model._target_key][:, -self.model.forecast_len :]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/val": v for k, v in losses.items()}

        losses.update(self._calculate_val_losses(y, y_hat))

        # Calculate the horizon MAE/MSE metrics
        if self.model.use_quantile_regression:
            y_hat_mid = self.model._quantiles_to_prediction(y_hat)
        else:
            y_hat_mid = y_hat

        mae_step, mse_step = self._calculate_step_metrics(y, y_hat_mid)

        # Store to make horizon-MAE plot
        self._val_horizon_maes.append(mae_step)

        # Also add each step to logged metrics
        losses.update({f"val_step_MAE/step_{i:03}": m for i, m in enumerate(mae_step)})
        losses.update({f"val_step_MSE/step_{i:03}": m for i, m in enumerate(mse_step)})

        # Calculate the persistance losses - we only need to do this once per training run
        # not every epoch
        if self.current_epoch==0:
            y_persist = (
                batch[self.model._target_key][:, -(self.model.forecast_len+1)]
                .unsqueeze(1).expand(-1, self.model.forecast_len)
            )
            mae_step_persist, mse_step_persist = self._calculate_step_metrics(y, y_persist)
            self._val_persistence_horizon_maes.append(mae_step_persist)
            losses.update(
                {
                    "MAE/val_persistence": mae_step_persist.mean(), 
                    "MSE/val_persistence": mse_step_persist.mean()
                }
            )

        # Log the metrics
        self.log_dict(losses, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Run on epoch end"""

        ds_val_results = xr.concat(self.all_val_results, dim="sample_num")
        self.all_val_results = []

        val_horizon_maes = np.mean(self._val_horizon_maes, axis=0)
        self._val_horizon_maes = []

        # We only run this on the first epoch
        if self.current_epoch==0:
            val_persistence_horizon_maes = np.mean(self._val_persistence_horizon_maes, axis=0)
            self._val_persistence_horizon_maes = []

        if isinstance(self.logger, pl.loggers.WandbLogger):
            # Calculate and log extreme error metrics
            val_error = ds_val_results["y"] - ds_val_results["y_hat"].sel(p_level=0.5)

            # Factor out this part of the string for brevity below
            s = "error_extremes/{}_percentile_median_forecast_error"
            s_abs = "error_extremes/{}_percentile_median_forecast_absolute_error"

            extreme_error_metrics = {
                s.format("2nd"): val_error.quantile(0.02).item(),
                s.format("5th"): val_error.quantile(0.05).item(),
                s.format("95th"): val_error.quantile(0.95).item(),
                s.format("98th"): val_error.quantile(0.98).item(),
                s_abs.format("95th"): np.abs(val_error).quantile(0.95).item(),
                s_abs.format("98th"): np.abs(val_error).quantile(0.98).item(),
            }

            self.log_dict(extreme_error_metrics, on_step=False, on_epoch=True)

            # Optionally save all validation results - these are overridden each epoch
            if self.hparams.save_all_validation_results:
                # Add attributes
                ds_val_results.attrs["epoch"] = self.current_epoch

                # Save locally to the wandb output dir
                wandb_log_dir = self.logger.experiment.dir
                filepath = f"{wandb_log_dir}/validation_results.netcdf"
                ds_val_results.to_netcdf(filepath)
                
                # Uplodad to wandb
                self.logger.experiment.save(filepath, base_path=wandb_log_dir, policy="now")
            
            # Create the horizon accuracy curve
            horizon_mae_plot = wandb_line_plot(
                x=np.arange(self.model.forecast_len), 
                y=val_horizon_maes,
                xlabel="Horizon step",
                ylabel="MAE",
                title="Val horizon loss curve",
            )
            
            wandb.log({"val_horizon_mae_plot": horizon_mae_plot})

            # Create persistence horizon accuracy curve but only on first epoch
            if self.current_epoch==0:
                persist_horizon_mae_plot = wandb_line_plot(
                    x=np.arange(self.model.forecast_len), 
                    y=val_persistence_horizon_maes,
                    xlabel="Horizon step",
                    ylabel="MAE",
                    title="Val persistence horizon loss curve",
                )
                wandb.log({"persistence_val_horizon_mae_plot": persist_horizon_mae_plot})
