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
from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import copy_batch_to_device
from torch.distributions import Normal
from torchmetrics.functional.regression import continuous_ranked_probability_score as crps_fn

from pvnet.datamodule import collate_fn
from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
from pvnet.training.plots import plot_sample_forecasts, wandb_line_plot


class PVNetLightningModule(pl.LightningModule):
    """Lightning module for training PVNet models"""

    def __init__(
        self,
        model: BaseModel,
        optimizer: AbstractOptimizer,
        save_all_validation_results: bool = False,
    ):
        """Lightning module for training PVNet models

        Args:
            model: The PVNet model
            optimizer: Optimizer
            save_all_validation_results: Whether to save all the validation predictions to wandb
        """
        super().__init__()

        self.model = model
        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        # Set up store for all all validation results so we can log these
        self.save_all_validation_results = save_all_validation_results

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

    def _calculate_nll(self, y_gmm, y_true):
        """
        Negative log-likelihood of y_true under the predicted GMM.

        Args:
            y_gmm:   (batch, forecast_len * num_components * 3)
            y_true:  (batch, forecast_len)
        """
        mus, sigmas, pis = self.model._parse_gmm_params(y_gmm)
        # expand y_true to [batch, forecast_len, num_components]
        y_exp = y_true.unsqueeze(-1).expand_as(mus)
        # compute component log-probs
        comp = Normal(mus, sigmas)
        log_p = comp.log_prob(y_exp)  # [batch, forecast_len, num_components]
        # weight them
        weighted = log_p + torch.log(pis + 1e-12)
        # log-sum-exp over components
        log_probs = torch.logsumexp(weighted, dim=-1)  # [batch, forecast_len]
        # negative log-likelihood
        nll = -log_probs.mean()  # mean over batch & horizon
        return nll
    
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
        elif self.model.use_gmm:
            losses["nll"] = self._calculate_nll(y_hat, y)
            y_hat = self.model._gmm_to_prediction(y_hat)
        
        losses.update({"MSE": F.mse_loss(y_hat, y), "MAE": F.l1_loss(y_hat, y)})

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
        elif self.model.use_gmm:
            opt_target = losses["nll/train"]
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

            b, h, q = y_hat.shape
            # crps_fn expects preds with last dim = ensemble members
            losses["CRPS"] = crps_fn(preds=y_hat.reshape(b * h, q), target=y.reshape(-1))

        if self.model.use_gmm:
            # Convert GMM into samples or quantiles
            mus, sigmas, pis = self.model._parse_gmm_params(y_hat)  # shape: [B, H, C]

            # Sample from GMM to get an ensemble of predictions
            num_samples = 20
            samples = self.model._sample_from_gmm(
                mus, sigmas, pis, n_samples=num_samples
            )
            # samples: [num_samples, batch, forecast_len]

            # reshape for TorchMetrics: [batch * forecast_len, ensemble_members]
            ensemble = samples.permute(1, 2, 0).reshape(-1, num_samples)  # [B*H, N]
            targets = y.reshape(-1)  # [B*H]

            losses["CRPS"] = crps_fn(preds=ensemble, target=targets)

            # Calculate the GMM loss
            losses["nll/val"] = self._calculate_nll(y_hat, y)
            # Collapse to mixture mean for further metrics
            y_hat = self.model._gmm_to_prediction(y_hat)
        
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
        
        target_key = self.model._target_key

        y = batch[target_key][:, -self.model.forecast_len :].cpu()
        ids = batch[f"{target_key}_id"].cpu().numpy()
        init_times_utc = pd.to_datetime(
            batch[f"{target_key}_time_utc"][:, self.model.history_len+1]
            .cpu().numpy().astype("datetime64[ns]")
        )

        data_vars = {
            "y": (["sample_num", "forecast_step"], y.numpy()),
        }
        coords = {
            "ids": ("sample_num", ids),
            "init_times_utc": ("sample_num", init_times_utc),
        }

        if self.model.use_gmm:
            # Parse GMM parameters from the raw output
            mus, sigmas, pis = self.model._parse_gmm_params(y_hat)

            # Move tensors to CPU and convert to numpy for storage
            mus = mus.cpu().numpy()
            sigmas = sigmas.cpu().numpy()
            pis = pis.cpu().numpy()

            # Store parameters for each component
            for i in range(self.model.num_gmm_components):
                data_vars[f"gmm_mean_{i}"] = (
                    ["sample_num", "forecast_step"],
                    mus[:, :, i],
                )
                data_vars[f"gmm_std_{i}"] = (
                    ["sample_num", "forecast_step"],
                    sigmas[:, :, i],
                )
                data_vars[f"gmm_weight_{i}"] = (
                    ["sample_num", "forecast_step"],
                    pis[:, :, i],
                )

            # Also store the point prediction (mixture mean)
            y_pred = (pis * mus).sum(axis=-1)
            data_vars["y_pred"] = (["sample_num", "forecast_step"], y_pred)

        elif self.model.use_quantile_regression:
            y_hat = y_hat.cpu().numpy()
            p_levels = self.model.output_quantiles
            data_vars["y_hat"] = (["sample_num", "forecast_step", "p_level"], y_hat)
            coords["p_level"] = p_levels
            
        else:
            y_hat = y_hat.cpu().numpy()
            p_levels = [0.5]

            data_vars["y_hat"] = (
                ["sample_num", "forecast_step", "p_level"],
                y_hat[..., None],
            )
            coords["p_level"] = p_levels

        ds_preds_batch = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
        )
        self.all_val_results.append(ds_preds_batch)

    def on_validation_epoch_start(self):
        """Run at start of val period"""
        # Set up stores which we will fill during validation
        self.all_val_results: list[xr.Dataset] = []
        self._val_horizon_maes: list[np.array] = []
        if self.current_epoch==0:
            self._val_persistence_horizon_maes: list[np.array] = []

        if self.logger is None:
            return        
        
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
            
            with torch.no_grad():
                y_hat = self.model(batch)

            batch = self.model._adapt_batch(batch)

            fig = plot_sample_forecasts(
                batch,
                y_hat,
                model=self.model,
                key_to_plot=self.model._target_key,
            )

            plot_name = f"val_forecast_samples/sample_set_{plot_num}"

            self.logger.experiment.log({plot_name: wandb.Image(fig)})

            plt.close(fig)

    def validation_step(self, batch: TensorBatch, batch_idx: int) -> None:
        """Run validation step"""

        y_hat = self.model(batch)
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
        elif self.model.use_gmm:
            y_hat_mid = self.model._gmm_to_prediction(y_hat)
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
                    "MSE/val_persistence": mse_step_persist.mean(),
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

            # Determine the point prediction based on the model type
            if self.model.use_gmm:
                # For GMM, the point prediction 'y_pred' (mixture mean) is already calculated
                point_prediction = ds_val_results["y_pred"]
            else:
                # For Quantiles or simple forecasts, use the median (p_level=0.5)
                point_prediction = ds_val_results["y_hat"].sel(p_level=0.5)

            # Calculate the error based on the correct point prediction
            val_error = ds_val_results["y"] - point_prediction            

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
            if self.save_all_validation_results:
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
