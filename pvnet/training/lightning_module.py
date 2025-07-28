"""Pytorch lightning module for training PVNet models"""
import tempfile

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from ocf_data_sampler.torch_datasets.sample.base import copy_batch_to_device

from pvnet.models.base_model import BaseModel
from pvnet.optimizers import AbstractOptimizer
from pvnet.training.plots import plot_sample_forecasts
from pvnet.training.stores import BatchAccumulator, MetricAccumulator, PredAccumulator


class PVNetLightningModule(pl.LightningModule):
    """Lightning module for training PVNet models"""

    def __init__(
        self,
        model: BaseModel,
        optimizer: AbstractOptimizer,
        save_validation_results_csv: bool = False,
    ):
        """Lightning module for training PVNet models

        Args:
            model: The PVNet model
            optimizer: Optimizer
            save_validation_results_csv: whether to save full CSV outputs from validation results
        """
        super().__init__()

        self.model = model
        self._optimizer = optimizer

        # Model must have lr to allow tuning
        # This setting is only used when lr is tuned with callback
        self.lr = None

        self._accumulated_metrics = MetricAccumulator()
        self._accumulated_batches = BatchAccumulator(key_to_keep=model._target_key)
        self._accumulated_y_hat = PredAccumulator()
        self._horizon_maes = MetricAccumulator()

        # Set up store for all all validation results so we can log these
        self.validation_epoch_results = []
        self.save_validation_results_csv = save_validation_results_csv

    def transfer_batch_to_device(
        self, 
        batch: TensorBatch, 
        device: torch.device, 
        dataloader_idx: int,
    ) -> dict:
        """Method to move custom batches to a given device"""
        return copy_batch_to_device(batch, device)

    def _calculate_quantile_loss(self, y_quantiles: list[float], y: torch.Tensor) -> torch.Tensor:
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

        losses.update({
                "MSE":  F.mse_loss(y_hat, y),
                "MAE": F.l1_loss(y_hat, y),
            }
        )

        return losses

    def _step_mae_and_mse_metrics(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor, 
        dict_key_root: str,
    ) -> dict[str, torch.Tensor]:
        """Calculate the MSE and MAE at each forecast step"""
        losses = {}

        mse_each_step = torch.mean((y_hat - y) ** 2, dim=0).cpu().numpy()
        mae_each_step = torch.mean(torch.abs(y_hat - y), dim=0).cpu().numpy()

        losses.update({f"MSE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mse_each_step)})
        losses.update({f"MAE_{dict_key_root}/step_{i:03}": m for i, m in enumerate(mae_each_step)})

        return losses

    def _calculate_val_losses(
        self, 
        y: torch.Tensor, 
        y_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate additional validation losses"""

        losses = {}

        if self.model.use_quantile_regression:
            # Add fraction below each quantile for calibration
            for i, quantile in enumerate(self.model.output_quantiles):
                below_quant = y <= y_hat[..., i]
                # Mask values small values, which are dominated by night
                mask = y >= 0.01
                losses[f"fraction_below_{quantile}_quantile"] = below_quant[mask].float().mean()

            # Take median value for remaining metric calculations
            y_hat = self.model._quantiles_to_prediction(y_hat)

        # Log the loss at each time horizon
        losses.update(self._step_mae_and_mse_metrics(y, y_hat, dict_key_root="horizon"))

        # TODO: We don't need this each epoch. Doing this once is fine
        # Log the persistance losses
        y_persist = y[:, -1].unsqueeze(1).expand(-1, self.model.forecast_len)
        losses["MAE_persistence/val"] = F.l1_loss(y_persist, y)
        losses["MSE_persistence/val"] = F.mse_loss(y_persist, y)

        # Log persistance loss at each time horizon
        losses.update(self._step_mae_and_mse_metrics(y, y_persist, dict_key_root="persistence"))
        return losses

    def _training_accumulate_log(
        self, 
        batch: TensorBatch, 
        losses: dict[str, torch.Tensor], 
        y_hat: torch.Tensor,
    ) -> None:
        """Internal function to accumulate training batches and log results.

        This is used when accummulating grad batches. Should make the variability in logged training
        step metrics indpendent on whether we accumulate N batches of size B or just use a larger
        batch size of N*B with no accumulaion.
        """

        losses = {k: v.detach().cpu() for k, v in losses.items()}
        y_hat = y_hat.detach().cpu()

        self._accumulated_metrics.append(losses)
        self._accumulated_batches.append(batch)
        self._accumulated_y_hat.append(y_hat)

        if not self.trainer.fit_loop._should_accumulate():
            losses = self._accumulated_metrics.flush()
            batch = self._accumulated_batches.flush()
            y_hat = self._accumulated_y_hat.flush()

            self.log_dict(losses, on_step=True, on_epoch=True)


    def training_step(self, batch: TensorBatch, batch_idx: int) -> torch.Tensor:
        """Run training step"""
        y_hat = self.model(batch)

        # Batch is adapted in the model forward method, but needs to be adapted here too
        batch = self.model._adapt_batch(batch)

        y = batch[self.model._target_key][:, -self.model.forecast_len :]

        losses = self._calculate_common_losses(y, y_hat)
        losses = {f"{k}/train": v for k, v in losses.items()}

        self._training_accumulate_log(batch, losses, y_hat)

        if self.model.use_quantile_regression:
            opt_target = losses["quantile_loss/train"]
        else:
            opt_target = losses["MAE/train"]
        return opt_target

    def _log_forecast_plot(
        self, 
        batch: TensorBatch,
        y_hat: torch.Tensor, 
        accum_batch_num: int,
    ) -> None:
        """Log forecast plot to wandb"""
        fig = plot_sample_forecasts(
            batch,
            y_hat,
            quantiles=self.model.output_quantiles,
            key_to_plot=self.model._target_key,
        )

        plot_name = f"val_forecast_samples/batch_idx_{accum_batch_num}"

        self.logger.experiment.log({plot_name: wandb.Image(fig)})

        plt.close(fig)

    def _log_validation_results(
        self, 
        batch: TensorBatch, 
        y_hat: torch.Tensor, 
        accum_batch_num: int,
    ) -> None:
        """Append validation results to self.validation_epoch_results"""

        # Get truth values - shape: (batch_size, history_len + forecast_len)
        y = batch[self.model._target_key][:, -self.model.forecast_len :].detach().cpu().numpy()
        batch_size = y.shape[0]

        # Get predictions - shape: (batch_size, forecast_len, (quantiles))
        y_hat = y_hat.detach().cpu().numpy()

        # Get time_utc - shape: (batch_size, history_len + forecast_len)
        time_utc_key = f"{self.model._target_key}_time_utc"
        time_utc = batch[time_utc_key][:, -self.model.forecast_len :].detach().cpu().numpy()

        # Get target ID and squueze from shape (batch_size, 1) to (batch_size,)
        target_id = batch[f"{self.model._target_key}_id"].detach().cpu().numpy().squeeze()

        for i in range(batch_size):
            y_i = y[i]
            y_hat_i = y_hat[i]
            time_utc_i = time_utc[i]
            target_id_i = target_id[i]

            results_dict = {
                "y": y_i,
                "time_utc": time_utc_i,
            }
            if self.model.use_quantile_regression:
                quantile_results  = {}
                for i, q in enumerate(self.model.output_quantiles):
                    quantile_results[f"y_quantile_{q}"] = y_hat_i[:, i]
                results_dict.update(quantile_results)
            else:
                results_dict["y_hat"] = y_hat_i

            results_df = pd.DataFrame(results_dict)
            results_df["id"] = target_id_i
            results_df["batch_idx"] = accum_batch_num
            results_df["example_idx"] = i

            self.validation_epoch_results.append(results_df)

    def validation_step(self, batch: TensorBatch, batch_idx: int) -> dict[str, torch.Tensor]:
        """Run validation step"""

        accum_batch_num = batch_idx // self.trainer.accumulate_grad_batches

        y_hat = self.model(batch)
        # Batch is adapted in the model forward method, but needs to be adapted here too
        batch = self.model._adapt_batch(batch)

        y = batch[self.model._target_key][:, -self.model.forecast_len :]

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            self._log_validation_results(batch, y_hat, accum_batch_num)

        losses = self._calculate_common_losses(y, y_hat)
        losses.update(self._calculate_val_losses(y, y_hat))

        # Store these to make horizon accuracy plot
        self._horizon_maes.append(
            {i: losses[f"MAE_horizon/step_{i:03}"] for i in range(self.model.forecast_len)}
        )

        logged_losses = {f"{k}/val": v for k, v in losses.items()}

        self.log_dict(logged_losses, on_step=False, on_epoch=True)

        if isinstance(self.logger, pl.loggers.WandbLogger):
            if accum_batch_num in [0, 1]:
                # Store these temporarily under self
                if not hasattr(self, "_val_y_hats"):
                    self._val_y_hats = PredAccumulator()
                    self._val_batches = BatchAccumulator(key_to_keep=self.model._target_key)

                self._val_y_hats.append(y_hat)
                self._val_batches.append(batch)

                # If batch has accumulated plot it
                if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                    y_hat = self._val_y_hats.flush()
                    batch = self._val_batches.flush()

                    self._log_forecast_plot(batch, y_hat, accum_batch_num)

                    del self._val_y_hats
                    del self._val_batches

        return logged_losses

    def on_validation_epoch_end(self) -> None:
        """Run on epoch end"""

        validation_results_df = pd.concat(self.validation_epoch_results)
        self.validation_epoch_results = []

        horizon_maes_dict = self._horizon_maes.flush()

        if isinstance(self.logger, pl.loggers.WandbLogger):
            # Log error distribution metrics
            val_error = validation_results_df["y"] - validation_results_df["y_quantile_0.5"]

            extreme_error_metrics = {
                "2nd_percentile_median_forecast_error": val_error.quantile(0.02),
                "5th_percentile_median_forecast_error": val_error.quantile(0.05),
                "95th_percentile_median_forecast_error": val_error.quantile(0.95),
                "98th_percentile_median_forecast_error": val_error.quantile(0.98),
                "95th_percentile_median_forecast_absolute_error": val_error.abs().quantile(0.95),
                "98th_percentile_median_forecast_absolute_error": val_error.abs().quantile(0.98),
            }

            self.log_dict(extreme_error_metrics, on_step=False, on_epoch=True)

            # Save all validation results
            if self.save_validation_results_csv:
                with tempfile.TemporaryDirectory() as tempdir:
                    filename = f"validation_results_{self.current_epoch}.csv"
                    validation_results_df.to_csv(tempdir / filename, index=False)

                    validation_artifact = wandb.Artifact(filename, type="dataset")
                    validation_artifact.add_file(filename)

                    wandb.log_artifact(validation_artifact)

            # Create the horizon accuracy curve
            per_step_losses = [[i, horizon_maes_dict[i]] for i in range(self.model.forecast_len)]
            table = wandb.Table(data=per_step_losses, columns=["horizon_step", "MAE"])
            plot = wandb.plot.line(table, "horizon_step", "MAE", title="Horizon loss curve")
            wandb.log({"horizon_loss_curve": plot})

    def configure_optimizers(self):
        """Configure the optimizers using learning rate found with LR finder if used"""
        if self.lr is not None:
            # Use learning rate found by learning rate finder callback
            self._optimizer.lr = self.lr
        return self._optimizer(self.model)
