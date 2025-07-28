"""Model which uses mutliple prediction heads"""
import torch
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from torch import nn

from pvnet.models.base_model import BaseModel


class Ensemble(BaseModel):
    """Ensemble of PVNet models"""

    def __init__(
        self,
        model_list: list[BaseModel],
        weights: list[float] | None = None,
    ):
        """Ensemble of PVNet models

        Args:
            model_list: A list of PVNet models to ensemble
            weights: A list of weighting to apply to each model. If None, the models are weighted
                equally.
        """

        # Surface check all the models are compatible
        output_quantiles = []
        history_minutes = []
        forecast_minutes = []
        target_key = []
        interval_minutes = []

        # Get some model properties from each model
        for model in model_list:
            output_quantiles.append(model.output_quantiles)
            history_minutes.append(model.history_minutes)
            forecast_minutes.append(model.forecast_minutes)
            target_key.append(model._target_key)
            interval_minutes.append(model.interval_minutes)

        # Check these properties are all the same
        for param_list in [
            output_quantiles,
            history_minutes,
            forecast_minutes,
            target_key,
            interval_minutes,
        ]:
            assert all([p == param_list[0] for p in param_list]), param_list

        super().__init__(
            history_minutes=history_minutes[0],
            forecast_minutes=forecast_minutes[0],
            output_quantiles=output_quantiles[0],
            target_key=target_key[0],
            interval_minutes=interval_minutes[0],
        )

        self.model_list = nn.ModuleList(model_list)

        if weights is None:
            weights = torch.ones(len(model_list)) / len(model_list)
        else:
            assert len(weights) == len(model_list)
            weights = torch.Tensor(weights) / sum(weights)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x: TensorBatch) -> torch.Tensor:
        """Run the model forward"""
        y_hat = 0
        for weight, model in zip(self.weights, self.model_list):
            y_hat = model(x) * weight + y_hat
        return y_hat
