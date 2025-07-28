"""Utility functions"""
from typing import Any

import numpy as np
import torch


class PredAccumulator:
    """A class for accumulating y-predictions using grad accumulation and small batch size.

    Attributes:
        _y_hats (list[torch.Tensor]): List of prediction tensors
    """

    def __init__(self):
        """Prediction accumulator"""
        self._y_hats = []

    def __bool__(self):
        return len(self._y_hats) > 0

    def append(self, y_hat: torch.Tensor):
        """Append a sub-batch of predictions"""
        self._y_hats.append(y_hat)

    def flush(self) -> torch.Tensor:
        """Return all appended predictions as single tensor and remove from accumulated store."""
        y_hat = torch.cat(self._y_hats, dim=0)
        self._y_hats = []
        return y_hat


class DictListAccumulator:
    """Abstract class for accumulating dictionaries of lists"""

    @staticmethod
    def _dict_list_append(d1: dict[str, Any], d2: dict[str, Any]) -> None:
        for k, v in d2.items():
            d1[k].append(v)

    @staticmethod
    def _dict_init_list(d) -> None:
        return {k: [v] for k, v in d.items()}


class MetricAccumulator(DictListAccumulator):
    """Dictionary of metrics accumulator.

    A class for accumulating, and finding the mean of logging metrics when using grad
    accumulation and the batch size is small.

    Attributes:
        _metrics (Dict[str, list[float]]): Dictionary containing lists of metrics.
    """

    def __init__(self):
        """Dictionary of metrics accumulator."""
        self._metrics = {}

    def __bool__(self) -> bool:
        return self._metrics != {}

    def append(self, loss_dict: dict[str, float]) -> None:
        """Append lictionary of metrics to self"""
        if not self:
            self._metrics = self._dict_init_list(loss_dict)
        else:
            self._dict_list_append(self._metrics, loss_dict)

    def flush(self) -> dict[str, float]:
        """Calculate mean of all accumulated metrics and clear"""
        mean_metrics = {k: np.mean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics


class BatchAccumulator(DictListAccumulator):
    """A class for accumulating batches when using grad accumulation and the batch size is small.

    Attributes:
        _batches (dict[str, list[torch.Tensor]]): Dictionary containing lists of metrics.
    """

    def __init__(self, key_to_keep: str = "gsp"):
        """Batch accumulator"""
        self._batches = {}
        self.key_to_keep = key_to_keep

    def __bool__(self) -> bool:
        return self._batches != {}

    def _filter_batch_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keep_keys = [
            self.key_to_keep,
            f"{self.key_to_keep}_id",
            f"{self.key_to_keep}_t0_idx",
            f"{self.key_to_keep}_time_utc",
        ]
        return {k: v for k, v in batch.items() if k in keep_keys}

    def append(self, batch: dict[str, torch.Tensor]) -> None:
        """Append batch to self"""
        if not self:
            self._batches = self._dict_init_list(self._filter_batch_dict(batch))
        else:
            self._dict_list_append(self._batches, self._filter_batch_dict(batch))

    def flush(self) -> dict[str, torch.Tensor]:
        """Concatenate all accumulated batches, return, and clear self"""
        batch = {}
        for k, v in self._batches.items():
            if k == f"{self.key_to_keep}_t0_idx":
                batch[k] = v[0]
            else:
                batch[k] = torch.cat(v, dim=0)
        self._batches = {}
        return batch
