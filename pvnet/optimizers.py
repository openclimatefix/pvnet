"""Optimizer factory-function classes.

Optimizer classes are used by the model like:
> optimizer_generator = SomeOptimizer(**kwargs)
> optimizer = optimizer_generator(model)
The returned object must be something `pytorch_lightning`'s `configure_optimizers()` accepts.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractOptimizer(ABC):
    """Abstract class for optimizer factories"""

    def __init__(self, lr: float = 0.0005, **opt_kwargs):
        """Store the learning rate and any extra kwargs passed to the torch optimizer"""
        self.lr = lr  # overwritten by the LR finder via PVNetLightningModule
        self.opt_kwargs = opt_kwargs

    @abstractmethod
    def __call__(self, model: nn.Module):
        """Return the optimizer (and optionally scheduler) for the model"""


class Adam(AbstractOptimizer):
    """Adam optimizer"""

    def __call__(self, model: nn.Module):
        """Return optimizer"""
        return torch.optim.Adam(model.parameters(), lr=self.lr, **self.opt_kwargs)


class AdamW(AbstractOptimizer):
    """AdamW optimizer"""

    def __call__(self, model: nn.Module):
        """Return optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=self.lr, **self.opt_kwargs)


class AdamWReduceLROnPlateau(AbstractOptimizer):
    """AdamW optimizer and reduce on plateau scheduler.

    `lr` may be a float, or a dict mapping parameter-name prefixes to learning rates, with a
    "default" key used for all remaining parameters.
    """

    def __init__(
        self,
        lr: float | dict[str, float] = 0.0005,
        patience: int = 3,
        factor: float = 0.5,
        threshold: float = 2e-4,
        **opt_kwargs,
    ):
        """AdamW optimizer and reduce on plateau scheduler"""
        super().__init__(lr, **opt_kwargs)
        self.scheduler_kwargs = {"patience": patience, "factor": factor, "threshold": threshold}

    def _param_groups(self, model: nn.Module) -> tuple[list[dict], float]:
        """Return the param groups and the default learning rate"""
        if not isinstance(self.lr, dict):
            return [{"params": list(model.parameters())}], self.lr

        remaining = dict(model.named_parameters())
        groups = []
        for prefix, lr in self.lr.items():
            if prefix == "default":
                continue
            names = [n for n in remaining if n.startswith(prefix)]
            groups.append({"params": [remaining.pop(n) for n in names], "lr": lr})
        groups.append({"params": list(remaining.values())})
        return groups, self.lr["default"]

    def __call__(self, model: nn.Module) -> dict:
        """Return optimizer and scheduler"""
        groups, lr = self._param_groups(model)
        opt = torch.optim.AdamW(groups, lr=lr, **self.opt_kwargs)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **self.scheduler_kwargs)
        monitor = "quantile_loss/val" if model.use_quantile_regression else "MAE/val"
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": monitor}}


class EmbAdamWReduceLROnPlateau(AdamWReduceLROnPlateau):
    """AdamW optimizer, with no weight decay on embeddings, and reduce on plateau scheduler"""

    def __init__(self, lr: float = 0.0005, weight_decay: float = 0.01, **kwargs):
        """AdamW optimizer and reduce on plateau scheduler"""
        super().__init__(lr, **kwargs)
        self.weight_decay = weight_decay

    def _param_groups(self, model: nn.Module) -> tuple[list[dict], float]:
        """Split params into embedding (no decay) and all others (decay)"""
        emb_ids = {
            id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()
        }
        params = list(model.parameters())
        decay = [p for p in params if id(p) not in emb_ids]
        no_decay = [p for p in params if id(p) in emb_ids]
        groups = [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return groups, self.lr
