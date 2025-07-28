"""Basic blocks for the lienar networks"""
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
from torch import nn


class AbstractLinearNetwork(nn.Module, metaclass=ABCMeta):
    """Abstract class for a network to combine the features from all the inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Abstract class for a network to combine the features from all the inputs.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
        """
        super().__init__()

    def cat_modes(self, x: OrderedDict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Concatenate modes of input data into 1D feature vector"""
        if isinstance(x, OrderedDict):
            return torch.cat([value for key, value in x.items()], dim=1)
        elif isinstance(x, torch.Tensor):
            return x
        else:
            raise ValueError(f"Input of unexpected type {type(x)}")

    @abstractmethod
    def forward(self, x: OrderedDict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        pass


class ResidualLinearBlock(nn.Module):
    """Residual block of 'full pre-activation' similar to the block in figure 4(e) of [1].

    This was the best performing residual block tested in the study. This implementation differs
    from that block just by using LeakyReLU activation to avoid dead neuron, and by including
    optional dropout in the residual branch. This is also a 1D fully connected layer residual block
    rather than a 2D convolutional block.

    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        in_features: int,
        n_layers: int = 2,
        dropout_frac: float = 0.0,
    ):
        """Residual block of 'full pre-activation' similar to the block in figure 4(e) of [1].

        Sources:
            [1] https://arxiv.org/pdf/1603.05027.pdf

        Args:
            in_features: Number of input features.
            n_layers: Number of layers in residual pathway.
            dropout_frac: Probability of an element to be zeroed.
        """
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers += [
                nn.BatchNorm1d(in_features),
                nn.Dropout(p=dropout_frac),
                nn.LeakyReLU(),
                nn.Linear(
                    in_features=in_features,
                    out_features=in_features,
                ),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        return self.model(x) + x
