"""Basic blocks for PV-site encoders"""
from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class AbstractSitesEncoder(nn.Module, metaclass=ABCMeta):
    """Abstract class for encoder for output data from multiple PV sites.

    The encoder will take an input of shape (batch_size, sequence_length, num_sites)
    and return an output of shape (batch_size, out_features).
    """

    def __init__(
        self,
        sequence_length: int,
        num_sites: int,
        out_features: int,
    ):
        """Abstract class for PV site-level encoder.

        Args:
            sequence_length: The time sequence length of the data.
            num_sites: Number of PV sites in the input data.
            out_features: Number of output features.
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sites = num_sites
        self.out_features = out_features

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Run model forward"""
