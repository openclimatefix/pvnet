"""Basic blocks for image sequence encoders"""
from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class AbstractNWPSatelliteEncoder(nn.Module, metaclass=ABCMeta):
    """Abstract class for NWP/satellite encoder.

    The encoder will take an input of shape (batch_size, sequence_length, channels, height, width)
    and return an output of shape (batch_size, out_features).
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
    ):
        """Abstract class for NWP/satellite encoder.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
        """
        super().__init__()
        self.out_features = out_features
        self.image_size_pixels = image_size_pixels
        self.sequence_length = sequence_length

    @abstractmethod
    def forward(self):
        """Run model forward"""
        pass


class ResidualConv3dBlock(nn.Module):
    """Residual block of 'full pre-activation' similar to the block in figure 4(e) of [1].

    This was the best performing residual block tested in the study. This implementation differs
    from that block just by using LeakyReLU activation to avoid dead neurons, and by including
    optional dropout in the residual branch. This is also a 3D fully connected layer residual block
    rather than a 2D convolutional block.

    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        dropout_frac: float = 0.0,
        batch_norm: bool = True,
    ):
        """Residual block of 'full pre-activation' similar to the block in figure 4(e) of [1].

        Sources:
            [1] https://arxiv.org/pdf/1603.05027.pdf

        Args:
            in_channels: Number of input channels.
            n_layers: Number of layers in residual pathway.
            dropout_frac: Probability of an element to be zeroed.
            batch_norm: Whether to use batchnorm
        """
        super().__init__()

        layers = []
        for i in range(n_layers):
            if batch_norm:
                layers.append(nn.BatchNorm3d(in_channels))
            layers.extend(
                [
                    nn.Dropout3d(p=dropout_frac),
                    nn.LeakyReLU(),
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(3, 3, 3),
                        padding=(1, 1, 1),
                    ),
                ]
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        return self.model(x) + x
