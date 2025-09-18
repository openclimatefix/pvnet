"""Encoder modules for the satellite/NWP data based on 3D concolutions.
"""

import torch
from torch import nn

from pvnet.models.late_fusion.encoders.basic_blocks import (
    AbstractNWPSatelliteEncoder,
    ResidualConv3dBlock,
)


class DefaultPVNet(AbstractNWPSatelliteEncoder):
    """This is the original encoding module used in PVNet, with a few minor tweaks."""

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        number_of_conv3d_layers: int = 4,
        conv3d_channels: int = 32,
        fc_features: int = 128,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        padding: int | tuple[int] = (1, 0, 0),
    ):
        """This is the original encoding module used in PVNet, with a few minor tweaks.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            number_of_conv3d_layers: Number of convolution 3d layers that are used.
            conv3d_channels: Number of channels used in each conv3d layer.
            fc_features: number of output nodes out of the hidden fully connected layer.
            spatial_kernel_size: The spatial size of the kernel used in the conv3d layers.
            temporal_kernel_size: The temporal size of the kernel used in the conv3d layers.
            padding: The padding used in the conv3d layers. If an int, the same padding
                is used in all dimensions
        """

        super().__init__(sequence_length, image_size_pixels, out_features)

        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        # Check that the output shape of the convolutional layers will be at least 1x1
        cnn_spatial_output_size = (
            image_size_pixels
            - ((spatial_kernel_size - 2 * padding[1]) - 1) * number_of_conv3d_layers
        )
        cnn_sequence_length = (
            sequence_length
            - ((temporal_kernel_size - 2 * padding[0]) - 1) * number_of_conv3d_layers
        )
        if not (cnn_spatial_output_size >= 1):
            raise ValueError(
                f"cannot use this many conv3d layers ({number_of_conv3d_layers}) with this input "
                f"spatial size ({image_size_pixels})"
            )

        conv_layers = []

        conv_layers += [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=conv3d_channels,
                kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
                padding=padding,
            ),
            nn.ELU(),
        ]
        for i in range(0, number_of_conv3d_layers - 1):
            conv_layers += [
                nn.Conv3d(
                    in_channels=conv3d_channels,
                    out_channels=conv3d_channels,
                    kernel_size=(temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
                    padding=padding,
                ),
                nn.ELU(),
            ]

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of the output of the 3D convolutional layers
        cnn_output_size = conv3d_channels * cnn_spatial_output_size**2 * cnn_sequence_length

        self.final_block = nn.Sequential(
            nn.Linear(in_features=cnn_output_size, out_features=fc_features),
            nn.ELU(),
            nn.Linear(in_features=fc_features, out_features=out_features),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        out = self.conv_layers(x)
        out = out.reshape(x.shape[0], -1)
        return self.final_block(out)


class ResConv3DNet(AbstractNWPSatelliteEncoder):
    """3D convolutional network based on ResNet architecture.

    The residual blocks are implemented based on the best performing block in [1].

    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        sequence_length: int,
        image_size_pixels: int,
        in_channels: int,
        out_features: int,
        hidden_channels: int = 32,
        n_res_blocks: int = 4,
        res_block_layers: int = 2,
        batch_norm: bool = True,
        dropout_frac: float = 0.0,
    ):
        """Fully connected deep network based on ResNet architecture.

        Args:
            sequence_length: The time sequence length of the data.
            image_size_pixels: The spatial size of the image. Assumed square.
            in_channels: Number of input channels.
            out_features: Number of output features.
            hidden_channels: Number of channels in middle hidden layers.
            n_res_blocks: Number of residual blocks to use.
            res_block_layers: Number of Conv3D layers used in each residual block.
            batch_norm: Whether to include batch normalisation.
            dropout_frac: Probability of an element to be zeroed in the residual pathways.
        """
        super().__init__(sequence_length, image_size_pixels, out_features)

        model = [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
            ),
        ]

        for i in range(n_res_blocks):
            model.extend(
                [
                    ResidualConv3dBlock(
                        in_channels=hidden_channels,
                        n_layers=res_block_layers,
                        dropout_frac=dropout_frac,
                        batch_norm=batch_norm,
                    ),
                    nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
                ]
            )

        # Calculate the size of the output of the 3D convolutional layers
        final_im_size = image_size_pixels // (2**n_res_blocks)
        cnn_output_size = hidden_channels * sequence_length * final_im_size * final_im_size

        model.extend(
            [
                nn.ELU(),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=cnn_output_size, out_features=out_features),
                nn.ELU(),
            ]
        )

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        return self.model(x)
