"""Linear networks used for the fusion model"""
from collections import OrderedDict

import torch
from torch import nn

from pvnet.models.late_fusion.linear_networks.basic_blocks import (
    AbstractLinearNetwork,
    ResidualLinearBlock,
)


class ResFCNet(AbstractLinearNetwork):
    """Fully connected deep network based on ResNet architecture.

    This architecture is similar to
    `ResFCNet`, except that it uses LeakyReLU activations internally, and batchnorm in the residual
    branches. The residual blocks are implemented based on the best performing block in [1].

    Sources:
        [1] https://arxiv.org/pdf/1603.05027.pdf
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fc_hidden_features: int = 128,
        n_res_blocks: int = 4,
        res_block_layers: int = 2,
        dropout_frac: float = 0.0,
    ):
        """Fully connected deep network based on ResNet architecture.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            fc_hidden_features: Number of features in middle hidden layers.
            n_res_blocks: Number of residual blocks to use.
            res_block_layers: Number of fully-connected layers used in each residual block.
            dropout_frac: Probability of an element to be zeroed in the residual pathways.
        """
        super().__init__(in_features, out_features)

        model = [nn.Linear(in_features=in_features, out_features=fc_hidden_features)]

        for i in range(n_res_blocks):
            model += [
                ResidualLinearBlock(
                    in_features=fc_hidden_features,
                    n_layers=res_block_layers,
                    dropout_frac=dropout_frac,
                )
            ]

        model += [
            nn.LeakyReLU(),
            nn.Linear(in_features=fc_hidden_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.01),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x: OrderedDict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Run model forward"""
        x = self.cat_modes(x)
        return self.model(x)
