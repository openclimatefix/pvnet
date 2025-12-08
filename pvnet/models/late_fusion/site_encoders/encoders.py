"""Encoder modules for the site-level PV data."""

import einops
import torch
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from torch import nn

from pvnet.models.late_fusion.linear_networks.networks import ResFCNet
from pvnet.models.late_fusion.site_encoders.basic_blocks import AbstractSitesEncoder


# TODO update this to work with the new sample data format
class SimpleLearnedAggregator(AbstractSitesEncoder):
    """A simple model which learns a different weighted-average across all PV sites for each GSP.

    Each sequence from each site is independently encodeded through some dense layers wih skip-
    connections, then the encoded form of each sequence is aggregated through a learned weighted-sum
    and finally put through more dense layers.

    This model was written to be a simplified version of a single-headed attention layer.
    """

    def __init__(
        self,
        sequence_length: int,
        num_sites: int,
        out_features: int,
        value_dim: int = 10,
        value_enc_resblocks: int = 2,
        final_resblocks: int = 2,
    ):
        """A simple sequence encoder and weighted-average model.

        Args:
            sequence_length: The time sequence length of the data.
            num_sites: Number of PV sites in the input data.
            out_features: Number of output features.
            value_dim: The number of features in each encoded sequence. Similar to the value
                dimension in single- or multi-head attention.
            value_dim: The number of features in each encoded sequence. Similar to the value
                dimension in single- or multi-head attention.
            value_enc_resblocks: Number of residual blocks in the value-encoder sub-network.
            final_resblocks: Number of residual blocks in the final sub-network.
        """

        super().__init__(sequence_length, num_sites, out_features)

        # Network used to encode each PV site sequence
        self._value_encoder = nn.Sequential(
            ResFCNet(
                in_features=sequence_length,
                out_features=value_dim,
                fc_hidden_features=value_dim,
                n_res_blocks=value_enc_resblocks,
                res_block_layers=2,
                dropout_frac=0,
            ),
        )

        # The learned weighted average is stored in an embedding layer for ease of use
        self._attention_network = nn.Sequential(
            nn.Embedding(318, num_sites),
            nn.Softmax(dim=1),
        )

        # Network used to process weighted average
        self.output_network = ResFCNet(
            in_features=value_dim,
            out_features=out_features,
            fc_hidden_features=value_dim,
            n_res_blocks=final_resblocks,
            res_block_layers=2,
            dropout_frac=0,
        )

    def _calculate_attention(self, x: TensorBatch) -> torch.Tensor:
        gsp_ids = x["gsp_id"].squeeze().int()
        attention = self._attention_network(gsp_ids)
        return attention

    def _encode_value(self, x: TensorBatch) -> torch.Tensor:
        # Shape: [batch size, sequence length, PV site]
        pv_site_seqs = x["pv"].float()
        batch_size = pv_site_seqs.shape[0]

        pv_site_seqs = pv_site_seqs.swapaxes(1, 2).flatten(0, 1)

        x_seq_enc = self._value_encoder(pv_site_seqs)
        x_seq_out = x_seq_enc.unflatten(0, (batch_size, self.num_sites))
        return x_seq_out

    def forward(self, x: TensorBatch) -> torch.Tensor:
        """Run model forward"""
        # Output has shape: [batch size, num_sites, value_dim]
        encodeded_seqs = self._encode_value(x)

        # Calculate learned averaging weights
        attn_avg_weights = self._calculate_attention(x)

        # Take weighted average across num_sites
        value_weighted_avg = (encodeded_seqs * attn_avg_weights.unsqueeze(-1)).sum(dim=1)

        # Put through final processing layers
        x_out = self.output_network(value_weighted_avg)

        return x_out


class SingleAttentionNetwork(AbstractSitesEncoder):
    """A simple attention-based model with a single multihead attention layer

    For the attention layer the query is based on the target alone, the key is based on the
    input ID and the recent input data, the value is based on the recent input data.

    """

    def __init__(
        self,
        sequence_length: int,
        num_sites: int,
        out_features: int,
        kdim: int = 10,
        id_embed_dim: int = 10,
        num_heads: int = 2,
        n_kv_res_blocks: int = 2,
        kv_res_block_layers: int = 2,
        use_id_in_value: bool = False,
        target_id_dim: int = 318,
        key_to_use: str = "generation",
        num_channels: int = 1,
        num_sites_in_inference: int = 1,
    ):
        """A simple attention-based model with a single multihead attention layer

        Args:
            sequence_length: The time sequence length of the data.
            num_sites: Number of sites in the input data.
            out_features: Number of output features. In this network this is also the embed and
                value dimension in the multi-head attention layer.
            kdim: The dimensions used the keys.
            id_embed_dim: Number of dimensiosn used in the site ID embedding layer(s).
            num_heads: Number of parallel attention heads. Note that `out_features` will be split
                across `num_heads` so `out_features` must be a multiple of `num_heads`.
            n_kv_res_blocks: Number of residual blocks to use in the key and value encoders.
            kv_res_block_layers: Number of fully-connected layers used in each residual block within
                the key and value encoders.
            use_id_in_value: Whether to use a site ID embedding in network used to produce the
                value for the attention layer.
            target_id_dim: The number of unique IDs.
            key_to_use: The key to use in the attention layer.
            num_channels: Number of channels in the input data
            num_sites_in_inference: Number of sites to use in inference.
                This is used to determine the number of sites to use in the
                attention layer, for a single site, 1 works, while for multiple sites
                this would be higher than that

        """
        super().__init__(sequence_length, num_sites, out_features)
        self.sequence_length = sequence_length
        self.target_id_embedding = nn.Embedding(target_id_dim, out_features)
        self.site_id_embedding = nn.Embedding(num_sites, id_embed_dim)
        self._ids = nn.parameter.Parameter(torch.arange(num_sites), requires_grad=False)
        self.use_id_in_value = use_id_in_value
        self.key_to_use = key_to_use
        self.num_channels = num_channels
        self.num_sites_in_inference = num_sites_in_inference

        if use_id_in_value:
            self.value_id_embedding = nn.Embedding(num_sites, id_embed_dim)

        self._value_encoder = nn.Sequential(
            ResFCNet(
                in_features=sequence_length * self.num_channels
                + int(use_id_in_value) * id_embed_dim,
                out_features=out_features,
                fc_hidden_features=sequence_length * self.num_channels,
                n_res_blocks=n_kv_res_blocks,
                res_block_layers=kv_res_block_layers,
                dropout_frac=0,
            ),
        )

        self._key_encoder = nn.Sequential(
            ResFCNet(
                in_features=id_embed_dim + sequence_length * self.num_channels,
                out_features=kdim,
                fc_hidden_features=id_embed_dim + sequence_length * self.num_channels,
                n_res_blocks=n_kv_res_blocks,
                res_block_layers=kv_res_block_layers,
                dropout_frac=0,
            ),
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=out_features,
            kdim=kdim,
            vdim=out_features,
            num_heads=num_heads,
            batch_first=True,
        )

    def _encode_inputs(self, x: TensorBatch) -> tuple[torch.Tensor, int]:
        # Shape: [batch size, sequence length, number of sites]
        # Shape: [batch size,  station_id, sequence length,  channels]
        input_data = x[f"{self.key_to_use}"]
        if len(input_data.shape) == 2:  # one site per sample
            input_data = input_data.unsqueeze(-1)  # add dimension of 1 to end to make 3D
        if len(input_data.shape) == 4:  # Has multiple channels
            input_data = input_data[:, :, : self.sequence_length]
            input_data = einops.rearrange(input_data, "b id s c -> b (s c) id")
        else:
            input_data = input_data[:, : self.sequence_length]
        site_seqs = input_data.float()
        batch_size = site_seqs.shape[0]
        site_seqs = site_seqs.swapaxes(1, 2)  # [batch size, location ID, sequence length]
        return site_seqs, batch_size

    def _encode_query(self, x: TensorBatch) -> torch.Tensor:
        ids = x["location_id"].int()
        query = self.target_id_embedding(ids).unsqueeze(1)
        return query

    def _encode_key(self, x: TensorBatch) -> torch.Tensor:
        site_seqs, batch_size = self._encode_inputs(x)

        # site ID embeddings are the same for each sample
        id_embed = torch.tile(self.site_id_embedding(self._ids), (batch_size, 1, 1))
        # Each concated (site sequence, site ID embedding) is processed with encoder
        x_seq_in = torch.cat((site_seqs, id_embed), dim=2).flatten(0, 1)
        key = self._key_encoder(x_seq_in)

        # Reshape to [batch size, site, kdim]
        key = key.unflatten(0, (batch_size, self.num_sites))
        return key

    def _encode_value(self, x: TensorBatch) -> torch.Tensor:
        site_seqs, batch_size = self._encode_inputs(x)

        if self.use_id_in_value:
            # site ID embeddings are the same for each sample
            id_embed = torch.tile(self.value_id_embedding(self._ids), (batch_size, 1, 1))
            # Each concated (site sequence, site ID embedding) is processed with encoder
            x_seq_in = torch.cat((site_seqs, id_embed), dim=2).flatten(0, 1)
        else:
            # Encode each site sequence independently
            x_seq_in = site_seqs.flatten(0, 1)
        value = self._value_encoder(x_seq_in)

        # Reshape to [batch size, site, vdim]
        value = value.unflatten(0, (batch_size, self.num_sites))
        return value

    def _attention_forward(
        self, x: dict, average_attn_weights: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor :]:
        query = self._encode_query(x)
        key = self._encode_key(x)
        value = self._encode_value(x)
        attn_output, attn_weights = self.multihead_attn(
            query, key, value, average_attn_weights=average_attn_weights
        )

        return attn_output, attn_weights

    def forward(self, x: TensorBatch) -> torch.Tensor:
        """Run model forward"""

        attn_output, _ = self._attention_forward(x)

        # Reshape from [batch_size, 1, vdim] to [batch_size, vdim]
        x_out = attn_output.squeeze()
        if len(x_out.shape) == 1:
            x_out = x_out.unsqueeze(0)

        return x_out
