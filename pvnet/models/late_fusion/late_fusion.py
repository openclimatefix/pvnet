"""The default composite model architecture for PVNet"""

import logging
from collections import OrderedDict
from typing import Any

import torch
from ocf_data_sampler.numpy_sample.common_types import TensorBatch
from omegaconf import DictConfig
from torch import nn

from pvnet.models.base_model import BaseModel
from pvnet.models.late_fusion.basic_blocks import ImageEmbedding
from pvnet.models.late_fusion.encoders.basic_blocks import AbstractNWPSatelliteEncoder
from pvnet.models.late_fusion.linear_networks.basic_blocks import AbstractLinearNetwork
from pvnet.models.late_fusion.site_encoders.basic_blocks import AbstractSitesEncoder

logger = logging.getLogger(__name__)


class LateFusionModel(BaseModel):
    """Neural network which combines information from different sources

    Architecture is roughly as follows:

    - Satellite data, if included, is put through an encoder which transforms it from 4D, with time,
        channel, height, and width dimensions to become a 1D feature vector.
    - NWP, if included, is put through a similar encoder.
    - PV site-level data, if included, is put through an encoder which transforms it from 2D, with
        time and system-ID dimensions, to become a 1D feature vector.
    - The satellite features*, NWP features*, PV site-level features*, GSP ID embedding*, and sun
        paramters* are concatenated into a 1D feature vector and passed through another neural
        network to combine them and produce a forecast.

    * if included
    """

    def __init__(
        self,
        output_network: AbstractLinearNetwork,
        output_quantiles: list[float] | None = None,
        nwp_encoders_dict: dict[str, AbstractNWPSatelliteEncoder] | None = None,
        sat_encoder: AbstractNWPSatelliteEncoder | None = None,
        pv_encoder: AbstractSitesEncoder | None = None,
        add_image_embedding_channel: bool = False,
        include_gsp_yield_history: bool = True,
        include_site_yield_history: bool = False,
        include_sun: bool = True,
        include_time: bool = False,
        location_id_mapping: dict[Any, int] | None = None,
        embedding_dim: int = 16,
        forecast_minutes: int = 30,
        history_minutes: int = 60,
        sat_history_minutes: int | None = None,
        min_sat_delay_minutes: int = 30,
        nwp_forecast_minutes: DictConfig | None = None,
        nwp_history_minutes: DictConfig | None = None,
        pv_history_minutes: int | None = None,
        target_key: str = "gsp",
        interval_minutes: int = 30,
        nwp_interval_minutes: DictConfig | None = None,
        pv_interval_minutes: int = 5,
        sat_interval_minutes: int = 5,
        adapt_batches: bool = False,
    ):
        """Neural network which combines information from different sources.

        Notes:
            In the args, where it says a module `m` is partially instantiated, it means that a
            normal pytorch module will be returned by running `mod = m(**kwargs)`. In this library,
            this partial instantiation is generally achieved using partial instantiation via hydra.
            However, the arg is still valid as long as `m(**kwargs)` returns a valid pytorch module
            - for example if `m` is a regular function.

        Args:
            output_network: A partially instantiated pytorch Module class used to combine the 1D
                features to produce the forecast.
            output_quantiles: A list of float (0.0, 1.0) quantiles to predict values for. If set to
                None the output is a single value.
            nwp_encoders_dict: A dictionary of partially instantiated pytorch Module class used to
                encode the NWP data from 4D into a 1D feature vector from different sources.
            sat_encoder: A partially instantiated pytorch Module class used to encode the satellite
                data from 4D into a 1D feature vector.
            pv_encoder: A partially instantiated pytorch Module class used to encode the site-level
                PV data from 2D into a 1D feature vector.
            add_image_embedding_channel: Add a channel to the NWP and satellite data with the
                embedding of the GSP ID.
            include_gsp_yield_history: Include GSP yield data.
            include_site_yield_history: Include Site yield data.
            include_sun: Include sun azimuth and altitude data.
            include_time: Include sine and cosine of dates and times.
            location_id_mapping: A dictionary mapping the location ID to an integer. ID embedding is
                not used if this is not provided.
            embedding_dim: Number of embedding dimensions to use for GSP ID.
            forecast_minutes: The amount of minutes that should be forecasted.
            history_minutes: The default amount of historical minutes that are used.
            sat_history_minutes: Length of recent observations used for satellite inputs. Defaults
                to `history_minutes` if not provided.
            min_sat_delay_minutes: Minimum delay with respect to t0 of the latest available
                satellite image.
            nwp_forecast_minutes: Period of future NWP forecast data used as input. Defaults to
                `forecast_minutes` if not provided.
            nwp_history_minutes: Period of historical NWP forecast used as input. Defaults to
                `history_minutes` if not provided.
            pv_history_minutes: Length of recent site-level PV data used as
                input. Defaults to `history_minutes` if not provided.
            target_key: The key of the target variable in the batch.
            interval_minutes: The interval between each sample of the target data
            nwp_interval_minutes: Dictionary of the intervals between each sample of the NWP
                data for each source
            pv_interval_minutes: The interval between each sample of the PV data
            sat_interval_minutes: The interval between each sample of the satellite data
            adapt_batches: If set to true, we attempt to slice the batches to the expected shape for
                the model to use. This allows us to overprepare batches and slice from them for the
                data we need for a model run.
        """
        super().__init__(
            history_minutes=history_minutes,
            forecast_minutes=forecast_minutes,
            output_quantiles=output_quantiles,
            target_key=target_key,
            interval_minutes=interval_minutes,
        )

        self.include_gsp_yield_history = include_gsp_yield_history
        self.include_site_yield_history = include_site_yield_history
        self.include_sat = sat_encoder is not None
        self.include_nwp = nwp_encoders_dict is not None and len(nwp_encoders_dict) != 0
        self.include_pv = pv_encoder is not None
        self.include_sun = include_sun
        self.include_time = include_time
        self.location_id_mapping = location_id_mapping
        self.embedding_dim = embedding_dim
        self.add_image_embedding_channel = add_image_embedding_channel
        self.interval_minutes = interval_minutes
        self.min_sat_delay_minutes = min_sat_delay_minutes
        self.adapt_batches = adapt_batches

        if self.location_id_mapping is None:
            logger.warning(
                "location_id_mapping` is not provided, defaulting to outdated GSP mapping"
                "(0 to 317)"
            )

            # Note 318 is the 2024 UK GSP count, so this is a temporary fix
            # for models trained with this default embedding
            self.location_id_mapping = {i: i for i in range(318)}

        # in the future location_id_mapping could be None,
        # and in this case use_id_embedding should be False
        self.use_id_embedding = self.embedding_dim is not None

        if self.use_id_embedding:
            num_embeddings = max(self.location_id_mapping.values()) + 1

        # Number of features expected by the output_network
        # Add to this as network pieces are constructed
        fusion_input_features = 0

        if self.include_sat:
            # Param checks
            assert sat_history_minutes is not None

            self.sat_sequence_len = (
                sat_history_minutes - min_sat_delay_minutes
            ) // sat_interval_minutes + 1

            self.sat_encoder = sat_encoder(
                sequence_length=self.sat_sequence_len,
                in_channels=sat_encoder.keywords["in_channels"] + add_image_embedding_channel,
            )
            if add_image_embedding_channel:
                self.sat_embed = ImageEmbedding(
                    num_embeddings, self.sat_sequence_len, self.sat_encoder.image_size_pixels
                )

            # Update num features
            fusion_input_features += self.sat_encoder.out_features

        if self.include_nwp:
            # Param checks
            assert nwp_forecast_minutes is not None
            assert nwp_history_minutes is not None

            # For each NWP encoder the forecast and history minutes must be set
            assert set(nwp_encoders_dict.keys()) == set(nwp_forecast_minutes.keys())
            assert set(nwp_encoders_dict.keys()) == set(nwp_history_minutes.keys())

            if nwp_interval_minutes is None:
                nwp_interval_minutes = dict.fromkeys(nwp_encoders_dict.keys(), 60)

            self.nwp_encoders_dict = torch.nn.ModuleDict()
            if add_image_embedding_channel:
                self.nwp_embed_dict = torch.nn.ModuleDict()

            for nwp_source in nwp_encoders_dict.keys():
                nwp_sequence_len = (
                    nwp_history_minutes[nwp_source] // nwp_interval_minutes[nwp_source]
                    + nwp_forecast_minutes[nwp_source] // nwp_interval_minutes[nwp_source]
                    + 1
                )

                self.nwp_encoders_dict[nwp_source] = nwp_encoders_dict[nwp_source](
                    sequence_length=nwp_sequence_len,
                    in_channels=(
                        nwp_encoders_dict[nwp_source].keywords["in_channels"]
                        + add_image_embedding_channel
                    ),
                )
                if add_image_embedding_channel:
                    self.nwp_embed_dict[nwp_source] = ImageEmbedding(
                        num_embeddings,
                        nwp_sequence_len,
                        self.nwp_encoders_dict[nwp_source].image_size_pixels,
                    )

                # Update num features
                fusion_input_features += self.nwp_encoders_dict[nwp_source].out_features

        if self.include_pv:
            assert pv_history_minutes is not None

            self.pv_encoder = pv_encoder(
                sequence_length=pv_history_minutes // pv_interval_minutes + 1,
                target_key_to_use=self._target_key,
                input_key_to_use="site",
            )

            # Update num features
            fusion_input_features += self.pv_encoder.out_features

        if self.use_id_embedding:
            self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

            # Update num features
            fusion_input_features += embedding_dim

        if self.include_sun:
            self.sun_fc1 = nn.Linear(
                in_features=2
                * (self.forecast_len + self.history_len + 1),
                out_features=16,
            )

            # Update num features
            fusion_input_features += 16

        if self.include_time:
            self.time_fc1 = nn.Linear(
                in_features=4
                * (self.forecast_len + self.history_len + 1),
                out_features=32,
            )

            # Update num features
            fusion_input_features += 32

        if include_gsp_yield_history:
            # Update num features
            fusion_input_features += self.history_len

        if include_site_yield_history:
            # Update num features
            fusion_input_features += self.history_len + 1

        self.output_network = output_network(
            in_features=fusion_input_features,
            out_features=self.num_output_features,
        )


    def forward(self, x: TensorBatch) -> torch.Tensor:
        """Run model forward"""

        if self.adapt_batches:
            x = self._adapt_batch(x)

        if self.use_id_embedding:
            # eg: x['gsp_id'] = [1] with location_id_mapping = {1:0}, would give [0]
            id = torch.tensor(
                [self.location_id_mapping[i.item()] for i in x[f"{self._target_key}_id"]],
                device=x[f"{self._target_key}_id"].device,
                dtype=torch.int64,
            )

        modes = OrderedDict()
        # ******************* Satellite imagery *************************
        if self.include_sat:
            # Shape: batch_size, seq_length, channel, height, width
            sat_data = x["satellite_actual"][:, : self.sat_sequence_len]
            sat_data = torch.swapaxes(sat_data, 1, 2).float()  # switch time and channels

            if self.add_image_embedding_channel:
                sat_data = self.sat_embed(sat_data, id)
            modes["sat"] = self.sat_encoder(sat_data)

        # *********************** NWP Data ************************************
        if self.include_nwp:
            # Loop through potentially many NMPs
            for nwp_source in self.nwp_encoders_dict:
                # Shape: batch_size, seq_len, n_chans, height, width
                nwp_data = x["nwp"][nwp_source]["nwp"].float()
                nwp_data = torch.swapaxes(nwp_data, 1, 2)  # Switch time and channels
                # Some NWP variables in our input data have overflowed to NaN
                nwp_data = torch.clip(nwp_data, min=-50, max=50)

                if self.add_image_embedding_channel:
                    nwp_data = self.nwp_embed_dict[nwp_source](nwp_data, id)

                nwp_out = self.nwp_encoders_dict[nwp_source](nwp_data)
                modes[f"nwp/{nwp_source}"] = nwp_out

        # *********************** Site Data *************************************
        # Add site-level yield history
        if self.include_site_yield_history:
            site_history = x["site"][:, : self.history_len + 1].float()
            site_history = site_history.reshape(site_history.shape[0], -1)
            modes["site"] = site_history

        # Add site-level yield history through PV encoder
        if self.include_pv:
            if self._target_key != "site":
                modes["site"] = self.pv_encoder(x)
            else:
                # Target is PV, so only take the history
                # Copy batch
                x_tmp = x.copy()
                x_tmp["site"] = x_tmp["site"][:, : self.history_len + 1]
                modes["site"] = self.pv_encoder(x_tmp)

        # *********************** GSP Data ************************************
        # Add gsp yield history
        if self.include_gsp_yield_history:
            gsp_history = x["gsp"][:, : self.history_len].float()
            gsp_history = gsp_history.reshape(gsp_history.shape[0], -1)
            modes["gsp"] = gsp_history

        # ********************** Embedding of GSP/Site ID ********************
        if self.use_id_embedding:
            modes["id"] = self.embed(id)

        if self.include_sun:
            sun = torch.cat((x["solar_azimuth"], x["solar_elevation"]), dim=1).float()
            sun = self.sun_fc1(sun)
            modes["sun"] = sun

        if self.include_time:
            time = torch.cat(
                (
                    x[f"{self._target_key}_date_sin"],
                    x[f"{self._target_key}_date_cos"],
                    x[f"{self._target_key}_time_sin"],
                    x[f"{self._target_key}_time_cos"],
                ),
                dim=1,
            ).float()
            time = self.time_fc1(time)
            modes["time"] = time

        out = self.output_network(modes)

        if self.use_quantile_regression:
            # Shape: batch_size, seq_length * num_quantiles
            out = out.reshape(out.shape[0], self.forecast_len, len(self.output_quantiles))

        return out
