""" Data module for pytorch lightning """

from ocf_data_sampler.torch_datasets.datasets.pvnet_uk import PVNetUKRegionalDataset
from pvnet.data.base_datamodule import BaseDataModule


class UKRegionalDataModule(BaseDataModule):
    """Datamodule which streams samples using sampler for ocf-data-sampler."""

    def _get_dataset(self, start_time: str | None, end_time: str | None) -> PVNetUKRegionalDataset:
        return PVNetUKRegionalDataset(self.configuration, start_time=start_time, end_time=end_time)
