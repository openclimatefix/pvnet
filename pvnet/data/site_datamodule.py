""" Data module for pytorch lightning """

from ocf_data_sampler.torch_datasets.datasets.site import SitesDataset
from pvnet.data.base_datamodule import BaseDataModule


class SitesDataModule(BaseDataModule):
    """Datamodule which streams samples using sampler for ocf-data-sampler."""

    def _get_dataset(self, start_time: str | None, end_time: str | None) -> SitesDataset:
        return SitesDataset(self.configuration, start_time=start_time, end_time=end_time)
