from pvnet.data import SitesDataModule



def test_sites_data_module(site_data_config_path):
    """Test SitesDataModule initialization"""

    _ = SitesDataModule(
        configuration=site_data_config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )