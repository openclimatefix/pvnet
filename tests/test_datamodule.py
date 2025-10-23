from pvnet.datamodule import PVNetDataModule



def test_sites_data_module(site_data_config_path):
    """Test PVNetDataModule initialization"""

    _ = PVNetDataModule(
        configuration=site_data_config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )