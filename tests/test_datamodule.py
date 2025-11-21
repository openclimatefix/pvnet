from pvnet.datamodule import PVNetDataModule



def test_data_module(data_config_path):
    """Test PVNetDataModule initialization"""

    _ = PVNetDataModule(
        configuration=data_config_path,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        train_period=[None, None],
        val_period=[None, None],
    )