import lightning
from pvnet.data import  UKRegionalStreamedDataModule
from pvnet.training.lightning_module import PVNetLightningModule
from pvnet.optimizers import EmbAdamWReduceLROnPlateau


def test_model_trainer_fit(session_tmp_path, uk_data_config_path, late_fusion_model):
    """Test end-to-end training."""

    datamodule = UKRegionalStreamedDataModule(
        configuration=uk_data_config_path,
        batch_size=2,
        num_workers=2,
        prefetch_factor=1,
        dataset_pickle_dir=f"{session_tmp_path}/dataset_pickles"
    )

    ligtning_model = PVNetLightningModule(
        model=late_fusion_model,
        optimizer=EmbAdamWReduceLROnPlateau(),
    )

    # Train the model for one batch
    trainer = lightning.Trainer(max_epochs=2, accelerator="cpu")
    trainer.fit(model=ligtning_model, datamodule=datamodule)
