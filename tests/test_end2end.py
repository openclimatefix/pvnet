import lightning

from pvnet.data import UKRegionalDataModule
from pvnet.optimizers import EmbAdamWReduceLROnPlateau
from pvnet.training.lightning_module import PVNetLightningModule


def test_model_trainer_fit(session_tmp_path, uk_data_config_path, late_fusion_model):
    """Test end-to-end training."""

    datamodule = UKRegionalDataModule(
        configuration=uk_data_config_path,
        batch_size=2,
        num_workers=2,
        prefetch_factor=None,
        dataset_pickle_dir=f"{session_tmp_path}/dataset_pickles"
    )

    lightning_model = PVNetLightningModule(
        model=late_fusion_model,
        optimizer=EmbAdamWReduceLROnPlateau(),
    )

    # Train the model for two batches
    trainer = lightning.Trainer(
        max_epochs=2,
        limit_val_batches=2, 
        limit_train_batches=2, 
        accelerator="cpu", 
        logger=False, 
        enable_checkpointing=False, 
    )
    trainer.fit(model=lightning_model, datamodule=datamodule)
