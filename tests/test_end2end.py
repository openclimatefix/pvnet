import lightning

from pvnet.optimizers import EmbAdamWReduceLROnPlateau
from pvnet.training.lightning_module import PVNetLightningModule


def test_model_trainer_fit(late_fusion_model_and_config, uk_streamed_datamodule):
    """Test end-to-end training."""

    # Unpack the model and its config from the new fixture
    late_fusion_model, model_config = late_fusion_model_and_config

    ligtning_model = PVNetLightningModule(
        model=late_fusion_model,
        optimizer=EmbAdamWReduceLROnPlateau(),
        # Pass the config object you just received
        model_config=model_config,
    )

    # Get a sample batch for testing
    batch = next(iter(uk_streamed_datamodule.train_dataloader()))

    # Run a forward pass to verify the training module works with the data
    y = late_fusion_model(batch)

    # Train the model for one batch
    trainer = lightning.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model=ligtning_model, datamodule=uk_streamed_datamodule)