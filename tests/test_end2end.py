import lightning
from pvnet.training.lightning_module import PVNetLightningModule
from pvnet.optimizers import EmbAdamWReduceLROnPlateau

def test_model_trainer_fit(late_fusion_model, uk_streamed_datamodule):
    """Test end-to-end training."""

    ligtning_model = PVNetLightningModule(
        model=late_fusion_model,
        optimizer=EmbAdamWReduceLROnPlateau(),
    )

    # Get a sample batch for testing
    batch = next(iter(uk_streamed_datamodule.train_dataloader()))

    # Run a forward pass to verify the training module works with the data
    y = late_fusion_model(batch)

    # Train the model for one batch
    trainer = lightning.pytorch.trainer.trainer.Trainer(fast_dev_run=True, accelerator="cpu")
    trainer.fit(model=ligtning_model, datamodule=uk_streamed_datamodule)
