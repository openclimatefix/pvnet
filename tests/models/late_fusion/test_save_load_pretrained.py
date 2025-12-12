from importlib.metadata import version
from pvnet.models import BaseModel
import pvnet.model_cards


card_path = f"{pvnet.model_cards.__path__[0]}/empty_model_card_template.md"


def test_save_pretrained(
    tmp_path, 
    late_fusion_model, 
    raw_late_fusion_model_kwargs, 
    data_config_path
):

    # Construct the model config
    model_config = {
        "_target_": "pvnet.models.LateFusionModel",
        **raw_late_fusion_model_kwargs,
    }

    # Save the model
    model_output_dir = f"{tmp_path}/saved_model"
    late_fusion_model.save_pretrained(
        save_directory=model_output_dir,
        model_config=model_config,
        data_config_path=data_config_path,
        wandb_repo="test",
        wandb_ids="abc",
        card_template_path=card_path,
        push_to_hub=False,
    )

    # Load the model
    _ = BaseModel.from_pretrained(model_id=model_output_dir, revision=None)


def test_create_hugging_face_model_card():

    # Create Hugging Face ModelCard
    card = BaseModel.create_hugging_face_model_card(card_path, wandb_repo="test", wandb_ids="abc")

    # Extract the card markdown
    card_markdown = card.content

    # Regex to find if the pvnet and ocf-data-sampler versions are present
    pvnet_version = version("pvnet")
    has_pvnet = f"pvnet=={pvnet_version}" in card_markdown

    ocf_sampler_version = version("ocf-data-sampler")
    has_ocf_data_sampler= f"ocf-data-sampler=={ocf_sampler_version}" in card_markdown

    assert has_pvnet, f"The hugging face card created does not display the PVNet package version"
    assert has_ocf_data_sampler, f"The hugging face card created does not display the ocf-data-sampler package version"
