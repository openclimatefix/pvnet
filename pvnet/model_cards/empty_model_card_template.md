---
{{ card_data }}
---
<!--
Do not remove elements like the above surrounded by two curly braces and do not add any more of them. These entries are required by the PVNet library and are automaticall infilled when the model is uploaded to huggingface
-->

<!-- Title - e.g. PVNet2, WindNet, PVNet India -->
# TEMPLATE

<!-- Provide a longer summary of what this model is/does. -->
## Model Description

<!-- e.g.
This model class uses satellite data, and numerical weather predictions to forecast the near-term (up to 8 hours ahead) PV power output at all Grid Service Points (GSPs) in Great Britain. More information can be found in the model repo [1]. The model repo also includes links to our workshop paper on this model and some experimental notes.
-->

- **Developed by:** openclimatefix
- **Model type:** Fusion model
- **Language(s) (NLP):** en
- **License:** mit

# Training Details

## Data

<!-- eg.
The model is trained on data from 2019-2022 and validated on data from 2022-2023. It uses NWP data from ECMWF IFS model, and the UK Met Office UKV model. It uses satellite data from the EUMETSAT MSG SEVIRI instrument.

See the data_config.yaml file for more information on the channels and window-size used for each input data source.
-->

<!-- The preprocessing section is not strictly necessary but perhaps nice to have -->
### Preprocessing

<!-- eg.
Data is prepared with the `ocf_data_sampler/torch_datasets/datasets/pvnet_uk` Dataset [2].
-->

## Results

<!-- Do not remove the lines below -->
The training logs for this model commit can be found here:
{{ wandb_links }}

<!-- The hardware section is also just nice to have -->
### Hardware
Trained on a single NVIDIA Tesla T4

<!-- Do not remove the section below -->
### Software

This model was trained using the following Open Climate Fix packages:

- [1] https://github.com/openclimatefix/PVNet
- [2] https://github.com/openclimatefix/ocf-data-sampler

<!-- Especially do not change the two lines below -->
The versions of these packages can be found below:
{{ package_versions }}
