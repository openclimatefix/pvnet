# PVNet
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-21-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/PVNet?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/PVNet/tags)
[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#overview-of-ocfs-nowcasting-repositories)


This project is used for training PVNet and running PVNet on live data.

PVNet is a multi-modal late-fusion model for predicting renewable energy generation from weather 
data. The NWP (Numerical Weather Prediction) and satellite data are sent through a neural network 
which encodes them down to 1D intermediate representations. These are concatenated together with 
recent generation, the calculated solar coordinates (azimuth and elevation) and the location ID 
which has been put through an embedding layer. This 1D concatenated feature vector is put through 
an output network which outputs predictions of the future energy yield.


## Experiments

Our paper based on this repo was accepted into the Tackling Climate Change with Machine Learning 
workshop at ICLR 2024 and can be viewed [here](https://www.climatechange.ai/papers/iclr2024/46).

Some more structured notes on experiments we have performed with PVNet are 
[here](https://docs.google.com/document/d/1VumDwWd8YAfvXbOtJEv3ZJm_FHQDzrKXR0jU9vnvGQg).


## Setup / Installation

```bash
git clone git@github.com:openclimatefix/PVNet.git
cd PVNet
pip install .
```

The commit history is extensive. To save download time, use a depth of 1:
```bash
git clone --depth 1 git@github.com:openclimatefix/PVNet.git
```
This means only the latest commit and its associated files will be downloaded.

Next, in the PVNet repo, install PVNet as an editable package:

```bash
pip install -e .
```

### Additional development dependencies

```bash
pip install ".[dev]"
```



## Getting started with running PVNet

Before running any code in PVNet, copy the example configuration to a
configs directory:

```
cp -r configs.example configs
```

You will be making local amendments to these configs. See the README in
`configs.example` for more info.

### Datasets

As a minimum, in order to create samples of data/run PVNet, you will need to
supply paths to NWP and GSP data. PV data can also be used. We list some
suggested locations for downloading such datasets below:

**GSP (Grid Supply Point)** - Regional PV generation data\
The University of Sheffield provides API access to download this data:
https://www.solar.sheffield.ac.uk/api/

Documentation for querying generation data aggregated by GSP region can be found
here:
https://docs.google.com/document/d/e/2PACX-1vSDFb-6dJ2kIFZnsl-pBQvcH4inNQCA4lYL9cwo80bEHQeTK8fONLOgDf6Wm4ze_fxonqK3EVBVoAIz/pub#h.9d97iox3wzmd

**NWP (Numerical weather predictions)**\
OCF maintains a Zarr formatted version of the German Weather Service's (DWD)
ICON-EU NWP model here:
https://huggingface.co/datasets/openclimatefix/dwd-icon-eu which includes the UK

**PV**\
OCF maintains a dataset of PV generation from 1311 private PV installations
here: https://huggingface.co/datasets/openclimatefix/uk_pv


### Connecting with ocf-data-sampler for sample creation

Outside the PVNet repo, clone the ocf-data-sampler repo and exit the conda env created for PVNet: https://github.com/openclimatefix/ocf-data-sampler
```bash
git clone git@github.com/openclimatefix/ocf-data-sampler.git
conda create -n ocf-data-sampler python=3.11
```

Then go inside the ocf-data-sampler repo to add packages

```bash
pip install .
```

Then exit this environment, and enter back into the pvnet conda environment and install ocf-data-sampler in editable mode (-e). This means the package is directly linked to the source code in the ocf-data-sampler repo.

```bash
pip install -e <PATH-TO-ocf-data-sampler-REPO>
```

If you install the local version of `ocf-data-sampler` that is more recent than the version 
specified in `PVNet` it is not guarenteed to function properly with this library.


### Set up and config example for streaming

We will use the following example config file to describe your data sources: `/PVNet/configs/datamodule/configuration/example_configuration.yaml`. Ensure that the file paths are set to the correct locations in `example_configuration.yaml`: search for `PLACEHOLDER` to find where to input the location of the files. Delete or comment the parts for data you are not using.

At run time, the datamodule config `PVNet/configs/datamodule/streamed_samples.yaml` points to your chosen configuration file:

configuration: "/FULL-PATH-TO-REPO/PVNet/configs/datamodule/configuration/example_configuration.yaml"

You can also update train/val/test time ranges here to match the period you have access to.

If downloading private data from a GCP bucket make sure to authenticate gcloud (the public satellite data does not need authentication):

gcloud auth login

You can provide multiple storage locations as a list. For example:

satellite:
  zarr_path:
    - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2020_nonhrv.zarr"
    - "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4/2021_nonhrv.zarr"

`ocf-data-sampler` is currently set up to use 11 channels from the satellite data (the 12th, HRV, is not used).

⚠️ NB: Our publicly accessible satellite data is currently saved with a blosc2 compressor, which is not supported by the tensorstore backend PVNet relies on now. We are in the process of updating this; for now, the paths above cannot be used with this codebase.

### Training PVNet

How PVNet is run is determined by the configuration files. The example configs in `PVNet/configs.example` work with **streamed_samples** using `datamodule/streamed_samples.yaml`.

Update the following before training:

1. In `configs/model/late_fusion.yaml`:
    - Update the list of encoders to match the data sources you are using. For different NWP sources, keep the same structure but ensure:
        - `in_channels`: the number of variables your NWP source supplies
        - `image_size_pixels`: spatial crop matching your NWP resolution and the settings in your datamodule configuration (unless you coarsened, e.g. for ECMWF)
2. In `configs/trainer/default.yaml`:
    - Set `accelerator: 0` if running on a system without a supported GPU
3. In `configs/datamodule/streamed_samples.yaml`:
    - Point `configuration:` to your local `example_configuration.yaml` (or your custom one)
    - Adjust the train/val/test time ranges to your available data

If you create custom config files, update the main `./configs/config.yaml` defaults:

defaults:
  - trainer: default.yaml
  - model: late_fusion.yaml
  - datamodule: streamed_samples.yaml
  - callbacks: null
  - experiment: null
  - hparams_search: null
  - hydra: default.yaml

Now train PVNet:

python run.py

You can override any setting with Hydra, e.g.:

python run.py datamodule=streamed_samples datamodule.configuration="/FULL-PATH/PVNet/configs/datamodule/configuration/example_configuration.yaml"

## Backtest

If you have successfully trained a PVNet model and have a saved model checkpoint you can create a backtest using this, e.g. forecasts on historical data to evaluate forecast accuracy/skill. This can be done by running one of the scripts in this repo such as [the UK GSP backtest script](scripts/backtest_uk_gsp.py) or the [the pv site backtest script](scripts/backtest_sites.py), further info on how to run these are in each backtest file.

## Testing

You can use `python -m pytest tests` to run tests

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/felix-e-h-p"><img src="https://avatars.githubusercontent.com/u/137530077?v=4?s=100" width="100px;" alt="Felix"/><br /><sub><b>Felix</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=felix-e-h-p" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sukh-P"><img src="https://avatars.githubusercontent.com/u/42407101?v=4?s=100" width="100px;" alt="Sukhil Patel"/><br /><sub><b>Sukhil Patel</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=Sukh-P" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AUdaltsova"><img src="https://avatars.githubusercontent.com/u/43303448?v=4?s=100" width="100px;" alt="Alexandra Udaltsova"/><br /><sub><b>Alexandra Udaltsova</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=AUdaltsova" title="Code">💻</a> <a href="https://github.com/openclimatefix/pvnet/pulls?q=is%3Apr+reviewed-by%3AAUdaltsova" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zakwatts"><img src="https://avatars.githubusercontent.com/u/47150349?v=4?s=100" width="100px;" alt="Megawattz"/><br /><sub><b>Megawattz</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=zakwatts" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mahdilamb"><img src="https://avatars.githubusercontent.com/u/4696915?v=4?s=100" width="100px;" alt="Mahdi Lamb"/><br /><sub><b>Mahdi Lamb</b></sub></a><br /><a href="#infra-mahdilamb" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://www.jacobbieker.com"><img src="https://avatars.githubusercontent.com/u/7170359?v=4?s=100" width="100px;" alt="Jacob Prince-Bieker"/><br /><sub><b>Jacob Prince-Bieker</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=jacobbieker" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/codderrrrr"><img src="https://avatars.githubusercontent.com/u/149995852?v=4?s=100" width="100px;" alt="codderrrrr"/><br /><sub><b>codderrrrr</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=codderrrrr" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chrisxbriggs.com"><img src="https://avatars.githubusercontent.com/u/617309?v=4?s=100" width="100px;" alt="Chris Briggs"/><br /><sub><b>Chris Briggs</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=confusedmatrix" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/tmi"><img src="https://avatars.githubusercontent.com/u/147159?v=4?s=100" width="100px;" alt="tmi"/><br /><sub><b>tmi</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=tmi" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://rdrn.me/"><img src="https://avatars.githubusercontent.com/u/19817302?v=4?s=100" width="100px;" alt="Chris Arderne"/><br /><sub><b>Chris Arderne</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=carderne" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Dakshbir"><img src="https://avatars.githubusercontent.com/u/144359831?v=4?s=100" width="100px;" alt="Dakshbir"/><br /><sub><b>Dakshbir</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=Dakshbir" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MAYANK12SHARMA"><img src="https://avatars.githubusercontent.com/u/145884197?v=4?s=100" width="100px;" alt="MAYANK SHARMA"/><br /><sub><b>MAYANK SHARMA</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=MAYANK12SHARMA" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lambaaryan011"><img src="https://avatars.githubusercontent.com/u/153702847?v=4?s=100" width="100px;" alt="aryan lamba "/><br /><sub><b>aryan lamba </b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=lambaaryan011" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/michael-gendy"><img src="https://avatars.githubusercontent.com/u/64384201?v=4?s=100" width="100px;" alt="michael-gendy"/><br /><sub><b>michael-gendy</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=michael-gendy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://adityasuthar.github.io/"><img src="https://avatars.githubusercontent.com/u/95685363?v=4?s=100" width="100px;" alt="Aditya Suthar"/><br /><sub><b>Aditya Suthar</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=adityasuthar" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/markus-kreft"><img src="https://avatars.githubusercontent.com/u/129367085?v=4?s=100" width="100px;" alt="Markus Kreft"/><br /><sub><b>Markus Kreft</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=markus-kreft" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://jack-kelly.com"><img src="https://avatars.githubusercontent.com/u/460756?v=4?s=100" width="100px;" alt="Jack Kelly"/><br /><sub><b>Jack Kelly</b></sub></a><br /><a href="#ideas-JackKelly" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zaryab-ali"><img src="https://avatars.githubusercontent.com/u/85732412?v=4?s=100" width="100px;" alt="zaryab-ali"/><br /><sub><b>zaryab-ali</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=zaryab-ali" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Lex-Ashu"><img src="https://avatars.githubusercontent.com/u/181084934?v=4?s=100" width="100px;" alt="Lex-Ashu"/><br /><sub><b>Lex-Ashu</b></sub></a><br /><a href="https://github.com/openclimatefix/pvnet/commits?author=Lex-Ashu" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
