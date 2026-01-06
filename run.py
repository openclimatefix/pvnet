"""Run training.

This file can be run for example using
>>  python run.py experiment=example_simple
"""

import logging
import sys

import hydra
from omegaconf import DictConfig

from pvnet.training import train
from pvnet.utils import print_config, run_config_utilities, validate_gpu_config

logging.basicConfig(stream=sys.stdout, level=logging.ERROR)


@hydra.main(config_path="configs/", config_name="config.yaml", version_base="1.2")
def main(config: DictConfig) -> None:
    """Runs training"""

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    run_config_utilities(config)
    validate_gpu_config(config)
    print_config(config, resolve=True)

    return train(config)


if __name__ == "__main__":
    main()
