from pathlib import Path
import yaml


def parse_yaml(filepath: Path) -> dict:
    with open(filepath) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    return hparams


def construct_params(_dict: dict) -> dict:
    params = {}
    for _, v in _dict.items():
        params.update(v)
    return params
