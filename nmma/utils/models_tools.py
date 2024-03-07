import argparse
import shutil
from os import environ, makedirs
from os.path import exists, expanduser, join
from pathlib import Path
import subprocess

import requests
from tqdm.auto import tqdm

pbar = {}

SOURCES = ["gitlab"]

# X-ray and Radio data
SKIP_FILTERS = [
    "X-ray-1keV",
    "X-ray-5keV",
    "radio-5.5GHz",
    "radio-1.25GHz",
    "radio-3GHz",
    "radio-6GHz",
]


def get_models_home(models_home=None) -> str:
    if models_home is None:
        models_home = environ.get("NMMA_MODELS", join("~", "nmma_models"))
    models_home = expanduser(models_home)
    if not exists(models_home):
        makedirs(models_home)
    return models_home


def clear_data_home(models_home=None):
    models_home = get_models_home(models_home)
    shutil.rmtree(models_home)


def download(file_info):
    url, filepath = file_info
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    chunk_size = 1024
    file_content = b""
    with tqdm(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"{str(filepath).split('/')[-1]}",
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            file_content += chunk
            pbar.update(len(chunk))

    if len(file_content) != total:
        raise ValueError(
            f"Downloaded file {filepath} is incomplete. "
            f"Only {len(file_content)} of {total} bytes were downloaded."
        )
    if not exists(Path(filepath).parent):
        try:
            makedirs(Path(filepath).parent)
        except Exception:
            pass
    with open(filepath, "wb") as f:
        f.write(file_content)

    return filepath


def decompress(file_path):
    if not file_path.endswith(".lzma"):
        raise ValueError(f"File {file_path} is not a .lzma file")
    if not exists(file_path):
        raise ValueError(f"File {file_path} does not exist")

    stdout, stderr = subprocess.Popen(
        ["lzma", "-d", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if stderr.decode("utf-8") != "" and "File exists" not in stderr.decode("utf-8"):
        raise RuntimeError(f"Error decompressing {file_path}: {stderr}")
    return file_path


def get_parser():

    parser = argparse.ArgumentParser(description="Download SVD models from GitLab")
    parser.add_argument(
        "--model", type=str, required=True, help="Name the model to be used"
    )
    parser.add_argument(
        "--svd-path",
        type=str,
        help="Path to the SVD models directory. If not provided, will use the default path",
    )
    parser.add_argument(
        "--filters",
        type=str,
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--refresh-models-list",
        type=bool,
        default=False,
        help="Refresh the list of models available on Gitlab",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="gitlab",
        help="Source of the models list. Must be one of ['gitlab']. Default is 'gitlab'",
    )

    return parser
