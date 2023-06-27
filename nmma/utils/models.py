import argparse
import re
import shutil
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from os import environ, makedirs
from os.path import exists, expanduser, join
from pathlib import Path

import requests
from requests.exceptions import ConnectionError
from tqdm.auto import tqdm
from yaml import load

PERMANENT_DOI = "8039909"
DOI = ""
MODELS = {}

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])

pbar = {}

# X-ray and Radio data
custom_filters = [
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
    makedirs(models_home, exist_ok=True)
    return models_home


def clear_data_home(models_home=None):
    models_home = get_models_home(models_home)
    shutil.rmtree(models_home)


def get_latest_zenodo_doi(permanent_doi):
    headers = {  # we emulate a browser request with a recent chrome version to avoid zenodo blocking us
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "*/*",
        "Host": "zenodo.org",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    r = requests.get(
        f"https://zenodo.org/record/{permanent_doi}",
        allow_redirects=True,
        headers=headers,
    )
    try:
        data = r.text.split("10.5281/zenodo.")[1]
        doi = re.findall(r"^\d+", data)[0]
    except Exception as e:
        raise ValueError(f"Could not find latest DOI: {str(e)}")
    return doi


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
    with open(filepath, "wb") as f:
        f.write(file_content)

    return filepath


def download_models_list(doi=None, models_home=None):
    # first we load the models list from zenodo
    models_home = get_models_home(models_home)
    if not exists(models_home):
        makedirs(models_home)
    r = requests.get(
        f"https://zenodo.org/record/{doi}/files/models.yaml", allow_redirects=True
    )
    with open(Path(models_home, "models.yaml"), "wb") as f:
        f.write(r.content)


def load_models_list(doi=None, models_home=None):
    # if models.yaml doesn't exist, download it
    global DOI
    models_home = get_models_home(models_home)
    if not exists(Path(models_home, "models.yaml")):
        print("Models list not found, downloading it from Zenodo")
        try:
            if doi in [None, ""]:
                DOI = get_latest_zenodo_doi(PERMANENT_DOI)
                doi = DOI
            download_models_list(doi=DOI, models_home=models_home)
        except ConnectionError:
            print(
                "Could not connect to Zenodo, models might not be available or up-to-date"
            )
            pass
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    with open(Path(models_home, "models.yaml"), "r") as f:
        models = load(f, Loader=Loader)

    # temporary mapping
    models["Bu2019lm"] = models["Bu2019bns"]
    return models


try:
    MODELS = load_models_list(DOI)
except Exception as e:
    print(f"Could not load models list: {str(e)}. Setting to empty.")
    MODELS = []


def refresh_models_list(models_home=None):
    global DOI
    global MODELS
    DOI = get_latest_zenodo_doi(PERMANENT_DOI)
    models_home = get_models_home(models_home)
    if exists(Path(models_home, "models.yaml")):
        Path(models_home, "models.yaml").unlink()
    models = MODELS
    try:
        models = load_models_list(DOI, models_home)
        MODELS = models
    except Exception as e:
        raise ValueError(f"Could not load models list: {str(e)}")
    return models


def get_model(
    models_home=None,
    model_name=None,
    filters=[],
    download_if_missing=True,
):
    global DOI
    global MODELS

    models_home = get_models_home(models_home)

    if DOI in [None, ""]:
        DOI = get_latest_zenodo_doi(PERMANENT_DOI)
    try:
        MODELS = load_models_list(DOI, models_home)
    except Exception as e:
        raise ValueError(f"Could not load models list: {str(e)}")

    print(f"Using models list found in {models_home}")

    base_url = f"https://zenodo.org/record/{DOI}/files"
    if model_name is None:
        raise ValueError("model_name must be specified, got None")
    if model_name not in MODELS:
        print(f"{model_name} not on Zenodo, trying local files")
        # raise ValueError("model_name must be one of %s, got %s" % (MODELS.keys(), model_name))
        return (
            [],
            None,
        )  # TODO: upload all the models on zenodo so we can throw an error here instead of returning an empty list
    model_info = MODELS[model_name]

    if not exists(models_home):
        makedirs(models_home)
    if not exists(Path(models_home, model_name)):
        makedirs(Path(models_home, model_name))
    if not exists(Path(models_home, model_name, "filters")):
        makedirs(Path(models_home, model_name, "filters"))

    filter_synonyms = [filt.replace("_", ":") for filt in model_info["filters"]]

    all_filters = list(set(model_info["filters"] + filter_synonyms + custom_filters))
    if filters in [[], None, ""] and "filters" in model_info:
        filters = model_info["filters"]

    missing_filters = list(set(filters).difference(set(all_filters)))
    if len(missing_filters) > 0:
        raise ValueError(
            f'Zenodo does not have filters {",".join(missing_filters)} for {model_name}'
        )

    core_format = "pkl"
    filter_format = "pkl"
    if "_tf" in model_name:
        filter_format = "h5"

    filepaths = [Path(models_home, f"{model_name}.{core_format}")] + [
        Path(models_home, model_name, f"{f}.{filter_format}") for f in filters
    ]
    urls = [f"{base_url}/{model_name}.{core_format}?download=1"] + [
        f"{base_url}/{model_name}_{f}.{filter_format}?download=1" for f in filters
    ]

    missing = [(u, f) for u, f in zip(urls, filepaths) if not f.exists()]
    if len(missing) > 0:
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")

        print(f"downloading {len(missing)} files for model {model_name}:")
        with ThreadPoolExecutor(
            max_workers=min(len(missing), max(cpu_count(), 8))
        ) as executor:
            executor.map(download, missing)

    # return the paths to the files and corresponding filters
    return [str(f) for f in filepaths], filters


def get_parser():

    parser = argparse.ArgumentParser(
        description="Download SVD models from Zenodo",
    )
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
        help="Refresh the list of models available on Zenodo",
    )

    return parser


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    refresh = False
    try:
        refresh = args.refresh_model_list
    except AttributeError:
        pass
    if refresh:
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None
        )

    filters = []
    if args.filters not in [None, ""]:
        try:
            filters = args.filters.split(",")
        except AttributeError:
            pass

    if args.model in [None, ""]:
        raise ValueError("a model must be specified with --model")

    get_model(
        models_home=args.svd_path if args.svd_path not in [None, ""] else None,
        model_name=args.model,
        filters=filters,
    )

    # example:
    # python nmma/utils/models.py --model="Bu2019lm" --filters=ztfr,ztfg,ztfi --svd-path='./svdmodels'
