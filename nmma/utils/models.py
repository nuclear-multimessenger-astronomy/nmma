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
    if not exists(Path(filepath).parent):
        try:
            makedirs(Path(filepath).parent)
        except Exception:
            pass
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
    if DOI in [None, ""]:
        try:
            DOI = get_latest_zenodo_doi(PERMANENT_DOI)
        except Exception:
            print(
                "Could not retrieve latest DOI, models won't be downloaded or updated. Will try using existing list or local files available."
            )
            pass

    if doi in [None, ""]:
        doi = DOI

    models_home = get_models_home(models_home)

    downloaded_if_missing = True
    if not exists(Path(models_home, "models.yaml")):
        try:
            if doi not in [None, ""]:
                download_models_list(doi=DOI, models_home=models_home)
            else:
                raise ConnectionError("No DOI provided")
        except ConnectionError:
            downloaded_if_missing = False
            pass

    models = {}

    if downloaded_if_missing:
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader
        try:
            with open(Path(models_home, "models.yaml"), "r") as f:
                models = load(f, Loader=Loader)
        except Exception as e:
            downloaded_if_missing = False
            print(
                f"Could not open the download models list, using local files instead: {str(e)}"
            )

    if not downloaded_if_missing:
        print("Attempting to retrieve local files...")

    files = [f for f in Path(models_home).glob("*") if f.is_file()]
    files = [f.stem for f in files]

    for f in files:
        name = f.split("/")[-1]
        filters = []
        if exists(Path(models_home, name)):
            filter_files = [
                f.stem for f in Path(models_home, name).glob("*") if f.is_file()
            ]
            for ff in filter_files:
                ff = ff.split("/")[-1]

                if name in ff:
                    ff = ff.replace(name, "")

                if ff.startswith("_"):
                    ff = ff[1:]

                if ff.endswith("_"):
                    ff = ff[:-1]

                if ff is not None and ff != "":
                    filters.append(ff)

        filters = list(set(filters))
        if name not in models:
            models[name] = {"filters": filters}
        elif "filters" not in models[name]:
            models[name]["filters"] = filters
        else:
            models[name]["filters"] = list(set(filters + models[name]["filters"]))

    return models, downloaded_if_missing is False


def refresh_models_list(models_home=None):
    global DOI
    global MODELS
    models_home = get_models_home(models_home)
    if exists(Path(models_home, "models.yaml")):
        Path(models_home, "models.yaml").unlink()
    models = MODELS
    try:
        models = load_models_list(DOI, models_home)[0]
        MODELS = models
    except Exception as e:
        raise ValueError(f"Could not load models list: {str(e)}")
    return models


def get_model(
    models_home=None,
    model_name=None,
    filters=[],
    download_if_missing=True,
    filters_only=False,
):
    global DOI
    global MODELS

    models_home = get_models_home(models_home)
    used_local = False
    try:
        MODELS, used_local = load_models_list(DOI, models_home)
    except Exception as e:
        raise ValueError(f"Could not load models list: {str(e)}")

    if used_local:
        print("Could not access Zenodo, used local models list instead.")

    base_url = f"https://zenodo.org/record/{DOI}/files"
    if model_name is None:
        raise ValueError("model_name must be specified, got None")
    if model_name not in MODELS:
        raise ValueError(f"model_name {model_name} not found in models list")
    model_info = MODELS[model_name]

    if not exists(models_home):
        makedirs(models_home)
    if not exists(Path(models_home, model_name)):
        makedirs(Path(models_home, model_name))

    filter_synonyms = [filt.replace("_", ":") for filt in model_info["filters"]]

    all_filters = list(set(model_info["filters"] + filter_synonyms))
    if filters in [[], None, ""] and "filters" in model_info:
        filters = model_info["filters"]

    skipped_filters = [f for f in filters if f in SKIP_FILTERS]

    # remove the skip_filters list from the filters
    filters = [f for f in filters if f not in SKIP_FILTERS]

    missing_filters = list(set(filters).difference(set(all_filters)))
    if len(missing_filters) > 0:
        if used_local:
            raise ValueError(
                f"local models list does not have filters {','.join(missing_filters)} for {model_name}"
            )
        else:
            raise ValueError(
                f'models list from zenodo does not have filters {",".join(missing_filters)} for {model_name}'
            )

    core_format = "pkl"
    filter_format = "pkl"
    if "_tf" in model_name:
        filter_format = "h5"

    filepaths = (
        [Path(models_home, f"{model_name}.{core_format}")] if not filters_only else []
    ) + [Path(models_home, model_name, f"{f}.{filter_format}") for f in filters]
    urls = (
        [f"{base_url}/{model_name}.{core_format}?download=1"]
        if not filters_only
        else []
    ) + [f"{base_url}/{model_name}_{f}.{filter_format}?download=1" for f in filters]

    missing = [(u, f) for u, f in zip(urls, filepaths) if not f.exists()]
    if len(missing) > 0:
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")
        if DOI in [None, ""]:
            print(
                f"Could not connect to ZENODO, and no local files found in `{models_home}`. Can't download missing models:"
            )
            for u, f in missing:
                print(f"    - {f}")
            raise OSError(
                "Data not found locally and no access to Zenodo. Can't download missing models."
            )

        print(f"downloading {len(missing)} files for model {model_name}:")
        with ThreadPoolExecutor(
            max_workers=min(len(missing), max(cpu_count(), 8))
        ) as executor:
            executor.map(download, missing)

    return [str(f) for f in filepaths], filters + skipped_filters


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


if __name__ == "__main__":
    main()
