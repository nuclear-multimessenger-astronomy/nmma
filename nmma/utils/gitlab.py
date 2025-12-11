from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from os.path import exists
from pathlib import Path
from os import makedirs
import requests
from requests.exceptions import ConnectionError
from yaml import load

from .models_tools import SKIP_FILTERS, download, decompress, get_models_home

REPO = "https://gitlab.com/Theodlz/nmma-models"

MODELS = {}

try:
    from mpi4py import MPI

    mpi_enabled = True
except ImportError:
    mpi_enabled = False


def download_and_decompress(file_info):
    download(file_info)
    decompress(file_info[1])


def mpi_barrier(comm):
    if mpi_enabled:
        comm.Barrier()


def download_models_list(models_home=None):
    # first we load the models list from gitlab
    models_home = get_models_home(models_home)
    if not exists(models_home):
        makedirs(models_home)
    r = requests.get(f"{REPO}/raw/main/models.yaml", allow_redirects=True)
    with open(Path(models_home, "models.yaml"), "wb") as f:
        f.write(r.content)


def load_models_list(models_home=None):

    models_home = get_models_home(models_home)

    downloaded_if_missing = True
    if not exists(Path(models_home, "models.yaml")):
        try:
            download_models_list(models_home=models_home)
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

    files = [f for f in Path(models_home).glob("*") if f.is_dir()]
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
    global MODELS
    models_home = get_models_home(models_home)
    if exists(Path(models_home, "models.yaml")):
        Path(models_home, "models.yaml").unlink()
    models = MODELS
    try:
        models = load_models_list(models_home)[0]
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
    global MODELS

    models_home = get_models_home(models_home)
    used_local = False
    try:
        MODELS, used_local = load_models_list(models_home)
    except Exception as e:
        raise ValueError(f"Could not load models list: {str(e)}")

    if used_local:
        print("Could not access GitLab, used local models list instead.")

    base_url = f"{REPO}/raw/main/models"
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
                f'models list from GitLab does not have filters {",".join(missing_filters)} for {model_name}'
            )

    core_format = "joblib"
    filter_format = "joblib"
    if "_tf" in model_name:
        filter_format = "h5"

    # Some models have underscores. Keep those, but drop '_tf' if it exists
    model_name_components = model_name.split("_")
    if "tf" in model_name_components:
        model_name_components.remove("tf")
    core_model_name = "_".join(model_name_components)

    filepaths = (
        [Path(models_home, f"{core_model_name}.{core_format}")]
        if not filters_only
        else []
    ) + [Path(models_home, model_name, f"{f}.{filter_format}") for f in filters]
    urls = (
        [f"{base_url}/{core_model_name}.{core_format}"] if not filters_only else []
    ) + [f"{base_url}/{model_name}/{f}.{filter_format}" for f in filters]

    comm = None
    if mpi_enabled:
        try:
            comm = MPI.COMM_WORLD
        except Exception as e:
            print("MPI could not be initialized:", e)
            comm = None

    rank = 0
    if comm:
        try:
            rank = comm.Get_rank()
        except Exception as e:
            print("Error getting MPI rank:", e)

    missing = [(f"{u}", f"{f}") for u, f in zip(urls, filepaths) if not f.exists()]
    if len(missing) > 0:
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")

        if rank == 0 or not comm:
            print(f"downloading {len(missing)} files for model {model_name}:")
            with ThreadPoolExecutor(
                max_workers=min(len(missing), max(cpu_count(), 8))
            ) as executor:
                executor.map(download_and_decompress, missing)
        mpi_barrier(comm)

    return [str(f) for f in filepaths], filters + skipped_filters
