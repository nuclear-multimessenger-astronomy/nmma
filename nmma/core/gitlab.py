from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
try:
    from yaml import CLoader as Loader, load
except ImportError:
    from yaml import Loader, load
import argparse
import shutil
import os 
from pathlib import Path
import subprocess

import requests
from tqdm.auto import tqdm

pbar = {}
MODELS = {}
REPO = "https://gitlab.com/Theodlz/nmma-models"
# DEFAULT_MODELS_HOME = os.path.join("~", "nmma_models")
code_dir = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_HOME = code_dir.parent / "nmma_models"

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
    if not models_home:
        models_home = os.environ.get("NMMA_MODELS", DEFAULT_MODELS_HOME)
    models_home = os.path.expanduser(models_home)
    os.makedirs(models_home, exist_ok=True)
    return models_home


def clear_data_home(models_home=None):
    models_home = get_models_home(models_home)
    shutil.rmtree(models_home)


def download(file_info):
    url, filepath = file_info
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    chunk_size = 4096
    file_content = b""

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f, tqdm(
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"{str(filepath).split('/')[-1]}",
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    if len(file_content) != total:
        raise ValueError(
            f"Downloaded file {filepath} is incomplete. "
            f"Only {len(file_content)} of {total} bytes were downloaded."
        )

    return filepath


def decompress(file_path):
    if not file_path.endswith(".lzma"):
        raise ValueError(f"File {file_path} is not a .lzma file")
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")

    stdout, stderr = subprocess.Popen(
        ["lzma", "-d", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if stderr.decode("utf-8") != "" and "File exists" not in stderr.decode("utf-8"):
        raise RuntimeError(f"Error decompressing {file_path}: {stderr}")
    return file_path


def download_and_decompress(file_info):
    download(file_info)
    decompress(file_info[1])

def download_models_list(models_home=None):
    # first we load the models list from gitlab
    models_home = get_models_home(models_home)
    os.makedirs(models_home, exist_ok=True)
    r = requests.get(f"{REPO}/raw/main/models.yaml", allow_redirects=True)
    with open(Path(models_home, "models.yaml"), "wb") as f:
        f.write(r.content)

def load_models_list(models_home=None):

    models_home = get_models_home(models_home)
    models_file = Path(models_home, "models.yaml")
    models = {}
    
    try:
        if not models_file.exists():
            download_models_list(models_home=models_home)
        with models_file.open("r") as f:
            models = load(f, Loader=Loader)
        downloaded_if_missing = True
    except Exception as e:
        downloaded_if_missing = False
        print(f"Could not open downloaded models list, using local files instead: {e}")

    files = [f for f in Path(models_home).glob("*") if f.is_dir()]
    files = [f.stem for f in files]


    for f in files:
        name = f.split("/")[-1]
        filters = []
        if Path(models_home, name).exists():
            filter_files = [
                f.stem for f in Path(models_home, name).glob("*") if f.is_file()
            ]
            for ff in filter_files:
                ff = ff.split("/")[-1]

                if name in ff:
                    ff = ff.replace(name, "")              
                ff = ff.strip("_")
                if ff:
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
    if Path(models_home, "models.yaml").exists():
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

    os.makedirs(Path(models_home, model_name), exist_ok=True)

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

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    except Exception as e:
        print("MPI could not be initialized:", e)
        comm = None
        rank = 0

    missing = [(f"{u}", f"{f}") for u, f in zip(urls, filepaths) if not f.exists()]
    if len(missing) > 0:
        if not download_if_missing:
            raise OSError("Data not found and `download_if_missing` is False")

        
        if rank == 0:
            print(f"downloading {len(missing)} files for model {model_name}:")
            with ThreadPoolExecutor(
                max_workers=min(len(missing), max(cpu_count(), 8))
            ) as executor:
                executor.map(download_and_decompress, missing)

        if comm:
            comm.Barrier()
    return [str(f) for f in filepaths], filters + skipped_filters


def get_parser():

    parser = argparse.ArgumentParser(description="Download SVD models from GitLab")
    parser.add_argument("--model", help="Name the model to be used")
    parser.add_argument("--svd-path",
        help="Path to the SVD models directory. If not provided, will use the default path")
    parser.add_argument("--filters",
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available" )
    parser.add_argument("--refresh-models-list", action="store_true",
        help="Refresh the list of models available on Gitlab")

    return parser

def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    if args.refresh_models_list:
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
    
    return get_model(
        models_home=args.svd_path if args.svd_path not in [None, ""] else None,
        model_name=args.model,
        filters=filters,
        download_if_missing=True,
        filters_only=False,
    )


if __name__ == "__main__":
    main()