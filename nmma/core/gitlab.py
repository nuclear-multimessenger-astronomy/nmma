from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

try:
    from yaml import CLoader as Loader
    from yaml import load
except ImportError:
    from yaml import Loader, load
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import requests
from tqdm.auto import tqdm

pbar = {}
MODELS = {}
REPO = "https://gitlab.com/Theodlz/nmma-models"
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


def get_models_home(models_home=None) -> Path:
    """Resolve (and create) the local SVD-model cache directory: the
    given path, else ``$NMMA_MODELS``, else ``DEFAULT_MODELS_HOME`` (a
    ``nmma_models/`` directory next to the repo).

    Parameters
    ----------
    models_home: str, optional

    Returns
    -------
    Path
        The resolved, expanded, existing directory path.
    """
    if not models_home:
        models_home = os.environ.get("NMMA_MODELS", DEFAULT_MODELS_HOME)
    models_home = Path(models_home).expanduser()
    models_home.mkdir(parents=True, exist_ok=True)
    return models_home


def clear_data_home(models_home=None):
    """Delete the SVD-model cache directory.

    Parameters
    ----------
    models_home: str, optional
    """
    models_home = get_models_home(models_home)
    shutil.rmtree(models_home)


def download(file_info):
    """Stream one file to disk with a progress bar.

    Parameters
    ----------
    file_info: tuple of (str, str)
        (url, filepath) to download to.

    Returns
    -------
    str
        ``filepath``, once written.
    """
    url, filepath = file_info
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    chunk_size = 4096
    downloaded = 0

    with (
        open(filepath, "wb") as f,
        tqdm(
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            desc=filepath.name,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            pbar.update(len(chunk))

    if downloaded != total:
        raise ValueError(
            f"Downloaded file {filepath} is incomplete. "
            f"Only {downloaded} of {total} bytes were downloaded."
        )

    return filepath


def decompress(file_path):
    """Decompress a .lzma file in place via the system lzma CLI (removes
    the .lzma file, creates the decompressed one alongside it, e.g.
    model.joblib.lzma -> model.joblib). Tolerates a "File exists" error
    (already decompressed from a previous run) rather than raising.

    Parameters
    ----------
    file_path: Path
        Path to the .lzma file. Must exist and end in ``.lzma``.

    Returns
    -------
    Path
        ``file_path`` (the original, still-.lzma-suffixed path) -- not
        the path to the decompressed file.
    """
    if not file_path.suffix == ".lzma":
        raise ValueError(f"File {file_path} is not a .lzma file")
    if not file_path.exists():
        raise ValueError(f"File {file_path} does not exist")

    stdout, stderr = subprocess.Popen(
        ["lzma", "-d", file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).communicate()
    if stderr.decode("utf-8") != "" and "File exists" not in stderr.decode("utf-8"):
        raise RuntimeError(f"Error decompressing {file_path}: {stderr}")
    return file_path


def download_and_decompress(file_info):
    """Download one model file and decompress it, if it's a .lzma file.

    Parameters
    ----------
    file_info: tuple of (str, str)
        (url, filepath) passed to ``download``.
    """
    file_path = download(file_info)
    if file_path.suffix == ".lzma":
        decompress(file_path)


def download_models_list(models_home=None):
    """Fetch the models catalog (models.yaml) from GitLab into
    ``models_home``, overwriting any existing copy.

    Parameters
    ----------
    models_home: str, optional
    """
    # first we load the models list from gitlab
    models_home = get_models_home(models_home)
    models_home.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"{REPO}/raw/main/models.yaml", allow_redirects=True)
    with open(models_home / "models.yaml", "wb") as f:
        f.write(r.content)


def load_models_list(models_home=None):
    """Load the models catalog, downloading it first if not already
    cached; on any failure (e.g. no network), falls back to inferring
    models and their filters from ``models_home``'s directory structure
    (one subdirectory per model, one file per filter).

    Parameters
    ----------
    models_home: str, optional

    Returns
    -------
    models: dict
        {model_name: {"filters": [...], ...}, ...}
    used_local: bool
        True if it fell back to the local directory listing.
    """
    models_home = get_models_home(models_home)
    models_file = models_home / "models.yaml"
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

    files = [f for f in models_home.glob("*") if f.is_dir()]
    files = [f.stem for f in files]

    for f in files:
        name = f.split("/")[-1]
        filters = []
        if (models_home / name).exists():
            filter_files = [
                f.stem for f in (models_home / name).glob("*") if f.is_file()
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
    """Discard the cached models.yaml and reload the catalog from
    GitLab, updating the module-level ``MODELS`` cache.

    Parameters
    ----------
    models_home: str, optional

    Returns
    -------
    dict
        The freshly loaded models catalog.
    """
    global MODELS
    models_home = get_models_home(models_home)
    if (models_home / "models.yaml").exists():
        (models_home / "models.yaml").unlink()
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
    """Resolve local file paths for a model's core file and/or filter
    files, downloading (and decompressing) whichever are missing.

    Defaults to all of the model's filters if none are given, and
    drops any of ``SKIP_FILTERS`` (X-ray/radio) from what's downloaded
    while still reporting them back as available. Under MPI, only rank
    0 downloads; other ranks wait at a barrier.

    Parameters
    ----------
    models_home: str, optional
        See ``get_models_home``.
    model_name: str
        Must be a key in the models catalog (see ``load_models_list``).
    filters: list of str, optional
        Which filters to fetch; defaults to all of the model's filters.
    download_if_missing: bool, default True
        If False, raise instead of downloading anything missing.
    filters_only: bool, default False
        If True, skip the model's core file, only fetch filter files.

    Returns
    -------
    filepaths: list of str
        Local paths, core file first (unless ``filters_only``) then one
        per requested filter.
    filters: list of str
        The filters actually included (requested filters plus any
        skipped ones from ``SKIP_FILTERS``).
    """
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

    (models_home / model_name).mkdir(parents=True, exist_ok=True)

    # FIX ME: filter_synonyms lets a caller pass either the underscore
    # form (the real, on-disk/on-GitLab filename) or its colon
    # substitute past the "is this a known filter" check below, but
    # nothing normalizes a colon-form request back to the underscore
    # form before `filepaths`/`urls` are built further down -- so
    # requesting e.g. "atlas:c" builds a path/URL for a file that
    # doesn't exist ("atlas:c.joblib" instead of the real
    # "atlas_c.joblib"). Confirmed directly. The one real caller
    # (nmma/em/model.py's get_model_data) already translates colon-form
    # filters to underscore form itself before calling get_model, so
    # this doesn't bite that path -- but a caller that passes
    # colon-form filters straight through would get a silently wrong
    # path. Needs the requested filters normalized to their
    # canonical (underscore) form here, not just validated.
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
                f"models list from GitLab does not have filters {','.join(missing_filters)} for {model_name}"
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
        [models_home / f"{core_model_name}.{core_format}"] if not filters_only else []
    ) + [models_home / model_name / f"{f}.{filter_format}" for f in filters]
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
                # Consume the iterator so URLError / OSError from worker threads
                # surfaces instead of being silently swallowed.
                list(executor.map(download_and_decompress, missing))
            still_missing = [f for _, f in missing if not Path(f).exists()]
            if still_missing:
                raise OSError(
                    f"failed to download {len(still_missing)} model file(s) for "
                    f"{model_name}: "
                    + ", ".join(still_missing[:3])
                    + (" ..." if len(still_missing) > 3 else "")
                )

        if comm:
            comm.Barrier()
    return [str(f) for f in filepaths], filters + skipped_filters


def get_parser():
    """Build the argparse parser for the svdmodel-download console
    script: --model, --svd-path, --filters, --refresh-models-list.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="Download SVD models from GitLab")
    parser.add_argument("--model", help="Name the model to be used")
    parser.add_argument(
        "--svd-path",
        help="Path to the SVD models directory. If not provided, will use the default path",
    )
    parser.add_argument(
        "--filters",
        help="A comma seperated list of filters to use (e.g. g,r,i). If none is provided, will use all the filters available",
    )
    parser.add_argument(
        "--refresh-models-list",
        action="store_true",
        help="Refresh the list of models available on Gitlab",
    )

    return parser


def main(args=None):
    """CLI entry point for svdmodel-download: optionally refresh the
    models catalog, then download a model's files via get_model.

    Parameters
    ----------
    args: argparse.Namespace, optional
        Parsed CLI args; if None, parsed from sys.argv via get_parser.

    Returns
    -------
    filepaths, filters
        See ``get_model``.
    """
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
