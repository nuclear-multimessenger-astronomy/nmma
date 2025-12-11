from .models_tools import SOURCES, get_models_home, get_parser  # noqa

try:
    from mpi4py import MPI

    mpi_enabled = True
except ImportError:
    mpi_enabled = False


def mpi_barrier(comm):
    if mpi_enabled:
        comm.Barrier()


def refresh_models_list(models_home=None, source=None):

    if source is None:
        source = SOURCES[0]
    if source not in ["gitlab"]:
        raise ValueError(f"source must be one of ['gitlab'], got {source}")

    sources_tried = []
    while True:
        try:
            if source == "gitlab":
                from .gitlab import refresh_models_list
            models = refresh_models_list(models_home=models_home)
            break
        except Exception as e:
            print(f"Error while refreshing models list from {source}: {str(e)}")
            sources_tried.append(source)
            remaining_sources = [s for s in SOURCES if s not in sources_tried]
            if len(remaining_sources) == 0:
                print("No more sources to try, exiting")
                raise e
            source = remaining_sources[0]
            print(f"Trying to refresh models list from {source} instead")

    return models


def get_model(
    models_home=None,
    model_name=None,
    filters=[],
    download_if_missing=True,
    filters_only=False,
    source=None,
):

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

    if source is None:
        source = SOURCES[0]
    if source not in ["gitlab"]:
        raise ValueError(f"source must be one of ['gitlab'], got {source}")

    sources_tried = []
    while True:
        try:
            if source == "gitlab":
                from .gitlab import get_model

            if rank == 0 or not MPI.Is_initialized():
                files, filters = get_model(
                    models_home=models_home,
                    model_name=model_name,
                    filters=filters,
                    download_if_missing=download_if_missing,
                    filters_only=filters_only,
                )
            mpi_barrier(comm)
            break
        except Exception as e:
            print(f"Error while getting model from {source}: {str(e)}")
            sources_tried.append(source)
            remaining_sources = [s for s in SOURCES if s not in sources_tried]
            if len(remaining_sources) == 0:
                print("No more sources to try, exiting")
                raise e
            source = remaining_sources[0]
            print(f"Trying to get model from {source} instead")

    return files, filters


def main(args=None):

    if args is None:
        parser = get_parser()
        args = parser.parse_args()

    if args.source is None:
        args.source = SOURCES[0]
    if args.source not in ["gitlab"]:
        raise ValueError(f"source must be one of ['gitlab'], got {args.source}")

    refresh = False
    try:
        refresh = args.refresh_model_list
    except AttributeError:
        pass
    if refresh:
        refresh_models_list(
            models_home=args.svd_path if args.svd_path not in [None, ""] else None,
            source=args.source,
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
        source=args.source,
    )


if __name__ == "__main__":
    main()
