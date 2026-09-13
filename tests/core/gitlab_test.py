import os
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nmma.core import gitlab


class FakeResponse:
    """Stands in for a requests response, so nothing in this module ever
    reaches the network."""

    def __init__(self, content=b"", chunks=None):
        self.content = content
        self._chunks = chunks if chunks is not None else [content]
        self.headers = {"content-length": str(sum(len(c) for c in self._chunks))}

    def iter_content(self, chunk_size=4096):
        return iter(self._chunks)


class ModelsHomeMixin:
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.models_home = self.tmp_dir / "models"
        self.original_models = dict(gitlab.MODELS)
        self.original_env = os.environ.get("NMMA_MODELS")

    def teardown_method(self):
        gitlab.MODELS = self.original_models
        if self.original_env is None:
            os.environ.pop("NMMA_MODELS", None)
        else:
            os.environ["NMMA_MODELS"] = self.original_env
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def write_models_yaml(self, content="Ka2017:\n  filters:\n    - sdss_g\n    - sdss_r\n"):
        self.models_home.mkdir(parents=True, exist_ok=True)
        (self.models_home / "models.yaml").write_text(content)


class TestGetModelsHome(ModelsHomeMixin):
    def test_an_explicit_path_is_used_and_created(self):
        resolved = gitlab.get_models_home(self.models_home)
        assert resolved == self.models_home
        assert resolved.is_dir()

    def test_the_environment_variable_is_used_when_no_path_is_given(self):
        os.environ["NMMA_MODELS"] = str(self.models_home)
        assert gitlab.get_models_home() == self.models_home

    def test_an_explicit_path_beats_the_environment_variable(self):
        os.environ["NMMA_MODELS"] = str(self.tmp_dir / "from_env")
        other = self.tmp_dir / "explicit"
        assert gitlab.get_models_home(other) == other

    def test_a_user_path_is_expanded(self):
        os.environ["NMMA_MODELS"] = "~/nmma_models_that_should_not_exist"
        resolved = gitlab.get_models_home()
        assert "~" not in str(resolved)
        shutil.rmtree(resolved, ignore_errors=True)

    def test_a_string_path_is_accepted(self):
        assert gitlab.get_models_home(str(self.models_home)) == self.models_home

    def test_the_default_lives_next_to_the_package(self):
        assert gitlab.DEFAULT_MODELS_HOME.name == "nmma_models"


class TestClearDataHome(ModelsHomeMixin):
    def test_the_directory_is_removed(self):
        self.write_models_yaml()
        gitlab.clear_data_home(self.models_home)
        assert not self.models_home.exists()


class TestDownload(ModelsHomeMixin):
    def test_the_file_is_written_and_parent_directories_are_created(self):
        target = self.models_home / "Ka2017" / "sdss_g.joblib"
        with patch.object(
            gitlab.requests, "get", return_value=FakeResponse(b"model-bytes")
        ):
            returned = gitlab.download(("https://example.invalid/f", target))
        assert returned == target
        assert target.read_bytes() == b"model-bytes"

    def test_chunked_content_is_reassembled(self):
        target = self.models_home / "core.joblib"
        with patch.object(
            gitlab.requests, "get", return_value=FakeResponse(chunks=[b"aa", b"bb", b"cc"])
        ):
            gitlab.download(("https://example.invalid/f", target))
        assert target.read_bytes() == b"aabbcc"

    def test_a_truncated_download_is_rejected(self):
        # the advertised content-length and the bytes actually received must
        # agree, otherwise a half-downloaded surrogate would be used as if
        # it were complete
        response = FakeResponse(b"model-bytes")
        response.headers["content-length"] = "999"
        target = self.models_home / "core.joblib"
        with patch.object(gitlab.requests, "get", return_value=response):
            with pytest.raises(ValueError):
                gitlab.download(("https://example.invalid/f", target))

    def test_the_stream_flag_is_set(self):
        target = self.models_home / "core.joblib"
        with patch.object(
            gitlab.requests, "get", return_value=FakeResponse(b"x")
        ) as mock_get:
            gitlab.download(("https://example.invalid/f", target))
        assert mock_get.call_args.kwargs["stream"]


class TestDecompress(ModelsHomeMixin):
    def make_archive(self, name="core.joblib.lzma"):
        self.models_home.mkdir(parents=True, exist_ok=True)
        path = self.models_home / name
        path.write_bytes(b"compressed")
        return path

    def test_a_non_lzma_file_is_rejected(self):
        path = self.make_archive("core.joblib")
        with pytest.raises(ValueError):
            gitlab.decompress(path)

    def test_a_missing_file_is_rejected(self):
        with pytest.raises(ValueError):
            gitlab.decompress(self.models_home / "absent.lzma")

    def test_the_external_decompressor_is_invoked(self):
        path = self.make_archive()
        process = MagicMock()
        process.communicate.return_value = (b"", b"")
        with patch.object(gitlab.subprocess, "Popen", return_value=process) as mock_popen:
            returned = gitlab.decompress(path)
        assert returned == path
        assert mock_popen.call_args[0][0] == ["lzma", "-d", path]

    def test_a_decompression_error_is_raised(self):
        path = self.make_archive()
        process = MagicMock()
        process.communicate.return_value = (b"", b"lzma: corrupt input")
        with patch.object(gitlab.subprocess, "Popen", return_value=process):
            with pytest.raises(RuntimeError):
                gitlab.decompress(path)

    def test_an_already_decompressed_file_is_tolerated(self):
        # re-running a download must not fail just because the output exists
        path = self.make_archive()
        process = MagicMock()
        process.communicate.return_value = (b"", b"lzma: core.joblib: File exists")
        with patch.object(gitlab.subprocess, "Popen", return_value=process):
            assert gitlab.decompress(path) == path

    def test_download_and_decompress_only_decompresses_archives(self):
        plain = self.models_home / "core.joblib"
        with (
            patch.object(gitlab, "download", return_value=plain),
            patch.object(gitlab, "decompress") as mock_decompress,
        ):
            gitlab.download_and_decompress(("https://example.invalid/f", plain))
        mock_decompress.assert_not_called()

    def test_download_and_decompress_unpacks_an_archive(self):
        archive = self.models_home / "core.joblib.lzma"
        with (
            patch.object(gitlab, "download", return_value=archive),
            patch.object(gitlab, "decompress") as mock_decompress,
        ):
            gitlab.download_and_decompress(("https://example.invalid/f", archive))
        mock_decompress.assert_called_once_with(archive)


class TestModelsList(ModelsHomeMixin):
    def test_the_models_list_is_fetched_into_the_models_home(self):
        with patch.object(
            gitlab.requests, "get", return_value=FakeResponse(b"Ka2017:\n")
        ) as mock_get:
            gitlab.download_models_list(self.models_home)
        assert (self.models_home / "models.yaml").read_bytes() == b"Ka2017:\n"
        assert "models.yaml" in mock_get.call_args[0][0]

    def test_an_existing_list_is_not_downloaded_again(self):
        self.write_models_yaml()
        with patch.object(gitlab, "download_models_list") as mock_download:
            models, used_local = gitlab.load_models_list(self.models_home)
        mock_download.assert_not_called()
        assert "Ka2017" in models
        assert not used_local

    def test_a_missing_list_is_downloaded(self):
        self.models_home.mkdir(parents=True, exist_ok=True)

        def fake_download(models_home=None):
            (Path(models_home) / "models.yaml").write_text("Ka2017:\n  filters: []\n")

        with patch.object(gitlab, "download_models_list", side_effect=fake_download):
            models, used_local = gitlab.load_models_list(self.models_home)
        assert "Ka2017" in models
        assert not used_local

    def test_a_failed_download_falls_back_to_the_local_files(self):
        self.models_home.mkdir(parents=True, exist_ok=True)
        (self.models_home / "Bu2019lm").mkdir()
        (self.models_home / "Bu2019lm" / "sdss_g.joblib").touch()
        with patch.object(
            gitlab, "download_models_list", side_effect=OSError("no network")
        ):
            models, used_local = gitlab.load_models_list(self.models_home)
        assert used_local
        assert "Bu2019lm" in models
        assert models["Bu2019lm"]["filters"] == ["sdss_g"]

    def test_locally_present_filters_are_added_to_the_downloaded_list(self):
        self.write_models_yaml("Ka2017:\n  filters:\n    - sdss_g\n")
        (self.models_home / "Ka2017").mkdir()
        (self.models_home / "Ka2017" / "sdss_r.joblib").touch()
        models, _ = gitlab.load_models_list(self.models_home)
        assert set(models["Ka2017"]["filters"]) == {"sdss_g", "sdss_r"}

    def test_the_model_name_prefix_is_stripped_from_local_filter_files(self):
        self.write_models_yaml("Ka2017: {}\n")
        (self.models_home / "Ka2017").mkdir()
        (self.models_home / "Ka2017" / "Ka2017_sdss_g.joblib").touch()
        models, _ = gitlab.load_models_list(self.models_home)
        assert models["Ka2017"]["filters"] == ["sdss_g"]

    def test_refresh_drops_the_cached_list_and_reloads_it(self):
        self.write_models_yaml()

        def fake_download(models_home=None):
            (Path(models_home) / "models.yaml").write_text("Bu2019lm:\n  filters: []\n")

        with patch.object(gitlab, "download_models_list", side_effect=fake_download):
            models = gitlab.refresh_models_list(self.models_home)
        assert "Bu2019lm" in models
        assert "Ka2017" not in models

    def test_refresh_updates_the_module_level_cache(self):
        self.write_models_yaml()

        def fake_download(models_home=None):
            (Path(models_home) / "models.yaml").write_text("Bu2019lm:\n  filters: []\n")

        with patch.object(gitlab, "download_models_list", side_effect=fake_download):
            gitlab.refresh_models_list(self.models_home)
        assert "Bu2019lm" in gitlab.MODELS

    def test_a_refresh_failure_is_reported(self):
        self.write_models_yaml()
        with patch.object(gitlab, "load_models_list", side_effect=OSError("boom")):
            with pytest.raises(ValueError):
                gitlab.refresh_models_list(self.models_home)


class TestGetModel(ModelsHomeMixin):
    def setup_method(self):
        super().setup_method()
        self.write_models_yaml(
            "Ka2017:\n  filters:\n    - sdss_g\n    - sdss_r\n"
            "Ka2017_tf:\n  filters:\n    - sdss_g\n"
            "Bu2019lm:\n  filters:\n    - sdss_g\n    - X-ray-1keV\n"
        )

    def call_get_model(self, **kwargs):
        """get_model downloads whatever is missing, so the download step is
        replaced by one that just creates the files it was asked for."""
        created = []

        def fake_download(file_info):
            _, filepath = file_info
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            created.append(path)

        with patch.object(
            gitlab, "download_and_decompress", side_effect=fake_download
        ) as mock_download:
            filepaths, filters = gitlab.get_model(models_home=self.models_home, **kwargs)
        return filepaths, filters, created, mock_download

    def test_a_missing_model_name_is_rejected(self):
        with pytest.raises(ValueError):
            gitlab.get_model(models_home=self.models_home, model_name=None)

    def test_an_unknown_model_name_is_rejected(self):
        with pytest.raises(ValueError):
            gitlab.get_model(models_home=self.models_home, model_name="NotAModel")

    def test_the_core_model_and_every_filter_are_returned(self):
        filepaths, filters, _, _ = self.call_get_model(model_name="Ka2017")
        assert filters == ["sdss_g", "sdss_r"]
        assert len(filepaths) == 3
        assert filepaths[0].endswith("Ka2017.joblib")

    def test_filters_only_leaves_out_the_core_model(self):
        filepaths, _, _, _ = self.call_get_model(model_name="Ka2017", filters_only=True)
        assert len(filepaths) == 2
        assert not any(p.endswith("Ka2017.joblib") for p in filepaths)

    def test_explicit_filters_are_honoured(self):
        filepaths, filters, _, _ = self.call_get_model(
            model_name="Ka2017", filters=["sdss_g"]
        )
        assert filters == ["sdss_g"]
        assert len(filepaths) == 2

    def test_colon_separated_filter_synonyms_are_accepted(self):
        _, filters, _, _ = self.call_get_model(
            model_name="Ka2017", filters=["sdss:g"]
        )
        assert filters == ["sdss:g"]

    def test_an_unavailable_filter_is_rejected(self):
        with pytest.raises(ValueError):
            self.call_get_model(model_name="Ka2017", filters=["not_a_filter"])

    def test_x_ray_and_radio_filters_are_not_downloaded_but_still_reported(self):
        # these carry no surrogate data of their own
        filepaths, filters, _, _ = self.call_get_model(
            model_name="Bu2019lm", filters=["sdss_g", "X-ray-1keV"]
        )
        assert filters == ["sdss_g", "X-ray-1keV"]
        assert not any("X-ray" in p for p in filepaths)

    def test_a_tensorflow_model_uses_h5_filter_files(self):
        filepaths, _, _, _ = self.call_get_model(model_name="Ka2017_tf")
        assert any(p.endswith("sdss_g.h5") for p in filepaths)

    def test_a_tensorflow_model_shares_the_plain_core_model(self):
        # the "_tf" suffix is dropped for the core file, which is shared
        filepaths, _, _, _ = self.call_get_model(model_name="Ka2017_tf")
        assert filepaths[0].endswith("Ka2017.joblib")

    def test_files_that_are_already_present_are_not_downloaded_again(self):
        self.call_get_model(model_name="Ka2017")
        _, _, _, mock_download = self.call_get_model(model_name="Ka2017")
        mock_download.assert_not_called()

    def test_downloading_can_be_switched_off(self):
        with pytest.raises(OSError):
            gitlab.get_model(
                models_home=self.models_home,
                model_name="Ka2017",
                download_if_missing=False,
            )

    def test_a_download_that_produces_no_file_is_reported(self):
        # a silent failure here would leave the pipeline loading a model that
        # was never written
        with patch.object(gitlab, "download_and_decompress", return_value=None):
            with pytest.raises(OSError):
                gitlab.get_model(models_home=self.models_home, model_name="Ka2017")

    def test_the_local_models_list_is_used_when_gitlab_is_unreachable(self):
        (self.models_home / "models.yaml").unlink()
        (self.models_home / "Ka2017").mkdir()
        (self.models_home / "Ka2017" / "sdss_g.joblib").touch()
        (self.models_home / "Ka2017.joblib").touch()
        with patch.object(
            gitlab, "download_models_list", side_effect=OSError("no network")
        ):
            _, filters = gitlab.get_model(
                models_home=self.models_home, model_name="Ka2017"
            )
        assert filters == ["sdss_g"]

    def test_a_filter_missing_from_the_local_list_is_reported_as_such(self):
        (self.models_home / "models.yaml").unlink()
        (self.models_home / "Ka2017").mkdir()
        (self.models_home / "Ka2017" / "sdss_g.joblib").touch()
        with patch.object(
            gitlab, "download_models_list", side_effect=OSError("no network")
        ):
            with pytest.raises(ValueError) as raised:
                gitlab.get_model(
                    models_home=self.models_home,
                    model_name="Ka2017",
                    filters=["sdss_r"],
                )
        assert "local models list" in str(raised.value)


class TestCommandLineInterface(ModelsHomeMixin):
    def test_the_parser_exposes_the_documented_flags(self):
        args = gitlab.get_parser().parse_args([])
        assert args.model is None
        assert args.svd_path is None
        assert args.filters is None
        assert not args.refresh_models_list

    def test_arguments_are_parsed(self):
        args = gitlab.get_parser().parse_args(
            ["--model", "Ka2017", "--filters", "sdss_g,sdss_r", "--refresh-models-list"]
        )
        assert args.model == "Ka2017"
        assert args.filters == "sdss_g,sdss_r"
        assert args.refresh_models_list

    def test_main_requires_a_model(self):
        args = Namespace(
            model=None, svd_path=None, filters=None, refresh_models_list=False
        )
        with pytest.raises(ValueError):
            gitlab.main(args)

    def test_main_splits_the_filter_list_and_forwards_the_request(self):
        args = Namespace(
            model="Ka2017",
            svd_path=str(self.models_home),
            filters="sdss_g,sdss_r",
            refresh_models_list=False,
        )
        with patch.object(gitlab, "get_model", return_value=([], [])) as mock_get_model:
            gitlab.main(args)
        kwargs = mock_get_model.call_args.kwargs
        assert kwargs["filters"] == ["sdss_g", "sdss_r"]
        assert kwargs["model_name"] == "Ka2017"
        assert kwargs["models_home"] == str(self.models_home)

    def test_an_empty_svd_path_falls_back_to_the_default(self):
        args = Namespace(
            model="Ka2017", svd_path="", filters=None, refresh_models_list=False
        )
        with patch.object(gitlab, "get_model", return_value=([], [])) as mock_get_model:
            gitlab.main(args)
        assert mock_get_model.call_args.kwargs["models_home"] is None

    def test_main_can_refresh_the_models_list_first(self):
        args = Namespace(
            model="Ka2017",
            svd_path=str(self.models_home),
            filters=None,
            refresh_models_list=True,
        )
        with (
            patch.object(gitlab, "refresh_models_list") as mock_refresh,
            patch.object(gitlab, "get_model", return_value=([], [])),
        ):
            gitlab.main(args)
        mock_refresh.assert_called_once_with(models_home=str(self.models_home))

