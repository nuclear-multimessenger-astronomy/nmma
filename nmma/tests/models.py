import pytest

from ..core.gitlab import get_model, refresh_models_list

# GitLab SVD-models fetch path is retired alongside the rest of the
# SVD-test suite; surrogate equivalents land via huggingface_hub now.
pytestmark = pytest.mark.skip(reason="GitLab SVD download retired; see fiesta_smoke")


def test_download_model_gitlab():
    # Test that we can download a model from GitLab
    files, filters = get_model(model_name="LANLTP1_tf", filters=["ztfg"])
    assert len(files) == 2
    assert len(filters) == 1


def test_refresh_models_list_gitlab():
    # Test that we can refresh the models list from GitLab
    models = refresh_models_list()
    assert len(models) > 0
