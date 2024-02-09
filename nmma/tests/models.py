from ..utils.models import get_model, refresh_models_list


def test_download_model_gitlab():
    # Test that we can download a model from GitLab
    files, filters = get_model(
        model_name="LANLTP1_tf", filters=["ztfg"], source="gitlab"
    )
    assert len(files) == 2
    assert len(filters) == 1


def test_download_model_zenodo():
    # Test that we can download a model from Zenodo
    files, filters = get_model(
        model_name="Ka2017_tf", filters=["ztfg"], source="zenodo"
    )
    assert len(files) == 2
    assert len(filters) == 1


def test_refresh_models_list_gitlab():
    # Test that we can refresh the models list from GitLab
    models = refresh_models_list(source="gitlab")
    assert len(models) > 0


def test_refresh_models_list_zenodo():
    # Test that we can refresh the models list from Zenodo
    models = refresh_models_list(source="zenodo")
    assert len(models) > 0
