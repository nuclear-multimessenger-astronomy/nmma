from ..core.gitlab import get_model, refresh_models_list

def test_download_model_gitlab():
    # Test that we can download a model from GitLab
    files, filters = get_model(model_name="LANLTP1_tf", filters=["ztfg"])
    assert len(files) == 2
    assert len(filters) == 1


def test_refresh_models_list_gitlab():
    # Test that we can refresh the models list from GitLab
    models = refresh_models_list()
    assert len(models) > 0
