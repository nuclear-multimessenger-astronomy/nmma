from argparse import Namespace

import pytest

from nmma.em import em_likelihood


@pytest.fixture
def stubbed(monkeypatch):
    """Stub out everything in setup_em_kwargs that needs a real model/data."""
    monkeypatch.setattr(
        em_likelihood.model, "create_light_curve_model_from_args", lambda a, f: "model"
    )
    monkeypatch.setattr(em_likelihood, "read_trigger_time", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        em_likelihood.utils, "setup_filtered_lc_data", lambda data, t: ([0.0], data)
    )
    monkeypatch.setattr(
        em_likelihood.utils, "check_model_time_consistency", lambda data, *a, **k: data
    )
    monkeypatch.setattr(
        em_likelihood.systematics, "FilterSystematicsHandler", lambda *a, **k: "sys"
    )


def em_kwargs(**args):
    namespace = Namespace(injection=None, allow_data_cuts=False, em_error_budget=1.0)
    for key, val in args.items():
        setattr(namespace, key, val)
    data_dump = dict(light_curve_data={}, filters=["ps1::g"], systematics_dict={})
    return em_likelihood.setup_em_kwargs({}, data_dump, namespace)


@pytest.mark.parametrize("limit", [24.0, {"ps1::g": 24.0}])
def test_detection_limit_is_forwarded(stubbed, limit):
    """--detection-limit is stored un-prefixed on args and must reach the likelihood."""
    assert em_kwargs(detection_limit=limit)["detection_limit"] == limit


def test_detection_limit_default_is_no_limit(stubbed):
    assert em_kwargs()["detection_limit"] == float("inf")


def test_verbose_is_forwarded(stubbed):
    assert em_kwargs(verbose=True)["verbose"] is True
