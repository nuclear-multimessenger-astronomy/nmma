"""Unit tests for the parallel-bilby sampling layer.

Everything MPI-specific (pool creation, checkpoint writing, the dynesty run
loop itself) needs a live MPI communicator and a real sampler, so these tests
cover the parts that can be exercised in a single process: the timing
decorator, the sampler-kwargs translation, the initial-point rejection loop,
metadata handling, and the module-level worker indirection the pool relies on.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from nmma.core import mpi_setup
except ImportError as error:  # pragma: no cover - depends on the bilby version
    pytest.skip(
        f"nmma.core.mpi_setup could not be imported ({error}); the installed "
        "bilby does not provide what the parallel-bilby layer expects.",
        allow_module_level=True,
    )


class TestTimeStorage:
    def test_the_wrapped_return_value_is_passed_through(self):
        @mpi_setup.time_storage
        def sample(value):
            return value * 2

        assert sample(21) == 42

    def test_the_wrapped_function_keeps_its_identity(self):
        @mpi_setup.time_storage
        def sample():
            """A docstring."""

        assert sample.__name__ == "sample"
        assert sample.__doc__ == "A docstring."

    def test_keyword_arguments_are_forwarded(self):
        @mpi_setup.time_storage
        def sample(first, second=0):
            return first + second

        assert sample(1, second=2) == 3

    def test_the_duration_is_logged_under_the_function_name(self):
        @mpi_setup.time_storage
        def sample():
            return None

        with patch.object(mpi_setup.logger, "info") as mock_info:
            sample()
        assert "sample" in mock_info.call_args[0][0]


class TestGetInitialPointFromPrior:
    """The rejection loop is plain Python on top of prior_transform,
    log_prior and log_likelihood, so it is exercised on a bare instance
    rather than a fully constructed sampler."""

    def make_worker(self, log_likelihood_values, log_prior_values=None):
        worker = object.__new__(mpi_setup.Worker)
        # ndim is a read-only property derived from the search parameter keys
        worker._search_parameter_keys = ["x", "y"]
        worker.prior_transform = lambda unit: unit * 2.0
        likelihoods = iter(log_likelihood_values)
        priors = iter(log_prior_values if log_prior_values is not None else [])
        worker.log_likelihood = lambda theta: next(likelihoods)
        worker.log_prior = (
            (lambda theta: next(priors)) if log_prior_values is not None else (lambda theta: -1.0)
        )
        return worker

    def test_a_good_draw_is_returned(self):
        worker = self.make_worker([-3.0])
        rng = np.random.default_rng(0)
        unit, theta, log_likelihood = worker.get_initial_point_from_prior(rng)
        assert len(unit) == 2
        np.testing.assert_allclose(theta, unit * 2.0)
        assert log_likelihood == -3.0

    def test_draws_with_a_non_finite_likelihood_are_rejected(self):
        for bad_value in [np.inf, -np.inf, np.nan, np.nan_to_num(-np.inf)]:
            worker = self.make_worker([bad_value, -3.0])
            _, _, log_likelihood = worker.get_initial_point_from_prior(
                np.random.default_rng(0)
            )
            assert log_likelihood == -3.0, str(bad_value)

    def test_draws_outside_the_prior_support_are_rejected(self):
        worker = self.make_worker([-3.0], log_prior_values=[-np.inf, -1.0])
        _, _, log_likelihood = worker.get_initial_point_from_prior(
            np.random.default_rng(0)
        )
        assert log_likelihood == -3.0

    def test_a_rejected_prior_draw_never_reaches_the_likelihood(self):
        # evaluating the likelihood is the expensive step, so the prior has
        # to be checked first
        worker = self.make_worker([-3.0], log_prior_values=[-np.inf, -1.0])
        worker.get_initial_point_from_prior(np.random.default_rng(0))
        with pytest.raises(StopIteration):
            worker.log_likelihood(None)

    def test_the_returned_unit_cube_point_maps_onto_the_returned_sample(self):
        worker = self.make_worker([-3.0])
        unit, theta, _ = worker.get_initial_point_from_prior(np.random.default_rng(7))
        assert np.all((unit >= 0.0) & (unit <= 1.0))
        np.testing.assert_allclose(theta, worker.prior_transform(unit))


class TestWorkerLogLikelihood:
    def test_the_search_parameters_are_zipped_onto_the_sample(self):
        worker = object.__new__(mpi_setup.Worker)
        worker.parameters = {"fixed": 1.0}
        worker._search_parameter_keys = ["x", "y"]
        worker.likelihood = MagicMock()
        worker.likelihood.log_likelihood_ratio.return_value = -2.0

        value = worker.log_likelihood(np.array([0.1, 0.2]))

        assert value == -2.0
        seen = worker.likelihood.log_likelihood_ratio.call_args[0][0]
        assert seen == {"fixed": 1.0, "x": 0.1, "y": 0.2}

    def test_the_stored_parameters_are_not_modified(self):
        worker = object.__new__(mpi_setup.Worker)
        worker.parameters = {"fixed": 1.0}
        worker._search_parameter_keys = ["x"]
        worker.likelihood = MagicMock()
        worker.likelihood.log_likelihood_ratio.return_value = -2.0

        worker.log_likelihood(np.array([0.1]))

        assert worker.parameters == {"fixed": 1.0}

    def test_it_samples_in_likelihood_ratio_space(self):
        # the noise log-likelihood is added back only when the result is
        # assembled, so sampling must use the ratio
        worker = object.__new__(mpi_setup.Worker)
        worker.parameters = {}
        worker._search_parameter_keys = ["x"]
        worker.likelihood = MagicMock()
        worker.log_likelihood(np.array([0.1]))
        worker.likelihood.log_likelihood.assert_not_called()
        worker.likelihood.log_likelihood_ratio.assert_called_once()


class TestInitSamplerKwargs:
    """Translates the parsed sampler settings into dynesty's own kwargs,
    including the bilby-implemented walk samplers."""

    def make_sampler(self, boundaries=None):
        sampler = object.__new__(mpi_setup.Dynesty)
        keys = ["x", "y"]
        sampler._search_parameter_keys = keys
        boundaries = boundaries or {}
        sampler.priors = {
            key: MagicMock(boundary=boundaries.get(key)) for key in keys
        }
        return sampler

    def base_kwargs(self, **overrides):
        kwargs = dict(sample="unif", bound="multi", walks=100, nlive=500)
        kwargs.update(overrides)
        return kwargs

    def test_the_dimensionality_is_taken_from_the_search_keys(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(self.base_kwargs(), 2, 60, 5000)
        assert sampler.init_sampler_kwargs["ndim"] == 2

    def test_no_boundaries_give_none_rather_than_empty_lists(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(self.base_kwargs(), 2, 60, 5000)
        assert sampler.init_sampler_kwargs["periodic"] is None
        assert sampler.init_sampler_kwargs["reflective"] is None

    def test_periodic_and_reflective_parameters_are_indexed(self):
        sampler = self.make_sampler({"x": "periodic", "y": "reflective"})
        sampler._init_sampler_kwargs(self.base_kwargs(), 2, 60, 5000)
        assert sampler.init_sampler_kwargs["periodic"] == [0]
        assert sampler.init_sampler_kwargs["reflective"] == [1]

    def test_act_walk_is_replaced_by_the_bilby_sampler(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(self.base_kwargs(sample="act-walk"), 2, 60, 5000)
        assert isinstance(sampler.init_sampler_kwargs["sample"], mpi_setup.dy_utils.ACTTrackingEnsembleWalk)
        assert sampler.init_sampler_kwargs["bound"] == "none"

    def test_acceptance_walk_is_replaced_by_the_bilby_sampler(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(
            self.base_kwargs(sample="acceptance-walk"), 2, 60, 5000
        )
        assert isinstance(sampler.init_sampler_kwargs["sample"], mpi_setup.dy_utils.EnsembleWalkSampler)
        assert sampler.init_sampler_kwargs["bound"] == "none"

    def test_rwalk_is_replaced_by_the_bilby_sampler(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(self.base_kwargs(sample="rwalk"), 2, 60, 5000)
        assert isinstance(sampler.init_sampler_kwargs["sample"], mpi_setup.dy_utils.AcceptanceTrackingRWalk)

    def test_the_acceptance_target_is_passed_on(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(
            self.base_kwargs(sample="acceptance-walk"), 2, 17, 5000
        )
        assert sampler.init_sampler_kwargs["sample"].naccept == 17

    def test_the_maximum_chain_length_is_passed_on(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(
            self.base_kwargs(sample="acceptance-walk"), 2, 60, 1234
        )
        assert sampler.init_sampler_kwargs["sample"].maxmcmc == 1234

    def test_a_live_point_bound_is_replaced_for_a_plain_dynesty_sampler(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(
            self.base_kwargs(sample="unif", bound="live"), 2, 60, 5000
        )
        assert sampler.init_sampler_kwargs["bound"] == "multi"
        assert sampler.init_sampler_kwargs["sample"] == "unif"

    def test_an_unrecognised_sampler_and_bound_are_left_alone(self):
        sampler = self.make_sampler()
        sampler._init_sampler_kwargs(
            self.base_kwargs(sample="slice", bound="multi"), 2, 60, 5000
        )
        assert sampler.init_sampler_kwargs["sample"] == "slice"
        assert sampler.init_sampler_kwargs["bound"] == "multi"


class TestFloatifyDict:
    def setup_method(self):
        self.sampler = object.__new__(mpi_setup.Dynesty)

    def test_numpy_floats_become_python_floats(self):
        result = self.sampler.floatify_dict({"dlogz": np.float64(0.1)})
        assert isinstance(result["dlogz"], float)
        assert not isinstance(result["dlogz"], np.floating)

    def test_nested_dictionaries_are_converted_too(self):
        result = self.sampler.floatify_dict(
            {"first_update": {"min_eff": np.float64(10.0)}}
        )
        assert isinstance(result["first_update"]["min_eff"], float)

    def test_other_values_are_left_untouched(self):
        values = {"label": "run", "nlive": 500, "walks": None}
        assert self.sampler.floatify_dict(dict(values)) == values

    def test_the_dictionary_is_converted_in_place(self):
        values = {"dlogz": np.float64(0.1)}
        assert self.sampler.floatify_dict(values) is values


class TestStorableMetadata:
    def make_sampler(self):
        from argparse import Namespace

        sampler = object.__new__(mpi_setup.Dynesty)
        sampler.meta_data = {"existing": np.float64(1.0)}
        sampler.args = Namespace(label="run", nlive=500)
        sampler.likelihood = MagicMock()
        sampler.likelihood.meta_data = {"model": "Bu2019lm"}
        sampler.init_sampler_kwargs = {"nlive": 500}
        sampler.sampler_kwargs = {"dlogz": np.float64(0.1)}
        return sampler

    def test_the_arguments_are_stored_as_a_plain_dictionary(self):
        metadata = self.make_sampler().storable_metadata()
        assert metadata["args"] == {"label": "run", "nlive": 500}

    def test_the_likelihood_metadata_is_carried_along(self):
        metadata = self.make_sampler().storable_metadata()
        assert metadata["likelihood"] == {"model": "Bu2019lm"}

    def test_both_sets_of_sampler_kwargs_are_recorded(self):
        metadata = self.make_sampler().storable_metadata()
        assert metadata["sampler_kwargs"] == {"nlive": 500}
        assert metadata["run_sampler_kwargs"] == {"dlogz": 0.1}

    def test_numpy_floats_are_converted_so_the_result_can_be_serialised(self):
        metadata = self.make_sampler().storable_metadata()
        assert isinstance(metadata["existing"], float)
        assert isinstance(metadata["run_sampler_kwargs"]["dlogz"], float)


class TestPooledWorkerFunctions:
    """The pool calls these module-level functions, which forward to the
    worker held in the module's global scope so that surrogate models stay
    loaded between evaluations."""

    def setup_method(self):
        self.original_worker = getattr(mpi_setup, "worker", None)
        self.worker = MagicMock()
        mpi_setup.worker = self.worker

    def teardown_method(self):
        if self.original_worker is None:
            if hasattr(mpi_setup, "worker"):
                del mpi_setup.worker
        else:
            mpi_setup.worker = self.original_worker

    def test_the_likelihood_call_is_forwarded(self):
        self.worker.log_likelihood.return_value = -2.0
        assert mpi_setup.pooled_log_likelihood(np.array([0.1])) == -2.0
        self.worker.log_likelihood.assert_called_once()

    def test_the_prior_transform_is_forwarded(self):
        self.worker.prior_transform.return_value = np.array([1.0])
        np.testing.assert_allclose(
            mpi_setup.pooled_prior_transform(np.array([0.5])), [1.0]
        )

    def test_the_initial_point_draw_is_forwarded(self):
        self.worker.get_initial_point_from_prior.return_value = ("unit", "theta", -1.0)
        assert mpi_setup.pooled_initial_point_from_prior("rng") == ("unit", "theta", -1.0)
        self.worker.get_initial_point_from_prior.assert_called_once_with("rng")

