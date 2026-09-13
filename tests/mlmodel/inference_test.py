import bilby
import numpy as np
import pytest
from bilby.core.prior import PriorDict, Uniform

try:
    import torch

    from nmma.mlmodel import inference

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is the optional neuralnet extra
    TORCH_AVAILABLE = False

needs_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch is not installed; install the neuralnet extra"
)

PARAMETERS = ["log10_mej", "log10_vej", "log10_Xlan"]


def ejecta_priors():
    priors = PriorDict()
    for name in PARAMETERS:
        priors[name] = Uniform(-3, 0, name)
    return priors


@needs_torch
class TestModuleConstants:
    def test_the_light_curve_length_matches_the_padding_target(self):
        from nmma.mlmodel import dataprocessing

        assert inference.num_points == dataprocessing.num_points


@needs_torch
class CastResultMixin:
    """The flow returns a bare tensor of samples; this turns it into the
    bilby result the rest of the package plots and saves."""

    def setup_method(self):
        self.priors = ejecta_priors()
        self.samples = torch.tensor(
            [
                [-1.0, -0.5, -2.0],
                [-1.5, -0.6, -2.5],
                [-2.0, -0.7, -3.0],
            ]
        )

    def cast(self, truth=None, samples=None):
        return inference.cast_as_bilby_result(
            self.samples if samples is None else samples, truth, self.priors
        )


@needs_torch
class TestCastAsBilbyResult(CastResultMixin):
    def test_a_bilby_result_is_returned(self):
        assert isinstance(self.cast(), bilby.result.Result)

    def test_the_three_ejecta_parameters_become_the_posterior_columns(self):
        # The column order is not fixed, because the keys are collected from
        # a set literal, so only the set of names is part of the contract.
        assert set(self.cast().posterior.columns) == set(PARAMETERS)

    def test_one_posterior_row_is_produced_per_sample(self):
        assert len(self.cast().posterior) == 3

    def test_each_column_of_the_tensor_maps_to_its_own_parameter(self):
        posterior = self.cast().posterior
        np.testing.assert_allclose(posterior["log10_mej"], [-1.0, -1.5, -2.0])
        np.testing.assert_allclose(posterior["log10_vej"], [-0.5, -0.6, -0.7])
        np.testing.assert_allclose(posterior["log10_Xlan"], [-2.0, -2.5, -3.0])

    def test_the_sampled_parameters_are_recorded_for_plotting(self):
        result = self.cast()
        assert set(result.search_parameter_keys) == set(PARAMETERS)

    def test_the_priors_are_carried_onto_the_result(self):
        result = self.cast()
        for name in PARAMETERS:
            assert name in result.priors, name

    def test_the_result_is_labelled(self):
        assert self.cast().label == "test_data"

    def test_a_single_sample_is_accepted(self):
        result = self.cast(samples=torch.tensor([[-1.0, -0.5, -2.0]]))
        assert len(result.posterior) == 1

    def test_a_large_sample_set_is_accepted(self):
        result = self.cast(samples=torch.randn(20000, 3))
        assert len(result.posterior) == 20000

    def test_an_extra_leading_axis_is_flattened_away(self):
        # The flow returns samples shaped (context, samples, parameters), and
        # each parameter column is flattened before it is stored.
        result = self.cast(samples=torch.randn(1, 5, 3))
        assert len(result.posterior) == 5


@needs_torch
class TestCastAsBilbyResultWithoutTruth(CastResultMixin):
    """A run on real data has no injected values to compare against."""

    def test_no_injection_parameters_are_recorded(self):
        assert self.cast(truth=None).injection_parameters is None

    def test_the_posterior_is_still_built(self):
        assert len(self.cast(truth=None).posterior) == 3


@needs_torch
class TestCastAsBilbyResultWithTruth(CastResultMixin):
    """An injection run records the true values so they can be drawn onto
    the corner plot."""

    truth = None

    def setup_method(self):
        super().setup_method()
        self.truth = torch.tensor([-1.2, -0.55, -2.2])

    def test_the_injected_values_are_recorded(self):
        result = self.cast(truth=self.truth)
        assert result.injection_parameters is not None
        assert set(result.injection_parameters) == set(PARAMETERS)

    def test_each_injected_value_maps_to_its_own_parameter(self):
        injected = self.cast(truth=self.truth).injection_parameters
        assert injected["log10_mej"] == pytest.approx(-1.2, abs=1.5e-5)
        assert injected["log10_vej"] == pytest.approx(-0.55, abs=1.5e-5)
        assert injected["log10_Xlan"] == pytest.approx(-2.2, abs=1.5e-5)

    def test_the_injected_values_are_plain_floats_not_tensors(self):
        # The result writer cannot serialise a tensor, so each value is cast.
        injected = self.cast(truth=self.truth).injection_parameters
        for name, value in injected.items():
            assert isinstance(value, float), name

    def test_the_posterior_is_unaffected_by_the_truth(self):
        with_truth = self.cast(truth=self.truth).posterior
        without = self.cast(truth=None).posterior
        np.testing.assert_allclose(with_truth["log10_mej"], without["log10_mej"])

    def test_a_tensor_truth_is_recognised_as_given(self):
        # The check is written as an equality against None, which a tensor
        # answers without raising, so the injection branch is reached.
        assert self.cast(truth=self.truth).injection_parameters is not None

    def test_a_truth_with_too_few_values_is_reported(self):
        with pytest.raises(IndexError):
            self.cast(truth=torch.tensor([-1.2, -0.55]))
