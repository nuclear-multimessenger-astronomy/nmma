import numpy as np
import pytest
from bilby.core.prior import Constraint, PriorDict, Uniform
from scipy.stats.distributions import rv_frozen

from nmma.core.base import NMMALikelihood
from nmma.population import pop_likelihood
from nmma.population.pop_likelihood import NeutronStarPopulation


def binary(mass_1=1.5, mass_2=1.4, mass_ratio=None):
    """The three parameters the population likelihood reads. The masses are
    source-frame, because a population is defined in the source frame."""
    if mass_ratio is None:
        mass_ratio = mass_2 / mass_1
    return {
        "mass_1_source": mass_1,
        "mass_2_source": mass_2,
        "mass_ratio": mass_ratio,
    }


class TestFlatModel:
    """The uniform neutron-star mass distribution of Landry and Read, which
    the paper places between 1.1 and 2.0 solar masses."""

    def setup_method(self):
        self.model = NeutronStarPopulation("flat")

    def test_the_distribution_is_a_frozen_uniform(self):
        assert isinstance(self.model.distribution, rv_frozen)
        assert self.model.distribution.dist.name == "uniform"

    def test_the_lower_mass_edge_is_the_intended_one(self):
        assert self.model.distribution.support()[0] == pytest.approx(1.1)

    def test_the_upper_mass_edge_overshoots_the_intended_one(self):
        # scipy's uniform takes a width as its scale, not an upper bound, so
        # passing the intended maximum of 2.0 as the scale stretches the
        # support to 1.1 + 2.0. Passing m_max - m_min would fix it. As it
        # stands the model accepts neutron stars far above any plausible
        # maximum mass.
        assert self.model.distribution.support()[1] == pytest.approx(3.1)

    def test_an_implausibly_heavy_neutron_star_is_still_allowed(self):
        assert np.isfinite(self.model.distribution.logpdf(3.0))

    def test_the_density_is_normalised_over_the_overshooting_support(self):
        # The density is one over the support width, so the wrong width also
        # makes every mass less likely than the paper intends.
        assert self.model.distribution.pdf(1.5) == pytest.approx(1 / 2.0)
        assert self.model.distribution.pdf(1.5) != pytest.approx(1 / 0.9)

    def test_the_density_is_flat_across_the_support(self):
        masses = np.linspace(1.2, 3.0, 20)
        densities = self.model.distribution.pdf(masses)
        np.testing.assert_allclose(densities, densities[0])

    def test_a_mass_below_the_lower_edge_is_excluded(self):
        assert self.model.distribution.logpdf(1.0) == -np.inf

    def test_a_mass_above_the_support_is_excluded(self):
        assert self.model.distribution.logpdf(3.2) == -np.inf

    def test_the_model_name_is_case_insensitive(self):
        for name in ["flat", "FLAT", "Flat"]:
            model = NeutronStarPopulation(name)
            assert model.distribution.support()[0] == pytest.approx(1.1), name


class TestPeakModel:
    """The peaked neutron-star mass distribution: a normal distribution
    centred at 1.5 solar masses, truncated to between 1.1 and 2.1."""

    def setup_method(self):
        self.model = NeutronStarPopulation("peak")

    def test_the_distribution_is_a_frozen_truncated_normal(self):
        assert isinstance(self.model.distribution, rv_frozen)
        assert self.model.distribution.dist.name == "truncnorm"

    def test_the_support_matches_the_intended_mass_range(self):
        # The truncation points are given in standard deviations from the
        # centre, so this model does not suffer the flat model's scale error.
        lower, upper = self.model.distribution.support()
        assert lower == pytest.approx(1.1)
        assert upper == pytest.approx(2.1)

    def test_the_density_peaks_at_the_central_mass(self):
        masses = np.linspace(1.1, 2.1, 201)
        peak = masses[np.argmax(self.model.distribution.pdf(masses))]
        assert peak == pytest.approx(1.5, abs=1.5 * 10**-2)

    def test_the_density_falls_off_away_from_the_peak(self):
        assert self.model.distribution.pdf(1.5) > self.model.distribution.pdf(2.0)
        assert self.model.distribution.pdf(1.5) > self.model.distribution.pdf(1.2)

    def test_a_mass_below_the_truncation_is_excluded(self):
        assert self.model.distribution.logpdf(1.0) == -np.inf

    def test_a_mass_above_the_truncation_is_excluded(self):
        assert self.model.distribution.logpdf(2.2) == -np.inf

    def test_the_density_is_normalised_over_the_truncated_range(self):
        masses = np.linspace(1.1, 2.1, 10001)
        integral = np.trapezoid(self.model.distribution.pdf(masses), masses)
        assert integral == pytest.approx(1.0, abs=1.5 * 10**-4)

    def test_the_model_name_is_case_insensitive(self):
        for name in ["peak", "PEAK", "Peak"]:
            model = NeutronStarPopulation(name)
            assert model.distribution.support()[1] == pytest.approx(2.1), name

    def test_the_two_models_are_different_distributions(self):
        flat = NeutronStarPopulation("flat")
        assert flat.distribution.logpdf(1.5) != pytest.approx(
            self.model.distribution.logpdf(1.5)
        )


class TestUnknownModel:
    """Only the two named models are implemented, and anything else falls
    through the constructor without being reported."""

    def test_an_unknown_name_is_accepted_without_complaint(self):
        # There is no else branch, so the object is built with no
        # distribution at all. Raising here would surface the mistake at
        # setup instead of deep inside the sampler.
        model = NeutronStarPopulation("does_not_exist")
        assert not hasattr(model, "distribution")

    def test_the_failure_only_appears_when_the_likelihood_is_evaluated(self):
        model = NeutronStarPopulation("does_not_exist")
        with pytest.raises(AttributeError):
            model.log_likelihood(binary())

    def test_the_command_line_default_names_no_implemented_model(self):
        # The joint parser defaults --population-model to "uniform", which
        # matches neither branch, so a population run started without an
        # explicit model builds an unusable likelihood.
        import argparse

        from nmma.joint.joint_parsing import injection_parsing

        default = injection_parsing(argparse.ArgumentParser()).parse_args([])
        assert default.population_model == "uniform"
        assert not hasattr(
            NeutronStarPopulation(default.population_model), "distribution"
        )

    def test_the_beta_exponent_is_still_stored(self):
        model = NeutronStarPopulation("does_not_exist", beta=2.0)
        assert model.beta == 2.0


class TestPairingExponent:
    """The mass ratio is weighted by a pairing exponent, which says how
    strongly the population favours equal-mass binaries."""

    def test_the_exponent_defaults_to_no_pairing_preference(self):
        assert NeutronStarPopulation("peak").beta == 0.0

    def test_the_exponent_is_stored_as_given(self):
        assert NeutronStarPopulation("peak", beta=1.5).beta == 1.5

    def test_no_pairing_preference_leaves_the_likelihood_to_the_masses_alone(self):
        model = NeutronStarPopulation("peak")
        parameters = binary(mass_ratio=0.5)
        expected = model.distribution.logpdf(1.5) + model.distribution.logpdf(1.4)
        assert model.log_likelihood(parameters) == pytest.approx(expected)

    def test_the_mass_ratio_term_does_not_depend_on_the_mass_ratio_without_pairing(
        self,
    ):
        model = NeutronStarPopulation("peak")
        first = model.log_likelihood(binary(mass_ratio=0.2))
        second = model.log_likelihood(binary(mass_ratio=0.9))
        assert first == pytest.approx(second)

    def test_a_positive_exponent_favours_equal_mass_binaries(self):
        model = NeutronStarPopulation("peak", beta=2.0)
        unequal = model.log_likelihood(binary(mass_ratio=0.5))
        equal = model.log_likelihood(binary(mass_ratio=1.0))
        assert equal > unequal

    def test_a_negative_exponent_favours_unequal_mass_binaries(self):
        model = NeutronStarPopulation("peak", beta=-2.0)
        unequal = model.log_likelihood(binary(mass_ratio=0.5))
        equal = model.log_likelihood(binary(mass_ratio=1.0))
        assert unequal > equal

    def test_the_mass_ratio_term_is_the_exponent_times_its_logarithm(self):
        model = NeutronStarPopulation("peak", beta=3.0)
        without = NeutronStarPopulation("peak", beta=0.0)
        difference = model.log_likelihood(
            binary(mass_ratio=0.4)
        ) - without.log_likelihood(binary(mass_ratio=0.4))
        assert difference == pytest.approx(3.0 * np.log(0.4))

    def test_an_equal_mass_binary_gets_no_pairing_contribution(self):
        model = NeutronStarPopulation("peak", beta=5.0)
        without = NeutronStarPopulation("peak", beta=0.0)
        assert model.log_likelihood(binary(mass_ratio=1.0)) == pytest.approx(
            without.log_likelihood(binary(mass_ratio=1.0))
        )

    def test_a_very_large_exponent_underflows_to_minus_infinity(self):
        # The term is computed as log(q**beta) rather than beta*log(q), so
        # the power underflows to zero before the logarithm is taken. Only
        # unrealistically large exponents reach this, but the algebraically
        # equal form would not.
        model = NeutronStarPopulation("peak", beta=2000.0)
        with np.errstate(divide="ignore"):
            value = model.log_likelihood(binary(mass_ratio=0.5))
        assert value == -np.inf
        assert np.isfinite(2000.0 * np.log(0.5))


class TestLogLikelihood:
    """Both components are drawn from the same mass distribution, so the
    likelihood is the sum of their densities plus the pairing term."""

    def setup_method(self):
        self.model = NeutronStarPopulation("peak")

    def test_both_components_contribute(self):
        expected = self.model.distribution.logpdf(1.6) + self.model.distribution.logpdf(
            1.3
        )
        assert self.model.log_likelihood(
            binary(1.6, 1.3, mass_ratio=1.0)
        ) == pytest.approx(expected)

    def test_the_components_are_interchangeable(self):
        first = self.model.log_likelihood(binary(1.7, 1.3, mass_ratio=1.0))
        second = self.model.log_likelihood(binary(1.3, 1.7, mass_ratio=1.0))
        assert first == pytest.approx(second)

    def test_a_binary_at_the_peak_is_the_most_likely(self):
        at_peak = self.model.log_likelihood(binary(1.5, 1.5, mass_ratio=1.0))
        off_peak = self.model.log_likelihood(binary(2.0, 1.2, mass_ratio=1.0))
        assert at_peak > off_peak

    def test_a_component_outside_the_population_is_excluded(self):
        assert self.model.log_likelihood(binary(2.5, 1.4, mass_ratio=1.0)) == -np.inf

    def test_either_component_being_outside_excludes_the_binary(self):
        assert self.model.log_likelihood(binary(1.5, 0.9, mass_ratio=1.0)) == -np.inf

    def test_a_table_of_binaries_is_evaluated_elementwise(self):
        parameters = {
            "mass_1_source": np.array([1.5, 1.6]),
            "mass_2_source": np.array([1.4, 1.3]),
            "mass_ratio": np.array([0.93, 0.81]),
        }
        values = self.model.log_likelihood(parameters)
        assert values.shape == (2,)
        assert np.all(np.isfinite(values))

    def test_one_excluded_row_does_not_exclude_the_others(self):
        parameters = {
            "mass_1_source": np.array([1.5, 3.0]),
            "mass_2_source": np.array([1.4, 1.3]),
            "mass_ratio": np.array([0.93, 0.43]),
        }
        values = self.model.log_likelihood(parameters)
        assert np.isfinite(values[0])
        assert values[1] == -np.inf

    def test_the_source_frame_masses_are_required(self):
        with pytest.raises(KeyError):
            self.model.log_likelihood({"mass_1": 1.5, "mass_2": 1.4, "mass_ratio": 0.9})

    def test_the_mass_ratio_is_required(self):
        with pytest.raises(KeyError):
            self.model.log_likelihood({"mass_1_source": 1.5, "mass_2_source": 1.4})

    def test_extra_parameters_are_ignored(self):
        parameters = binary(mass_ratio=1.0)
        parameters["luminosity_distance"] = 40.0
        assert self.model.log_likelihood(parameters) == pytest.approx(
            self.model.log_likelihood(binary(mass_ratio=1.0))
        )

    def test_the_flat_model_gives_the_same_value_for_any_allowed_pair(self):
        flat = NeutronStarPopulation("flat")
        first = flat.log_likelihood(binary(1.3, 1.2, mass_ratio=1.0))
        second = flat.log_likelihood(binary(2.0, 1.9, mass_ratio=1.0))
        assert first == pytest.approx(second)


class TestUseAsAMessengerLikelihood:
    """The joint pipeline wraps the population model in the generic NMMA
    likelihood, which is the only way it is ever evaluated."""

    def setup_method(self):
        self.priors = PriorDict()
        self.priors["mass_1_source"] = Uniform(1.1, 2.0, "mass_1_source")
        self.model = NeutronStarPopulation("peak")
        self.likelihood = NMMALikelihood(self.model, self.priors)

    def test_the_population_model_is_kept_as_the_submodel(self):
        assert self.likelihood.sub_model is self.model

    def test_the_wrapped_likelihood_reports_the_population_value(self):
        parameters = binary(mass_ratio=1.0)
        assert self.likelihood.sub_log_likelihood(parameters) == pytest.approx(
            self.model.log_likelihood(parameters)
        )

    def test_a_population_has_no_noise_evidence(self):
        # There is no data and so no noise hypothesis; the wrapper falls
        # back to zero because the model defines no noise likelihood.
        assert self.likelihood.noise_log_likelihood() == 0.0

    def test_an_excluded_binary_is_floored_rather_than_left_infinite(self):
        value = self.likelihood.sub_log_likelihood(binary(2.5, 1.4, mass_ratio=1.0))
        assert np.isfinite(value)
        assert value < -1e300

    def test_the_full_likelihood_runs_through_the_conversion_and_constraints(self):
        value = self.likelihood.log_likelihood(binary(mass_ratio=1.0))
        assert value == pytest.approx(
            self.model.log_likelihood(binary(mass_ratio=1.0))
        )

    def test_a_violated_constraint_floors_the_likelihood(self):
        priors = PriorDict()
        priors["mass_1_source"] = Uniform(1.1, 2.0, "mass_1_source")
        priors["forbidden"] = Constraint(0, 1, "forbidden")
        likelihood = NMMALikelihood(NeutronStarPopulation("peak"), priors)
        parameters = binary(mass_ratio=1.0)
        parameters["forbidden"] = 5.0
        assert likelihood.log_likelihood(parameters) < -1e300

    def test_the_representation_names_the_population_model(self):
        assert "NeutronStarPopulation" in repr(self.likelihood)

    def test_the_joint_pipeline_builds_it_from_the_model_name(self):
        # joint_likelihood.setup_from_args constructs the model from the
        # argument value alone, so the name is the whole configuration.
        from argparse import Namespace

        args = Namespace(population_model="peak")
        model = NeutronStarPopulation(args.population_model)
        assert model.distribution.support()[1] == pytest.approx(2.1)


class TestPackageExports:
    def test_the_likelihood_module_is_reachable_from_the_package(self):
        from nmma import population

        assert population.pop_likelihood is pop_likelihood

    def test_the_population_class_is_the_one_the_joint_module_imports(self):
        from nmma.joint.joint_likelihood import (
            NeutronStarPopulation as imported_class,
        )

        assert imported_class is NeutronStarPopulation
