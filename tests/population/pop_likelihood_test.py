import numpy as np
import pytest
from bilby.core.prior import Constraint, PriorDict, Uniform
from scipy.stats.distributions import rv_frozen

from nmma.core.base import NMMALikelihood
from nmma.population import pop_likelihood
from nmma.population.pop_likelihood import (
    NeutronStarPopulation,
    PeakNeutronStarPopulation,
    build_population_model,
)


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


class TestNeutronStarPopulation:
    """The base class is the uniform (flat) neutron-star mass distribution
    of Landry and Read, which the paper places between 1.1 and 2.0 solar
    masses by default."""

    def setup_method(self):
        self.m_min = 1.1
        self.m_max = 2.0
        self.model = NeutronStarPopulation(m_min=self.m_min, m_max=self.m_max)

    def test_the_mass_limits_can_be_set_by_user(self):
        custom_m_min, custom_m_max = 1.0, 3.0
        model = NeutronStarPopulation(m_min=custom_m_min, m_max=custom_m_max)
        lower, upper = model.distribution.support()
        assert lower == pytest.approx(custom_m_min)
        assert upper == pytest.approx(custom_m_max)

    def test_the_density_is_normalised_over_the_support(self):
        mass_inside_support = (self.m_max + self.m_min) / 2
        # The density is one over the support width (m_max - m_min).
        assert self.model.distribution.pdf(mass_inside_support) == pytest.approx(
            1 / (self.m_max - self.m_min)
        )

    def test_the_beta_exponent_defaults_to_no_pairing_preference(self):
        assert self.model.beta == 0.0

    def test_the_beta_exponent_is_stored_as_given(self):
        custom_beta = 1.5
        assert NeutronStarPopulation(beta=custom_beta).beta == custom_beta


class TestPeakNeutronStarPopulation:
    """The peaked neutron-star mass distribution: a normal distribution
    centred at 1.5 solar masses by default, truncated to between 1.1 and
    2.1."""

    def setup_method(self):
        self.m_min = 1.1
        self.m_max = 2.1
        self.loc = 1.5
        self.scale = 1.0
        self.model = PeakNeutronStarPopulation(
            m_min=self.m_min, m_max=self.m_max, loc=self.loc, scale=self.scale
        )

    def test_it_is_a_neutron_star_population(self):
        assert isinstance(self.model, NeutronStarPopulation)

    def test_the_distribution_is_a_frozen_truncated_normal(self):
        assert isinstance(self.model.distribution, rv_frozen)
        assert self.model.distribution.dist.name == "truncnorm"

    def test_the_default_support_matches_the_intended_mass_range(self):
        lower, upper = self.model.distribution.support()
        assert lower == pytest.approx(self.m_min)
        assert upper == pytest.approx(self.m_max)

    def test_the_mass_limits_and_peak_can_be_set_by_user(self):
        custom_m_min, custom_m_max = 1.2, 2.4
        custom_loc, custom_scale = 1.8, 0.5
        model = PeakNeutronStarPopulation(
            m_min=custom_m_min,
            m_max=custom_m_max,
            loc=custom_loc,
            scale=custom_scale,
        )
        lower, upper = model.distribution.support()
        assert lower == pytest.approx(custom_m_min)
        assert upper == pytest.approx(custom_m_max)

    def test_the_density_peaks_at_the_central_mass(self):
        masses = np.linspace(self.m_min, self.m_max, 201)
        peak = masses[np.argmax(self.model.distribution.pdf(masses))]
        assert peak == pytest.approx(self.loc, abs=1.5 * 10**-2)

    def test_the_density_falls_off_away_from_the_peak(self):
        mass_above_peak = self.loc + 0.5
        mass_below_peak = self.loc - 0.3
        peak_density = self.model.distribution.pdf(self.loc)
        assert peak_density > self.model.distribution.pdf(mass_above_peak)
        assert peak_density > self.model.distribution.pdf(mass_below_peak)

    def test_a_mass_below_the_truncation_is_excluded(self):
        mass_below_support = self.m_min - 0.1
        assert self.model.distribution.logpdf(mass_below_support) == -np.inf

    def test_a_mass_above_the_truncation_is_excluded(self):
        mass_above_support = self.m_max + 0.1
        assert self.model.distribution.logpdf(mass_above_support) == -np.inf

    def test_the_density_is_normalised_over_the_truncated_range(self):
        masses = np.linspace(self.m_min, self.m_max, 10001)
        integral = np.trapezoid(self.model.distribution.pdf(masses), masses)
        assert integral == pytest.approx(1.0, abs=1.5 * 10**-4)

    def test_the_two_models_are_different_distributions(self):
        flat = NeutronStarPopulation()
        assert flat.distribution.logpdf(self.loc) != pytest.approx(
            self.model.distribution.logpdf(self.loc)
        )


class TestBuildPopulationModel:
    """The joint pipeline selects a population model by its CLI name; the
    factory maps that name onto the right class."""

    def test_flat_builds_a_neutron_star_population(self):
        model = build_population_model("flat")
        assert type(model) is NeutronStarPopulation

    def test_peak_builds_a_peak_neutron_star_population(self):
        model = build_population_model("peak")
        assert type(model) is PeakNeutronStarPopulation

    def test_the_model_name_is_case_insensitive(self):
        for name in ["flat", "FLAT", "Flat"]:
            assert type(build_population_model(name)) is NeutronStarPopulation, name

    def test_keyword_arguments_are_passed_through_to_the_model(self):
        custom_m_min, custom_m_max, custom_beta = 1.0, 3.0, 2.0
        model = build_population_model(
            "flat", m_min=custom_m_min, m_max=custom_m_max, beta=custom_beta
        )
        assert model.distribution.support() == pytest.approx(
            (custom_m_min, custom_m_max)
        )
        assert model.beta == custom_beta

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(KeyError):
            build_population_model("does_not_exist")


class TestPairingExponent:
    """The mass ratio is weighted by a pairing exponent, which says how
    strongly the population favours equal-mass binaries."""

    def setup_method(self):
        self.equal_mass_ratio = 1.0
        self.unequal_mass_ratio = 0.5

    def test_no_pairing_preference_leaves_the_likelihood_to_the_masses_alone(self):
        model = PeakNeutronStarPopulation()
        parameters = binary(mass_ratio=self.unequal_mass_ratio)
        expected = model.distribution.logpdf(
            parameters["mass_1_source"]
        ) + model.distribution.logpdf(parameters["mass_2_source"])
        assert model.log_likelihood(parameters) == pytest.approx(expected)

    def test_the_mass_ratio_term_does_not_depend_on_the_mass_ratio_without_pairing(
        self,
    ):
        model = PeakNeutronStarPopulation()
        low_mass_ratio, high_mass_ratio = 0.2, 0.9
        first = model.log_likelihood(binary(mass_ratio=low_mass_ratio))
        second = model.log_likelihood(binary(mass_ratio=high_mass_ratio))
        assert first == pytest.approx(second)

    def test_a_positive_exponent_favours_equal_mass_binaries(self):
        positive_beta = 2.0
        model = PeakNeutronStarPopulation(beta=positive_beta)
        unequal = model.log_likelihood(binary(mass_ratio=self.unequal_mass_ratio))
        equal = model.log_likelihood(binary(mass_ratio=self.equal_mass_ratio))
        assert equal > unequal

    def test_a_negative_exponent_favours_unequal_mass_binaries(self):
        negative_beta = -2.0
        model = PeakNeutronStarPopulation(beta=negative_beta)
        unequal = model.log_likelihood(binary(mass_ratio=self.unequal_mass_ratio))
        equal = model.log_likelihood(binary(mass_ratio=self.equal_mass_ratio))
        assert unequal > equal

    def test_the_mass_ratio_term_is_the_exponent_times_its_logarithm(self):
        beta = 3.0
        mass_ratio = 0.4
        model = PeakNeutronStarPopulation(beta=beta)
        without = PeakNeutronStarPopulation(beta=0.0)
        difference = model.log_likelihood(
            binary(mass_ratio=mass_ratio)
        ) - without.log_likelihood(binary(mass_ratio=mass_ratio))
        assert difference == pytest.approx(beta * np.log(mass_ratio))

    def test_an_equal_mass_binary_gets_no_pairing_contribution(self):
        beta = 5.0
        model = PeakNeutronStarPopulation(beta=beta)
        without = PeakNeutronStarPopulation(beta=0.0)
        parameters = binary(mass_ratio=self.equal_mass_ratio)
        assert model.log_likelihood(parameters) == pytest.approx(
            without.log_likelihood(parameters)
        )

    def test_a_very_large_exponent_underflows_to_minus_infinity(self):
        # The term is computed as log(q**beta) rather than beta*log(q), so
        # the power underflows to zero before the logarithm is taken. Only
        # unrealistically large exponents reach this, but the algebraically
        # equal form would not.
        huge_beta = 2000.0
        model = PeakNeutronStarPopulation(beta=huge_beta)
        with np.errstate(divide="ignore"):
            value = model.log_likelihood(binary(mass_ratio=self.unequal_mass_ratio))
        assert value == -np.inf
        assert np.isfinite(huge_beta * np.log(self.unequal_mass_ratio))


class TestLogLikelihood:
    """Both components are drawn from the same mass distribution, so the
    likelihood is the sum of their densities plus the pairing term."""

    def setup_method(self):
        self.m_min = 1.1
        self.m_max = 2.1
        self.equal_mass_ratio = 1.0
        self.model = PeakNeutronStarPopulation(m_min=self.m_min, m_max=self.m_max)

    def test_both_components_contribute(self):
        mass_1, mass_2 = 1.6, 1.3
        expected = self.model.distribution.logpdf(
            mass_1
        ) + self.model.distribution.logpdf(mass_2)
        assert self.model.log_likelihood(
            binary(mass_1, mass_2, mass_ratio=self.equal_mass_ratio)
        ) == pytest.approx(expected)

    def test_the_components_are_interchangeable(self):
        mass_1, mass_2 = 1.7, 1.3
        first = self.model.log_likelihood(
            binary(mass_1, mass_2, mass_ratio=self.equal_mass_ratio)
        )
        second = self.model.log_likelihood(
            binary(mass_2, mass_1, mass_ratio=self.equal_mass_ratio)
        )
        assert first == pytest.approx(second)

    def test_a_binary_at_the_peak_is_the_most_likely(self):
        peak_mass = 1.5
        off_peak_mass_1, off_peak_mass_2 = 2.0, 1.2
        at_peak = self.model.log_likelihood(
            binary(peak_mass, peak_mass, mass_ratio=self.equal_mass_ratio)
        )
        off_peak = self.model.log_likelihood(
            binary(off_peak_mass_1, off_peak_mass_2, mass_ratio=self.equal_mass_ratio)
        )
        assert at_peak > off_peak

    def test_a_component_outside_the_population_is_excluded(self):
        mass_outside_support = 2.5
        assert self.model.log_likelihood(
            binary(mass_outside_support, 1.4, mass_ratio=self.equal_mass_ratio)
        ) == -np.inf

    def test_either_component_being_outside_excludes_the_binary(self):
        mass_outside_support = 0.9
        assert self.model.log_likelihood(
            binary(1.5, mass_outside_support, mass_ratio=self.equal_mass_ratio)
        ) == -np.inf

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
        mass_outside_support = 3.0
        parameters = {
            "mass_1_source": np.array([1.5, mass_outside_support]),
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
        parameters = binary(mass_ratio=self.equal_mass_ratio)
        parameters["luminosity_distance"] = 40.0
        assert self.model.log_likelihood(parameters) == pytest.approx(
            self.model.log_likelihood(binary(mass_ratio=self.equal_mass_ratio))
        )

    def test_the_flat_model_gives_the_same_value_for_any_allowed_pair(self):
        flat_m_min, flat_m_max = 1.1, 2.0
        flat = NeutronStarPopulation(m_min=flat_m_min, m_max=flat_m_max)
        first = flat.log_likelihood(
            binary(1.3, 1.2, mass_ratio=self.equal_mass_ratio)
        )
        second = flat.log_likelihood(
            binary(flat_m_max, flat_m_max - 0.1, mass_ratio=self.equal_mass_ratio)
        )
        assert first == pytest.approx(second)


class TestUseAsAMessengerLikelihood:
    """The joint pipeline wraps the population model in the generic NMMA
    likelihood, which is the only way it is ever evaluated."""

    def setup_method(self):
        self.prior_m_min = 1.1
        self.prior_m_max = 2.0
        self.equal_mass_ratio = 1.0
        self.priors = PriorDict()
        self.priors["mass_1_source"] = Uniform(
            self.prior_m_min, self.prior_m_max, "mass_1_source"
        )
        self.model = PeakNeutronStarPopulation()
        self.likelihood = NMMALikelihood(self.model, self.priors)

    def test_the_population_model_is_kept_as_the_submodel(self):
        assert self.likelihood.sub_model is self.model

    def test_the_wrapped_likelihood_reports_the_population_value(self):
        parameters = binary(mass_ratio=self.equal_mass_ratio)
        assert self.likelihood.sub_log_likelihood(parameters) == pytest.approx(
            self.model.log_likelihood(parameters)
        )

    def test_a_population_has_no_noise_evidence(self):
        # There is no data and so no noise hypothesis; the wrapper falls
        # back to zero because the model defines no noise likelihood.
        assert self.likelihood.noise_log_likelihood() == 0.0

    def test_an_excluded_binary_is_floored_rather_than_left_infinite(self):
        mass_outside_support = 2.5
        value = self.likelihood.sub_log_likelihood(
            binary(mass_outside_support, 1.4, mass_ratio=self.equal_mass_ratio)
        )
        assert np.isfinite(value)
        assert value < -1e300

    def test_the_full_likelihood_runs_through_the_conversion_and_constraints(self):
        parameters = binary(mass_ratio=self.equal_mass_ratio)
        value = self.likelihood.log_likelihood(parameters)
        assert value == pytest.approx(self.model.log_likelihood(parameters))

    def test_a_violated_constraint_floors_the_likelihood(self):
        priors = PriorDict()
        priors["mass_1_source"] = Uniform(
            self.prior_m_min, self.prior_m_max, "mass_1_source"
        )
        priors["forbidden"] = Constraint(0, 1, "forbidden")
        likelihood = NMMALikelihood(PeakNeutronStarPopulation(), priors)
        parameters = binary(mass_ratio=self.equal_mass_ratio)
        parameters["forbidden"] = 5.0
        assert likelihood.log_likelihood(parameters) < -1e300

    def test_the_representation_names_the_population_model(self):
        assert "NeutronStarPopulation" in repr(self.likelihood)

    def test_the_joint_pipeline_builds_it_from_the_model_name(self):
        # joint_likelihood.setup_from_args constructs the model from the
        # argument value alone, via the same factory as build_population_model.
        from argparse import Namespace

        args = Namespace(population_model="peak")
        model = build_population_model(args.population_model)
        peak_m_max = 2.1
        assert model.distribution.support()[1] == pytest.approx(peak_m_max)


class TestPackageExports:
    def test_the_likelihood_module_is_reachable_from_the_package(self):
        from nmma import population

        assert population.pop_likelihood is pop_likelihood

    def test_the_joint_module_uses_the_same_factory(self):
        from nmma.joint.joint_likelihood import (
            build_population_model as imported_factory,
        )

        assert imported_factory is build_population_model
