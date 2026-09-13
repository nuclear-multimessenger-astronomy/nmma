from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from bilby.core.likelihood import Likelihood
from bilby.core.prior import Constraint, DeltaFunction, PriorDict, Uniform

from nmma.joint import joint_likelihood
from nmma.joint.joint_likelihood import MultiMessengerLikelihood


class StubMessengerLikelihood(Likelihood):
    """A stand-in for one messenger's likelihood. MultiMessengerLikelihood
    only asks a sub-likelihood for a log likelihood, a sanity check, the two
    conversions and a diagnostic, so nothing else needs to be real.

    Note that bilby's JointLikelihood deepcopies the likelihoods it is given,
    so assertions have to read the copies held in ``likelihoods``, never the
    instances passed to the constructor."""

    def __init__(self, name="stub", logl=-1.0, noise_logl=-5.0, sane=True):
        super().__init__(parameters=dict())
        self.name = name
        self.logl = logl
        self.noise_logl = noise_logl
        self.sane = sane
        self.seen_parameters = None
        self.conv_functions = []
        self.conversion_was_set_up = False

    def __repr__(self):
        return self.name

    def log_likelihood(self, parameters=None):
        return self.logl

    def noise_log_likelihood(self):
        return self.noise_logl

    def sub_log_likelihood(self, parameters):
        self.seen_parameters = parameters
        return self.logl

    def sanity_checks(self):
        return self.sane

    def parameter_conversion(self, parameters):
        parameters = dict(parameters)
        parameters[f"{self.name}_added"] = 1.0
        return parameters

    def posterior_conversion(self, samples):
        samples = samples.copy()
        samples[f"{self.name}_added"] = 1.0
        return samples

    def final_diagnostics(self, bestfit_params, args, result=None):
        return f"{self.name}_diagnostics"

    def setup_parameter_conversion(self):
        self.conversion_was_set_up = True


class StubBuiltLikelihood(StubMessengerLikelihood):
    """A stub for the likelihoods setup_from_args constructs itself: it
    passes the priors positionally and the setup helper's kwargs by name,
    so the constructor has to absorb both."""

    def __init__(self, *args, name="built", **kwargs):
        super().__init__(name=name)
        self.init_args = args
        self.init_kwargs = kwargs


class StubEMLikelihood(StubBuiltLikelihood):
    pass


class StubGWLikelihood(StubBuiltLikelihood):
    pass


class StubEoSLikelihood(StubBuiltLikelihood):
    pass


def simple_priors(with_constraint=False):
    priors = PriorDict()
    priors["x"] = Uniform(0, 1, "x")
    if with_constraint:
        priors["mass_ratio_constraint"] = Constraint(0, 1, "mass_ratio_constraint")
    return priors


UNSET = object()


class LikelihoodMixin:
    def build(self, names=("A", "B"), priors=None, instructions=UNSET, **kwargs):
        if priors is None:
            priors = simple_priors()
        if instructions is UNSET:
            instructions = {}
        self.stubs = [StubMessengerLikelihood(name, **kwargs) for name in names]
        return MultiMessengerLikelihood(self.stubs, priors, instructions)


class TestInitialisation(LikelihoodMixin):
    def test_every_messenger_likelihood_is_kept(self):
        likelihood = self.build(("A", "B", "C"))
        assert len(likelihood.likelihoods) == 3

    def test_the_noise_evidence_is_summed_once_at_construction(self):
        likelihood = self.build(("A", "B"), noise_logl=-5.0)
        assert likelihood._noise_logl == -10.0

    def test_the_noise_evidence_is_cached_rather_than_recomputed(self):
        # Each sub-likelihood's noise evidence is fixed for the run, so it
        # is summed once instead of on every sampler call.
        likelihood = self.build(("A",), noise_logl=-3.0)
        likelihood.likelihoods[0].noise_logl = -99.0
        assert likelihood._noise_logl == -3.0

    def test_the_priors_are_stored(self):
        priors = simple_priors()
        likelihood = self.build(priors=priors)
        assert likelihood.priors is priors

    def test_constraint_priors_are_split_out_by_the_mixin(self):
        likelihood = self.build(priors=simple_priors(with_constraint=True))
        assert list(likelihood.constraints) == ["mass_ratio_constraint"]

    def test_without_constraints_the_constraint_set_is_empty(self):
        assert self.build().constraints == {}

    def test_the_conversion_instructions_are_stored(self):
        likelihood = self.build(instructions={"ejecta": True})
        assert likelihood.conversion_instructions == {"ejecta": True}

    def test_the_default_instructions_are_an_empty_dictionary(self):
        stubs = [StubMessengerLikelihood("A"), StubMessengerLikelihood("B")]
        likelihood = MultiMessengerLikelihood(stubs, simple_priors())
        assert likelihood.conversion_instructions == {}

    def test_a_non_likelihood_messenger_is_refused(self):
        with pytest.raises(ValueError):
            MultiMessengerLikelihood(["not a likelihood"], simple_priors())


class TestRepresentation(LikelihoodMixin):
    """The representation is logged to say which messengers a run combines."""

    def test_a_single_messenger_is_named_directly(self):
        likelihood = self.build(("Solo",))
        assert repr(likelihood) == "MultiMessengerLikelihood with Solo"

    def test_two_messengers_are_joined_with_and(self):
        likelihood = self.build(("A", "B"))
        assert repr(likelihood) == "MultiMessengerLikelihood with A and B"

    def test_three_messengers_are_comma_separated_before_the_last(self):
        likelihood = self.build(("A", "B", "C"))
        assert repr(likelihood) == "MultiMessengerLikelihood with A, B and C"


class TestSetupParameterConversion(LikelihoodMixin):
    """The conversion instructions decide which chain of conversions turns
    sampled parameters into the quantities each messenger needs."""

    def test_instructions_build_a_multimessenger_conversion(self):
        likelihood = self.build(instructions={"ejecta": True})
        assert likelihood.multi_conversion is not None
        assert len(likelihood.multi_conversion._conversions) == 1

    def test_no_instructions_still_builds_an_empty_conversion_chain(self):
        likelihood = self.build(instructions={})
        assert likelihood.multi_conversion._conversions == ()

    def test_instructions_of_none_fall_back_to_chaining_the_sub_likelihoods(self):
        likelihood = self.build(instructions=None)
        assert (
            likelihood.parameter_conversion.__func__
            == MultiMessengerLikelihood.basic_parameter_conversion
        )
        assert (
            likelihood.posterior_conversion.__func__
            == MultiMessengerLikelihood.basic_posterior_conversion
        )

    def test_instructions_of_none_build_no_conversion_object(self):
        likelihood = self.build(instructions=None)
        assert not hasattr(likelihood, "multi_conversion")

    def test_an_em_likelihood_contributes_its_own_conversion(self):
        with patch.object(joint_likelihood, "EMTransientLikelihood", StubEMLikelihood):
            stubs = [StubEMLikelihood(name="EM"), StubMessengerLikelihood("Other")]
            likelihood = MultiMessengerLikelihood(stubs, simple_priors(), {})
        assert "em" in likelihood.conversion_instructions
        assert likelihood.conversion_instructions["em"].__self__.name == "EM"

    def test_a_gw_likelihood_contributes_its_own_conversion(self):
        with patch.object(
            joint_likelihood, "GravitationalWaveTransientLikelihood", StubGWLikelihood
        ):
            stubs = [StubGWLikelihood(name="GW"), StubMessengerLikelihood("Other")]
            likelihood = MultiMessengerLikelihood(stubs, simple_priors(), {})
        assert "gw" in likelihood.conversion_instructions

    def test_an_eos_likelihood_contributes_its_own_conversion(self):
        with patch.object(
            joint_likelihood, "EquationofStateLikelihood", StubEoSLikelihood
        ):
            stubs = [StubEoSLikelihood(name="EoS"), StubMessengerLikelihood("Other")]
            likelihood = MultiMessengerLikelihood(stubs, simple_priors(), {})
        assert "eos" in likelihood.conversion_instructions

    def test_a_placeholder_instruction_is_replaced_by_the_real_conversion(self):
        # setup_from_args marks a messenger's conversion with True before the
        # likelihoods exist; setting up the conversion swaps in the method.
        with patch.object(joint_likelihood, "EMTransientLikelihood", StubEMLikelihood):
            stubs = [StubEMLikelihood(name="EM")]
            likelihood = MultiMessengerLikelihood(stubs, simple_priors(), {"em": True})
        assert likelihood.conversion_instructions["em"] != True  # noqa: E712
        assert callable(likelihood.conversion_instructions["em"])


class TestSanityChecks(LikelihoodMixin):
    def test_all_messengers_sane_passes(self):
        assert self.build(("A", "B"), sane=True).sanity_checks()

    def test_one_insane_messenger_fails_the_whole_check(self):
        likelihood = self.build(("A", "B"))
        likelihood.likelihoods[0].sane = False
        assert not likelihood.sanity_checks()

    def test_the_check_is_a_product_so_it_fails_on_any_messenger(self):
        likelihood = self.build(("A", "B", "C"))
        likelihood.likelihoods[2].sane = False
        assert not likelihood.sanity_checks()


class TestSubLogLikelihood(LikelihoodMixin):
    def test_the_messenger_log_likelihoods_are_added(self):
        likelihood = self.build(("A", "B"), logl=-2.5)
        assert likelihood.sub_log_likelihood({"x": 0.5}) == -5.0

    def test_every_messenger_sees_the_same_parameters(self):
        likelihood = self.build(("A", "B"))
        parameters = {"x": 0.5}
        likelihood.sub_log_likelihood(parameters)
        for sub in likelihood.likelihoods:
            assert sub.seen_parameters == parameters

    def test_a_minus_infinite_total_is_replaced_by_a_finite_floor(self):
        # Samplers reject the sample either way, but a literal -inf breaks
        # evidence bookkeeping, so it is clipped to the lowest float.
        likelihood = self.build(("A",), logl=-np.inf)
        value = likelihood.sub_log_likelihood({})
        assert np.isfinite(value)
        assert value < -1e300

    def test_a_not_a_number_total_is_also_floored(self):
        likelihood = self.build(("A",), logl=np.nan)
        value = likelihood.sub_log_likelihood({})
        assert np.isfinite(value)
        assert value < -1e300

    def test_one_infinite_messenger_drags_the_total_down(self):
        likelihood = self.build(("A", "B"))
        likelihood.likelihoods[0].logl = -np.inf
        assert np.isfinite(likelihood.sub_log_likelihood({}))


class TestLogLikelihood(LikelihoodMixin):
    """The mixin's log_likelihood converts, checks constraints, then sums."""

    def test_the_summed_messenger_likelihood_is_returned(self):
        likelihood = self.build(("A", "B"), instructions=None, logl=-2.0)
        assert likelihood.log_likelihood({"x": 0.5}) == -4.0

    def test_a_violated_constraint_floors_the_likelihood(self):
        priors = PriorDict()
        priors["x"] = Uniform(0, 1, "x")
        priors["forbidden"] = Constraint(0, 1, "forbidden")
        likelihood = self.build(("A",), priors=priors, instructions=None)
        value = likelihood.log_likelihood({"x": 0.5, "forbidden": 5.0})
        assert value < -1e300

    def test_a_satisfied_constraint_leaves_the_likelihood_alone(self):
        priors = PriorDict()
        priors["x"] = Uniform(0, 1, "x")
        priors["allowed"] = Constraint(0, 1, "allowed")
        likelihood = self.build(("A",), priors=priors, instructions=None, logl=-2.0)
        assert likelihood.log_likelihood({"x": 0.5, "allowed": 0.5}) == -2.0

    def test_an_insane_setup_floors_the_likelihood(self):
        likelihood = self.build(("A",), instructions=None)
        likelihood.likelihoods[0].sane = False
        assert likelihood.log_likelihood({"x": 0.5}) < -1e300


class TestParameterConversion(LikelihoodMixin):
    def test_the_conversion_is_delegated_to_the_multimessenger_object(self):
        likelihood = self.build()
        likelihood.multi_conversion = MagicMock()
        likelihood.multi_conversion.convert_to_multimessenger_parameters.return_value = {
            "converted": True
        }
        assert likelihood.parameter_conversion({"x": 1}) == {"converted": True}
        likelihood.multi_conversion.convert_to_multimessenger_parameters.assert_called_once_with(
            {"x": 1}
        )

    def test_the_basic_conversion_chains_every_messenger_in_order(self):
        likelihood = self.build(("A", "B"))
        converted = likelihood.basic_parameter_conversion({"x": 0.5})
        assert converted["A_added"] == 1.0
        assert converted["B_added"] == 1.0

    def test_the_basic_conversion_keeps_the_sampled_parameters(self):
        likelihood = self.build(("A",))
        assert likelihood.basic_parameter_conversion({"x": 0.5})["x"] == 0.5


class TestPosteriorConversion(LikelihoodMixin):
    """Posterior conversion runs over a whole table of samples, and only
    numeric columns survive because the result writer cannot store objects."""

    def samples(self):
        return pd.DataFrame({"x": [0.1, 0.2], "label": ["first", "second"]})

    def test_the_core_conversion_runs_before_the_messenger_conversions(self):
        likelihood = self.build(("A",))
        likelihood.multi_conversion = MagicMock()
        likelihood.multi_conversion.core_conversion.side_effect = lambda df: df.assign(
            core=1.0
        )
        converted = likelihood.posterior_conversion(self.samples())
        assert "core" in converted.columns
        assert "A_added" in converted.columns

    def test_non_numeric_columns_are_dropped(self):
        likelihood = self.build(("A",))
        converted = likelihood.basic_posterior_conversion(self.samples())
        assert "label" not in converted.columns
        assert "x" in converted.columns

    def test_every_messenger_adds_its_own_columns(self):
        likelihood = self.build(("A", "B"))
        converted = likelihood.basic_posterior_conversion(self.samples())
        assert "A_added" in converted.columns
        assert "B_added" in converted.columns

    def test_the_sample_rows_are_preserved(self):
        likelihood = self.build(("A",))
        converted = likelihood.basic_posterior_conversion(self.samples())
        assert len(converted) == 2


class TestFinalDiagnostics(LikelihoodMixin):
    def test_one_diagnostic_is_returned_per_messenger(self):
        likelihood = self.build(("A", "B"))
        assert likelihood.final_diagnostics({"x": 0.5}, Namespace()) == [
            "A_diagnostics",
            "B_diagnostics",
        ]

    def test_the_result_object_is_passed_through(self):
        likelihood = self.build(("A",))
        sub = likelihood.likelihoods[0]
        sub.final_diagnostics = MagicMock(return_value="done")
        result = object()
        likelihood.final_diagnostics({"x": 0.5}, Namespace(), result)
        sub.final_diagnostics.assert_called_once_with({"x": 0.5}, Namespace(), result)


class SetupFromArgsMixin:
    """setup_from_args is the bridge from a generation data dump to a
    likelihood, so each messenger's setup helper is replaced by a stub and
    only the wiring decisions are exercised."""

    def setup_method(self):
        self.logger = MagicMock()
        self.priors = PriorDict()
        self.priors["x"] = Uniform(0, 1, "x")
        self.eos_converter = MagicMock()
        self.eos_converter.compute_macro_parameters = MagicMock(name="macro")

        self.patches = [
            patch.object(joint_likelihood, "EMTransientLikelihood", StubEMLikelihood),
            patch.object(
                joint_likelihood,
                "GravitationalWaveTransientLikelihood",
                StubGWLikelihood,
            ),
            patch.object(
                joint_likelihood, "EquationofStateLikelihood", StubEoSLikelihood
            ),
            patch.object(
                joint_likelihood, "setup_em_kwargs", return_value={"name": "EM"}
            ),
            patch.object(
                joint_likelihood, "setup_gw_kwargs", return_value={"name": "GW"}
            ),
            patch.object(
                joint_likelihood,
                "setup_eos_kwargs",
                return_value={
                    "name": "EoS",
                    "constraint_dict": {"mtov": 2.0},
                    "eos_converter": self.eos_converter,
                },
            ),
            patch.object(
                joint_likelihood, "EoSConverter", return_value=self.eos_converter
            ),
        ]
        for patcher in self.patches:
            patcher.start()

    def teardown_method(self):
        for patcher in self.patches:
            patcher.stop()

    def data_dump(self, messengers=(), modifiers=()):
        return {
            "messengers": list(messengers),
            "analysis_modifiers": list(modifiers),
        }

    def args(self, **kwargs):
        defaults = dict(
            ejecta_conversion=False, cosmology=None, population_model="uniform"
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def setup(self, messengers=(), modifiers=(), priors=None, **kwargs):
        return MultiMessengerLikelihood.setup_from_args(
            self.data_dump(messengers, modifiers),
            self.priors if priors is None else priors,
            self.args(**kwargs),
            self.logger,
        )


class TestSetupFromArgsMessengerSelection(SetupFromArgsMixin):
    def test_no_messengers_at_all_is_an_error(self):
        with pytest.raises(ValueError):
            self.setup()

    def test_a_single_messenger_is_returned_on_its_own(self):
        # A joint likelihood over one messenger would only add overhead.
        likelihood = self.setup(messengers=["em"])
        assert isinstance(likelihood, StubEMLikelihood)
        assert not isinstance(likelihood, MultiMessengerLikelihood)

    def test_a_single_messenger_has_its_conversion_set_up(self):
        likelihood = self.setup(messengers=["em"])
        assert likelihood.conversion_was_set_up

    def test_two_messengers_are_combined_into_a_joint_likelihood(self):
        likelihood = self.setup(messengers=["gw", "em"])
        assert isinstance(likelihood, MultiMessengerLikelihood)
        assert len(likelihood.likelihoods) == 2

    def test_the_gw_likelihood_is_built_from_its_own_helper(self):
        likelihood = self.setup(messengers=["gw", "em"])
        assert any(isinstance(sub, StubGWLikelihood) for sub in likelihood.likelihoods)

    def test_the_em_likelihood_is_built_from_its_own_helper(self):
        likelihood = self.setup(messengers=["gw", "em"])
        assert any(isinstance(sub, StubEMLikelihood) for sub in likelihood.likelihoods)

    def test_the_helpers_receive_the_dump_the_args_and_the_logger(self):
        self.setup(messengers=["gw", "em"])
        joint_likelihood.setup_gw_kwargs.assert_called_once()
        joint_likelihood.setup_em_kwargs.assert_called_once()
        # The EM helper also needs the priors, to add systematics parameters.
        assert joint_likelihood.setup_em_kwargs.call_args.args[0] is self.priors

    def test_a_gw_run_pins_fixed_parameters_as_delta_functions(self):
        # Marginalisation setup needs every fixed value to be a prior.
        priors = PriorDict()
        priors["x"] = Uniform(0, 1, "x")
        priors["fixed"] = 2.0
        self.setup(messengers=["gw", "em"], priors=priors)
        assert isinstance(priors["fixed"], DeltaFunction)

    def test_a_population_messenger_is_wrapped_in_a_plain_nmma_likelihood(self):
        with patch.object(
            joint_likelihood, "NeutronStarPopulation", return_value=MagicMock()
        ) as population:
            with patch.object(
                joint_likelihood,
                "NMMALikelihood",
                return_value=StubMessengerLikelihood("pop"),
            ):
                likelihood = self.setup(messengers=["em", "pop"])
        population.assert_called_once_with("uniform")
        assert len(likelihood.likelihoods) == 2


class TestSetupFromArgsEoSHandling(SetupFromArgsMixin):
    """How the EOS enters the run depends on whether it is sampled, read
    from a table, or replaced by universal relations."""

    def test_a_constrained_eos_becomes_its_own_likelihood(self):
        likelihood = self.setup(messengers=["gw", "em", "eos"])
        assert any(isinstance(sub, StubEoSLikelihood) for sub in likelihood.likelihoods)

    def test_a_constrained_eos_without_gravitational_waves_fails_to_set_up(self):
        # A constrained sampled EOS records its conversion as the placeholder
        # True, which the no-GW branch then dereferences as if it were an
        # EoSConverter. An EM-plus-EOS run therefore cannot be set up at all.
        # Once the placeholder is resolved before that branch, this should
        # build a two-messenger likelihood instead of raising.
        with pytest.raises(AttributeError):
            self.setup(messengers=["em", "eos"])

    def test_an_unconstrained_eos_is_only_a_conversion(self):
        # With no constraints there is nothing to evaluate, so the EOS
        # contributes a parameter conversion instead of a likelihood term.
        joint_likelihood.setup_eos_kwargs.return_value = {
            "name": "EoS",
            "constraint_dict": {},
            "eos_converter": self.eos_converter,
        }
        likelihood = self.setup(messengers=["gw", "em", "eos"])
        assert not any(
            isinstance(sub, StubEoSLikelihood) for sub in likelihood.likelihoods
        )
        assert "eos" in likelihood.conversion_instructions

    def test_a_tabulated_eos_is_taken_from_the_analysis_modifiers(self):
        likelihood = self.setup(messengers=["gw", "em"], modifiers=["tabulated_eos"])
        joint_likelihood.EoSConverter.assert_called_once()
        assert joint_likelihood.EoSConverter.call_args.args[1] == "tabulated"
        assert "eos" in likelihood.conversion_instructions

    def test_sampled_tidal_deformabilities_use_universal_relations(self):
        priors = PriorDict()
        priors["lambda_1"] = Uniform(0, 1000, "lambda_1")
        priors["lambda_2"] = Uniform(0, 1000, "lambda_2")
        self.setup(messengers=["gw", "em"], priors=priors)
        assert joint_likelihood.EoSConverter.call_args.args[1] == "qur"

    def test_an_eos_without_gravitational_waves_only_computes_macro_parameters(self):
        # Without a GW likelihood there are no component masses to map onto
        # the EOS, so only the bulk neutron-star properties are computed.
        self.setup(messengers=["em"], modifiers=["tabulated_eos"])
        assert (
            self.eos_converter.parameter_conversion
            is self.eos_converter.compute_macro_parameters
        )

    def test_an_eos_alongside_gravitational_waves_keeps_the_full_conversion(self):
        self.setup(messengers=["gw", "em"], modifiers=["tabulated_eos"])
        assert (
            self.eos_converter.parameter_conversion
            != self.eos_converter.compute_macro_parameters
        )

    def test_a_sampled_eos_takes_precedence_over_a_tabulated_one(self):
        self.setup(messengers=["gw", "em", "eos"], modifiers=["tabulated_eos"])
        joint_likelihood.EoSConverter.assert_not_called()


class TestSetupFromArgsConversionInstructions(SetupFromArgsMixin):
    def test_a_hubble_run_records_the_cosmology(self):
        likelihood = self.setup(
            messengers=["gw", "em"], modifiers=["Hubble"], cosmology="Planck15"
        )
        assert likelihood.conversion_instructions["cosmo"] == "Planck15"

    def test_without_the_hubble_modifier_no_cosmology_is_recorded(self):
        likelihood = self.setup(messengers=["gw", "em"])
        assert "cosmo" not in likelihood.conversion_instructions

    def test_sampled_wind_ejecta_switch_on_the_ejecta_conversion(self):
        priors = PriorDict()
        priors["log10_mej_wind"] = Uniform(-3, -1, "log10_mej_wind")
        likelihood = self.setup(messengers=["gw", "em"], priors=priors)
        assert likelihood.conversion_instructions["ejecta"]

    def test_sampled_dynamical_ejecta_switch_on_the_ejecta_conversion(self):
        priors = PriorDict()
        priors["log10_mej_dyn"] = Uniform(-3, -1, "log10_mej_dyn")
        likelihood = self.setup(messengers=["gw", "em"], priors=priors)
        assert likelihood.conversion_instructions["ejecta"]

    def test_the_ejecta_conversion_can_be_requested_explicitly(self):
        likelihood = self.setup(messengers=["gw", "em"], ejecta_conversion=True)
        assert likelihood.conversion_instructions["ejecta"]

    def test_without_ejecta_parameters_no_ejecta_conversion_is_added(self):
        likelihood = self.setup(messengers=["gw", "em"])
        assert "ejecta" not in likelihood.conversion_instructions

    def test_a_single_messenger_still_gains_the_eos_conversion(self):
        likelihood = self.setup(messengers=["em"], modifiers=["tabulated_eos"])
        assert self.eos_converter in likelihood.conv_functions

    def test_the_setup_is_logged(self):
        self.setup(messengers=["gw", "em"])
        assert self.logger.info.called
