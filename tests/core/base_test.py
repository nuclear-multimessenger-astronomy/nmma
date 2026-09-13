import inspect
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import bilby
import h5py
import numpy as np
import pandas as pd
from bilby.core.prior import (
    ConditionalPriorDict,
    Constraint,
    Interped,
    MultivariateGaussian,
    PriorDict,
    Uniform,
)

from nmma.core import base


class StubSubModel:
    """A minimal stand-in for a messenger likelihood: the only things
    NMMALikelihood asks of a sub-model are log_likelihood and, optionally,
    noise_log_likelihood."""

    def __init__(self, log_likelihood_value=-1.0, noise_log_likelihood_value=-5.0):
        self.log_likelihood_value = log_likelihood_value
        self.noise_log_likelihood_value = noise_log_likelihood_value
        self.seen_parameters = None

    def log_likelihood(self, parameters):
        self.seen_parameters = parameters
        return self.log_likelihood_value

    def noise_log_likelihood(self):
        return self.noise_log_likelihood_value


class StubSubModelWithoutNoise:
    def log_likelihood(self, parameters):
        return -1.0


def simple_priors():
    priors = PriorDict()
    priors["x"] = Uniform(0, 1, "x")
    return priors


class TestInitialisationArgsFromSignatureAndNamespace(unittest.TestCase):
    @staticmethod
    def example(required, first=1, second=2):
        pass

    def test_signature_defaults_are_returned(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace()
        )
        self.assertEqual(kwargs, {"first": 1, "second": 2})

    def test_parameters_without_a_default_are_not_invented(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace()
        )
        self.assertNotIn("required", kwargs)

    def test_namespace_values_override_the_defaults(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(first=10)
        )
        self.assertEqual(kwargs["first"], 10)

    def test_a_required_parameter_is_filled_from_the_namespace(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(required="value")
        )
        self.assertEqual(kwargs["required"], "value")

    def test_a_none_in_the_namespace_does_not_override_the_default(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(first=None)
        )
        self.assertEqual(kwargs["first"], 1)

    def test_a_prefix_lets_a_shorthand_argument_reach_the_parameter(self):
        # this is how e.g. --tmin reaches a "min" constructor parameter
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(prefix_first=10), ["prefix_"]
        )
        self.assertEqual(kwargs["first"], 10)

    def test_the_unprefixed_name_is_tried_after_the_given_prefixes(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(first=10), ["prefix_"]
        )
        self.assertEqual(kwargs["first"], 10)

    def test_the_first_matching_prefix_wins(self):
        kwargs = base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(a_first=1, b_first=2), ["a_", "b_"]
        )
        self.assertEqual(kwargs["first"], 1)

    def test_it_maps_onto_a_real_class_constructor(self):
        class Example:
            def __init__(self, model, filters=None, nlive=2048):
                pass

        kwargs = base.initialisation_args_from_signature_and_namespace(
            Example, Namespace(model="Me2017", nlive=32, unrelated="ignored")
        )
        self.assertEqual(kwargs, {"model": "Me2017", "filters": None, "nlive": 32})
        self.assertNotIn("unrelated", kwargs)

    def test_the_prefix_default_accumulates_across_calls(self):
        # prefixes defaults to a mutable list and the function appends "" to
        # it on every call, so the shared default grows without bound and a
        # caller-supplied list is modified in place. This test documents that
        # known gap rather than asserting it is correct.
        signature = inspect.signature(base.initialisation_args_from_signature_and_namespace)
        shared_default = signature.parameters["prefixes"].default
        before = len(shared_default)
        base.initialisation_args_from_signature_and_namespace(self.example, Namespace())
        self.assertEqual(len(shared_default), before + 1)

        caller_list = ["prefix_"]
        base.initialisation_args_from_signature_and_namespace(
            self.example, Namespace(), caller_list
        )
        self.assertEqual(caller_list, ["prefix_", ""])


class TestNMMALikelihood(unittest.TestCase):
    def setUp(self):
        self.sub_model = StubSubModel()
        self.priors = simple_priors()
        self.likelihood = base.NMMALikelihood(self.sub_model, self.priors)

    def test_the_sub_model_is_stored(self):
        self.assertIs(self.likelihood.sub_model, self.sub_model)

    def test_repr_names_the_sub_model(self):
        self.assertIn("NMMALikelihood", repr(self.likelihood))
        self.assertIn("StubSubModel", repr(self.likelihood))

    def test_the_noise_log_likelihood_is_cached_from_the_sub_model(self):
        self.assertEqual(self.likelihood.noise_log_likelihood(), -5.0)

    def test_a_sub_model_without_a_noise_likelihood_gives_zero(self):
        likelihood = base.NMMALikelihood(StubSubModelWithoutNoise(), simple_priors())
        self.assertEqual(likelihood.noise_log_likelihood(), 0.0)

    def test_log_likelihood_delegates_to_the_sub_model(self):
        self.assertEqual(self.likelihood.log_likelihood({"x": 0.5}), -1.0)
        self.assertEqual(self.sub_model.seen_parameters, {"x": 0.5})

    def test_calling_the_likelihood_exponentiates_the_log_likelihood(self):
        self.assertAlmostEqual(self.likelihood({"x": 0.5}), np.exp(-1.0))

    def test_a_non_finite_sub_likelihood_is_clipped(self):
        likelihood = base.NMMALikelihood(StubSubModel(np.nan), simple_priors())
        self.assertEqual(
            likelihood.log_likelihood({"x": 0.5}), np.nan_to_num(-np.inf)
        )

    def test_an_infinite_sub_likelihood_is_clipped(self):
        likelihood = base.NMMALikelihood(StubSubModel(-np.inf), simple_priors())
        self.assertEqual(
            likelihood.log_likelihood({"x": 0.5}), np.nan_to_num(-np.inf)
        )

    def test_identity_conversion_returns_its_input(self):
        parameters = {"x": 0.5}
        self.assertIs(self.likelihood.identity_conversion(parameters), parameters)

    def test_no_conversion_functions_leaves_the_parameters_alone(self):
        self.assertEqual(self.likelihood.parameter_conversion({"x": 0.5}), {"x": 0.5})

    def test_conversion_functions_are_applied_in_reverse_order(self):
        # the "main" conversions are appended last and must run first
        calls = []
        self.likelihood.conv_functions = [
            lambda p: (calls.append("added_first"), p)[1],
            lambda p: (calls.append("added_last"), p)[1],
        ]
        self.likelihood.parameter_conversion({"x": 0.5})
        self.assertEqual(calls, ["added_last", "added_first"])

    def test_posterior_conversion_uses_the_same_chain(self):
        self.likelihood.conv_functions = [lambda p: dict(p, converted=True)]
        self.assertTrue(self.likelihood.posterior_conversion({"x": 0.5})["converted"])

    def test_setup_parameter_conversion_adds_the_cosmology_conversion(self):
        priors = simple_priors()
        priors["Hubble_constant"] = Uniform(50, 100, "Hubble_constant")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        likelihood.setup_parameter_conversion()
        self.assertIn(base.cosmology_to_distance, likelihood.conv_functions)

    def test_no_cosmology_conversion_without_a_hubble_prior(self):
        self.likelihood.setup_parameter_conversion()
        self.assertEqual(self.likelihood.conv_functions, [])

    def test_sanity_checks_pass_by_default(self):
        self.assertTrue(self.likelihood.sanity_checks())

    def test_final_diagnostics_is_forwarded_to_the_sub_model(self):
        self.sub_model.final_diagnostics = MagicMock(return_value="figure")
        result = self.likelihood.final_diagnostics({"x": 0.5}, Namespace())
        self.assertEqual(result, "figure")

    def test_final_diagnostics_is_silent_when_the_sub_model_has_none(self):
        self.assertIsNone(self.likelihood.final_diagnostics({"x": 0.5}, Namespace()))

    def test_post_process_bestfit_converts_the_best_fit_parameters(self):
        posterior = pd.DataFrame({"x": [0.1, 0.9], "log_likelihood": [1.0, 3.0]})
        self.sub_model.final_diagnostics = MagicMock(return_value="figure")
        with patch.object(base, "read_bestfit_from_posterior") as mock_read:
            mock_read.return_value = posterior.loc[1].to_dict()
            result = self.likelihood.post_process_bestfit(Namespace())
        self.assertEqual(result, "figure")
        self.assertAlmostEqual(
            self.sub_model.final_diagnostics.call_args[0][0]["x"], 0.9
        )


class TestNMMALikelihoodConstraints(unittest.TestCase):
    def test_constraints_are_split_out_of_the_priors(self):
        priors = simple_priors()
        priors["con"] = Constraint(0, 1, "con")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        self.assertEqual(list(likelihood.constraints), ["con"])

    def test_the_full_prior_dict_is_still_stored(self):
        priors = simple_priors()
        priors["con"] = Constraint(0, 1, "con")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        self.assertIs(likelihood.priors, priors)

    def test_no_constraints_gives_an_empty_mapping(self):
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        self.assertEqual(likelihood.constraints, {})

    def test_a_bare_constraint_is_accepted(self):
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        likelihood.constraints = Constraint(0, 1, "con")
        self.assertEqual(list(likelihood.constraints), ["con"])

    def test_a_plain_dict_of_constraints_is_accepted(self):
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        likelihood.constraints = {"con": Constraint(0, 1, "con")}
        self.assertEqual(list(likelihood.constraints), ["con"])

    def test_a_plain_dict_of_non_constraints_is_rejected(self):
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        with self.assertRaises(AssertionError):
            likelihood.constraints = {"x": Uniform(0, 1, "x")}

    def test_a_satisfied_constraint_evaluates_to_true(self):
        priors = simple_priors()
        priors["con"] = Constraint(0, 1, "con")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        self.assertTrue(likelihood.evaluate_constraints({"con": 0.5}))

    def test_a_violated_constraint_evaluates_to_false(self):
        priors = simple_priors()
        priors["con"] = Constraint(0, 1, "con")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        self.assertFalse(likelihood.evaluate_constraints({"con": 5.0}))

    def test_no_constraints_evaluate_to_true(self):
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        self.assertTrue(likelihood.evaluate_constraints({"x": 0.5}))

    def test_a_violated_constraint_short_circuits_the_log_likelihood(self):
        priors = simple_priors()
        priors["con"] = Constraint(0, 1, "con")
        sub_model = StubSubModel()
        likelihood = base.NMMALikelihood(sub_model, priors)
        value = likelihood.log_likelihood({"x": 0.5, "con": 5.0})
        self.assertEqual(value, np.nan_to_num(-np.inf))
        self.assertIsNone(sub_model.seen_parameters)

    def test_a_failed_sanity_check_short_circuits_the_log_likelihood(self):
        sub_model = StubSubModel()
        likelihood = base.NMMALikelihood(sub_model, simple_priors())
        likelihood.sanity_checks = lambda: False
        self.assertEqual(
            likelihood.log_likelihood({"x": 0.5}), np.nan_to_num(-np.inf)
        )
        self.assertIsNone(sub_model.seen_parameters)


class TestCheckParameterEquivalencies(unittest.TestCase):
    def setUp(self):
        self.likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())

    def test_a_single_inclination_parameter_is_accepted(self):
        self.likelihood.check_parameter_equivalencies(["theta_jn", "chirp_mass"])

    def test_two_equivalent_inclination_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.likelihood.check_parameter_equivalencies(["theta_jn", "KNtheta"])

    def test_all_inclination_spellings_are_covered(self):
        for name in ["inclination_EM", "KNtheta", "cos_theta_jn", "thetaObs"]:
            with self.assertRaises(ValueError, msg=name):
                self.likelihood.check_parameter_equivalencies(["theta_jn", name])

    def test_two_distance_parameters_are_allowed(self):
        # two of the three fixes the cosmology, which is a legitimate setup
        self.likelihood.check_parameter_equivalencies(
            ["redshift", "Hubble_constant"]
        )

    def test_three_distance_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.likelihood.check_parameter_equivalencies(
                ["redshift", "luminosity_distance", "Hubble_constant"]
            )

    def test_two_mass_parameters_are_allowed(self):
        self.likelihood.check_parameter_equivalencies(["chirp_mass", "mass_ratio"])

    def test_three_mass_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            self.likelihood.check_parameter_equivalencies(
                ["chirp_mass", "mass_ratio", "mass_1"]
            )

    def test_the_check_runs_when_the_priors_are_set(self):
        priors = PriorDict()
        priors["theta_jn"] = Uniform(0, np.pi, "theta_jn")
        priors["KNtheta"] = Uniform(0, 90, "KNtheta")
        with self.assertRaises(ValueError):
            base.NMMALikelihood(StubSubModel(), priors)


class TestNMMADummyPrior(unittest.TestCase):
    def test_the_setup_properties_are_stored(self):
        prior = base.NMMADummyPrior({"file": "eos.h5"})
        self.assertEqual(prior.setup_props, {"file": "eos.h5"})
        self.assertEqual(prior.name, "NMMADummyPrior")

    def test_from_repr_parses_a_literal(self):
        prior = base.NMMADummyPrior.from_repr("{'file': 'eos.h5'}")
        self.assertEqual(prior.setup_props, {"file": "eos.h5"})

    def test_it_can_be_read_back_out_of_a_prior_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prior_file = Path(tmp_dir) / "test.prior"
            prior_file.write_text(
                "x = Uniform(minimum=0, maximum=1, name='x')\n"
                "eos_h5 = NMMADummyPrior({'file': 'eos.h5'})\n"
            )
            with patch.dict(
                bilby.core.prior.__dict__, {"NMMADummyPrior": base.NMMADummyPrior}
            ):
                priors = PriorDict(str(prior_file))
        self.assertIsInstance(priors["eos_h5"], base.NMMADummyPrior)


class TestAdjustHubblePrior(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.hubble_values = np.linspace(50.0, 100.0, 51)
        self.weights = np.exp(-((self.hubble_values - 70.0) ** 2) / 50.0)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def write_weight_file(self, with_header):
        path = self.tmp_dir / "hubble.dat"
        if with_header:
            pd.DataFrame(
                {"Hubble": self.hubble_values, "prior_weight": self.weights}
            ).to_csv(path, sep=" ", index=False)
        else:
            np.savetxt(path, np.column_stack([self.hubble_values, self.weights]))
        return str(path)

    def test_a_headed_weight_file_becomes_an_interped_prior(self):
        priors = adjusted = base.adjust_hubble_prior(
            PriorDict(), Namespace(Hubble_weight=self.write_weight_file(True))
        )
        self.assertIsInstance(adjusted["Hubble_constant"], Interped)
        self.assertAlmostEqual(priors["Hubble_constant"].minimum, 50.0)
        self.assertAlmostEqual(priors["Hubble_constant"].maximum, 100.0)

    def test_a_bare_two_column_weight_file_is_also_accepted(self):
        adjusted = base.adjust_hubble_prior(
            PriorDict(), Namespace(Hubble_weight=self.write_weight_file(False))
        )
        self.assertIsInstance(adjusted["Hubble_constant"], Interped)

    def test_the_interpolated_prior_peaks_where_the_weights_do(self):
        adjusted = base.adjust_hubble_prior(
            PriorDict(), Namespace(Hubble_weight=self.write_weight_file(True))
        )
        prior = adjusted["Hubble_constant"]
        self.assertGreater(prior.prob(70.0), prior.prob(50.0))

    def test_an_existing_hubble_prior_is_overwritten(self):
        priors = PriorDict()
        priors["Hubble_constant"] = Uniform(50, 100, "Hubble_constant")
        adjusted = base.adjust_hubble_prior(
            priors, Namespace(Hubble_weight=self.write_weight_file(True))
        )
        self.assertIsInstance(adjusted["Hubble_constant"], Interped)

    def test_no_weight_file_leaves_the_priors_untouched(self):
        priors = PriorDict()
        self.assertEqual(
            base.adjust_hubble_prior(priors, Namespace(Hubble_weight=None)), priors
        )

    def test_the_cosmology_is_set_when_hubble_sampling_is_requested(self):
        with patch.object(base, "set_cosmology") as mock_set:
            base.adjust_hubble_prior(
                PriorDict(), Namespace(Hubble=True, Hubble_weight=None, cosmology="Planck15")
            )
        mock_set.assert_called_once_with("Planck15")

    def test_the_cosmology_is_set_when_a_hubble_prior_is_present(self):
        priors = PriorDict()
        priors["Hubble_constant"] = Uniform(50, 100, "Hubble_constant")
        with patch.object(base, "set_cosmology") as mock_set:
            base.adjust_hubble_prior(priors, Namespace(Hubble_weight=None))
        mock_set.assert_called_once()


class TestH5ToMultivarPrior(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.h5_path = self.tmp_dir / "eos.h5"
        rng = np.random.default_rng(0)
        with h5py.File(self.h5_path, "w") as f:
            f.create_dataset("TOV_mass", data=rng.normal(2.1, 0.1, 500))
            f.create_dataset("R_14", data=rng.normal(12.0, 0.5, 500))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_every_dataset_becomes_a_multivariate_gaussian_prior(self):
        priors = base.h5_to_multivar_prior(str(self.h5_path))
        self.assertEqual(set(priors), {"TOV_mass", "R_14"})
        for prior in priors.values():
            self.assertIsInstance(prior, MultivariateGaussian)

    def test_the_result_is_at_least_a_conditional_prior_dict(self):
        priors = base.h5_to_multivar_prior(str(self.h5_path))
        self.assertIsInstance(priors, ConditionalPriorDict)

    def test_an_existing_conditional_prior_dict_is_not_downgraded(self):
        priors = ConditionalPriorDict()
        priors["x"] = Uniform(0, 1, "x")
        result = base.h5_to_multivar_prior(str(self.h5_path), priors)
        self.assertIsInstance(result, ConditionalPriorDict)
        self.assertIn("x", result)

    def test_existing_priors_are_preserved(self):
        priors = {"x": Uniform(0, 1, "x")}
        result = base.h5_to_multivar_prior(str(self.h5_path), priors)
        self.assertIn("x", result)

    def test_the_fitted_distribution_recovers_the_sample_mean(self):
        priors = base.h5_to_multivar_prior(str(self.h5_path))
        with h5py.File(self.h5_path, "r") as f:
            expected = np.mean(f["TOV_mass"][:])
        distribution = priors["TOV_mass"].dist
        index = distribution.names.index("TOV_mass")
        self.assertAlmostEqual(distribution.mus[0][index], expected, places=6)

    def test_a_namespace_path_is_accepted(self):
        priors = base.h5_to_multivar_prior(
            Namespace(**{"h5 file path": str(self.h5_path)})
        )
        self.assertEqual(set(priors), {"TOV_mass", "R_14"})


class TestAdjustPriorsForNmma(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_priors_without_dummies_are_returned_unchanged(self):
        priors = simple_priors()
        self.assertEqual(base.adjust_priors_for_nmma(priors), priors)

    def test_a_prior_file_path_is_read(self):
        prior_file = self.tmp_dir / "test.prior"
        prior_file.write_text("x = Uniform(minimum=0, maximum=1, name='x')\n")
        priors = base.adjust_priors_for_nmma(str(prior_file))
        self.assertIsInstance(priors, PriorDict)
        self.assertIn("x", priors)

    def test_an_h5_dummy_prior_is_replaced_by_multivariate_gaussians(self):
        h5_path = self.tmp_dir / "eos.h5"
        rng = np.random.default_rng(0)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("TOV_mass", data=rng.normal(2.1, 0.1, 200))
            f.create_dataset("R_14", data=rng.normal(12.0, 0.5, 200))
        priors = PriorDict()
        priors["eos_h5"] = base.NMMADummyPrior(str(h5_path))
        adjusted = base.adjust_priors_for_nmma(priors)
        self.assertNotIn("eos_h5", adjusted)
        self.assertIn("TOV_mass", adjusted)

    def test_a_hubble_dummy_prior_is_replaced_by_an_interped_prior(self):
        weight_file = self.tmp_dir / "hubble.dat"
        hubble = np.linspace(50.0, 100.0, 51)
        np.savetxt(weight_file, np.column_stack([hubble, np.ones_like(hubble)]))
        priors = PriorDict()
        priors["hubble_weighting"] = base.NMMADummyPrior(
            Namespace(Hubble_weight=str(weight_file), Hubble=False)
        )
        adjusted = base.adjust_priors_for_nmma(priors)
        self.assertNotIn("hubble_weighting", adjusted)
        self.assertIsInstance(adjusted["Hubble_constant"], Interped)

    def test_the_replacement_is_logged_when_a_logger_is_given(self):
        weight_file = self.tmp_dir / "hubble.dat"
        hubble = np.linspace(50.0, 100.0, 51)
        np.savetxt(weight_file, np.column_stack([hubble, np.ones_like(hubble)]))
        priors = PriorDict()
        priors["hubble_weighting"] = base.NMMADummyPrior(
            Namespace(Hubble_weight=str(weight_file), Hubble=False)
        )
        logger = MagicMock()
        base.adjust_priors_for_nmma(priors, logger=logger)
        self.assertTrue(logger.info.called)


class TestCheckPriorsAndLikelihoodForNmma(unittest.TestCase):
    def test_constraints_left_in_the_priors_are_moved_to_the_likelihood(self):
        priors = simple_priors()
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        priors["con"] = Constraint(0, 1, "con")
        priors, likelihood = base.check_priors_and_likelihood_for_nmma(priors, likelihood)
        self.assertNotIn("con", priors)
        self.assertIn("con", likelihood.constraints)

    def test_the_sampling_priors_survive(self):
        priors = simple_priors()
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        priors, _ = base.check_priors_and_likelihood_for_nmma(priors, likelihood)
        self.assertIn("x", priors)

    def test_the_final_parameter_conversion_is_set_up(self):
        priors = simple_priors()
        priors["Hubble_constant"] = Uniform(50, 100, "Hubble_constant")
        likelihood = base.NMMALikelihood(StubSubModel(), priors)
        _, likelihood = base.check_priors_and_likelihood_for_nmma(priors, likelihood)
        self.assertIn(base.cosmology_to_distance, likelihood.conv_functions)

    def test_the_duplicate_key_branch_is_unreachable(self):
        # The guard compares len(set(keys)) against len(keys) for the same
        # dict, and dictionary keys are unique by construction, so the two
        # are always equal and the branch that resets the conversion function
        # never runs. This test documents that known gap rather than
        # asserting it is correct.
        priors = simple_priors()
        likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        original_conversion = priors.conversion_function
        _, likelihood = base.check_priors_and_likelihood_for_nmma(priors, likelihood)
        self.assertIs(priors.conversion_function, original_conversion)
        self.assertEqual(likelihood.conv_functions, [])


class TestBilbySampling(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        self.priors = simple_priors()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def make_args(self, **overrides):
        args = Namespace(
            sampler="dynesty",
            outdir=str(self.tmp_dir),
            label="test",
            nlive=32,
            sampling_seed=42,
            soft_init=False,
            cpus=1,
            skip_sampling=False,
            bestfit=False,
            plot=False,
            sampler_kwargs={},
        )
        args.__dict__.update(overrides)
        return args

    def make_result(self):
        result = MagicMock()
        result.posterior = pd.DataFrame({"x": [0.1, 0.9], "fixed": [1.0, 1.0]})
        return result

    def test_the_sampler_is_called_with_the_parsed_settings(self):
        with patch.object(base, "run_sampler", return_value=self.make_result()) as mock:
            base.bilby_sampling(self.likelihood, self.priors, self.make_args())
        kwargs = mock.call_args.kwargs
        self.assertEqual(kwargs["sampler"], "dynesty")
        self.assertEqual(kwargs["nlive"], 32)
        self.assertEqual(kwargs["seed"], 42)
        self.assertEqual(kwargs["label"], "test")
        self.assertFalse(kwargs["save"])

    def test_extra_sampler_kwargs_are_forwarded(self):
        args = self.make_args(sampler_kwargs={"walks": 50})
        with patch.object(base, "run_sampler", return_value=self.make_result()) as mock:
            base.bilby_sampling(self.likelihood, self.priors, args)
        self.assertEqual(mock.call_args.kwargs["walks"], 50)

    def test_a_dictionary_of_arguments_is_filled_in_from_the_defaults(self):
        settings = {"sampler": "dynesty", "outdir": str(self.tmp_dir), "label": "test"}
        # the dictionary path rebuilds a full Namespace from the analysis
        # parser, which reads sys.argv unless it is given something else
        with (
            patch.object(sys, "argv", ["nmma"]),
            patch.object(base, "run_sampler", return_value=self.make_result()) as mock,
        ):
            base.bilby_sampling(self.likelihood, self.priors, settings)
        self.assertEqual(mock.call_args.kwargs["nlive"], 2048)

    def test_the_dictionary_path_inherits_the_surrounding_command_line(self):
        # The defaults come from parsing sys.argv, so any unrelated command
        # line in the calling process is parsed too and an argument the
        # analysis parser does not know aborts the run. This test documents
        # that known gap rather than asserting it is correct.
        settings = {"sampler": "dynesty", "outdir": str(self.tmp_dir), "label": "test"}
        with (
            patch.object(sys, "argv", ["nmma", "--not-an-nmma-argument"]),
            patch.object(base, "run_sampler", return_value=self.make_result()),
        ):
            with self.assertRaises(SystemExit):
                base.bilby_sampling(self.likelihood, self.priors, settings)

    def test_reactive_sampling_drops_the_live_point_count(self):
        args = self.make_args(sampler="ultranest", reactive_sampling=True)
        with patch.object(base, "run_sampler", return_value=self.make_result()) as mock:
            base.bilby_sampling(self.likelihood, self.priors, args)
        self.assertIsNone(mock.call_args.kwargs["nlive"])

    def test_reactive_sampling_is_rejected_for_other_samplers(self):
        args = self.make_args(sampler="dynesty", reactive_sampling=True)
        with patch.object(base, "run_sampler", return_value=self.make_result()):
            with self.assertRaises(ValueError):
                base.bilby_sampling(self.likelihood, self.priors, args)

    def test_skip_sampling_caps_the_iterations_per_sampler(self):
        for sampler, key in [
            ("pymultinest", "max_iter"),
            ("ultranest", "niter"),
            ("dynesty", "maxiter"),
        ]:
            args = self.make_args(sampler=sampler, skip_sampling=True)
            with patch.object(
                base, "run_sampler", return_value=self.make_result()
            ) as mock:
                base.bilby_sampling(self.likelihood, self.priors, args)
            self.assertEqual(mock.call_args.kwargs[key], 1, msg=sampler)

    def test_non_zero_mpi_ranks_do_no_post_processing(self):
        result = self.make_result()
        with patch.object(base, "run_sampler", return_value=result):
            returned = base.bilby_sampling(
                self.likelihood, self.priors, self.make_args(), rank=1
            )
        self.assertIsNone(returned)
        result.save_to_file.assert_not_called()

    def test_rank_zero_saves_and_plots_the_result(self):
        result = self.make_result()
        with patch.object(base, "run_sampler", return_value=result):
            returned = base.bilby_sampling(self.likelihood, self.priors, self.make_args())
        self.assertIs(returned, result)
        result.save_to_file.assert_called_once()
        result.save_posterior_samples.assert_called_once()
        result.plot_corner.assert_called_once()

    def test_injection_parameters_are_restricted_to_the_varying_columns(self):
        result = self.make_result()
        with patch.object(base, "run_sampler", return_value=result):
            base.bilby_sampling(
                self.likelihood,
                self.priors,
                self.make_args(),
                injection_parameters={"x": 0.5, "fixed": 1.0, "absent": 2.0},
            )
        plotted = result.plot_corner.call_args[0][0]
        self.assertEqual(plotted, {"x": 0.5})

    def test_the_best_fit_post_processing_runs_when_requested(self):
        result = self.make_result()
        self.likelihood.post_process_bestfit = MagicMock()
        with patch.object(base, "run_sampler", return_value=result):
            base.bilby_sampling(
                self.likelihood, self.priors, self.make_args(bestfit=True)
            )
        self.likelihood.post_process_bestfit.assert_called_once()

    def test_a_corner_plot_failure_is_retried_without_latex_labels(self):
        result = self.make_result()
        result.plot_corner.side_effect = [RuntimeError("bad label"), None]
        with patch.object(base, "run_sampler", return_value=result):
            base.bilby_sampling(self.likelihood, self.priors, self.make_args())
        self.assertEqual(result.plot_corner.call_count, 2)
        self.assertIsNone(result.parameter_labels_with_unit)


class TestMultiAnalysisLoop(unittest.TestCase):
    def setUp(self):
        self.likelihood = base.NMMALikelihood(StubSubModel(), simple_priors())
        self.priors = simple_priors()
        self.seen_args = []

    def analysis_setup(self, args):
        self.seen_args.append(args)
        return self.priors, self.likelihood, None

    def make_args(self, **overrides):
        args = Namespace(
            sampler="pymultinest", label="run", outdir="outdir", nlive=32, verbose=True
        )
        args.__dict__.update(overrides)
        return args

    def run_loop(self, args):
        with (
            patch.object(base, "bilby_sampling", return_value="result") as mock_sampling,
            patch.object(
                base,
                "check_priors_and_likelihood_for_nmma",
                side_effect=lambda p, l: (p, l),
            ),
        ):
            out = base.multi_analysis_loop(args, self.analysis_setup)
        return out, mock_sampling

    def test_a_plain_run_calls_the_sampler_once(self):
        out, mock_sampling = self.run_loop(self.make_args())
        self.assertEqual(out, "result")
        self.assertEqual(mock_sampling.call_count, 1)
        self.assertEqual(len(self.seen_args), 1)

    def test_a_single_key_multi_sweeps_over_the_values(self):
        args = self.make_args(multi={"nlive": [16, 32, 64]})
        _, mock_sampling = self.run_loop(args)
        self.assertEqual(mock_sampling.call_count, 3)
        self.assertEqual([a.nlive for a in self.seen_args], [16, 32, 64])
        self.assertEqual(
            [a.label for a in self.seen_args], ["run_0", "run_1", "run_2"]
        )

    def test_a_named_multi_applies_each_set_of_changes(self):
        args = self.make_args(
            multi={"low": {"nlive": 16}, "high": {"nlive": 64, "sampler": "dynesty"}}
        )
        _, mock_sampling = self.run_loop(args)
        self.assertEqual(mock_sampling.call_count, 2)
        self.assertEqual([a.label for a in self.seen_args], ["run_low", "run_high"])
        self.assertEqual(self.seen_args[1].sampler, "dynesty")

    def test_an_unknown_key_in_a_named_multi_is_rejected(self):
        args = self.make_args(
            multi={"bad": {"not_an_argument": 1}, "good": {"nlive": 16}}
        )
        with self.assertRaises(KeyError):
            self.run_loop(args)

    def test_a_named_multi_with_a_single_run_is_read_as_a_value_sweep(self):
        # The two multi shapes are told apart by the number of top-level
        # entries, so a named specification that happens to describe exactly
        # one run takes the single-key branch instead: the run name is used
        # as the argument to vary and its change dictionary is iterated over
        # as a list of values, which yields the change's keys. This test
        # documents that known gap rather than asserting it is correct.
        args = self.make_args(multi={"label": {"nlive": 16}})
        _, mock_sampling = self.run_loop(args)
        self.assertEqual(mock_sampling.call_count, 1)
        self.assertEqual(self.seen_args[0].label, "run_0")
        self.assertEqual(self.seen_args[0].nlive, 32)

    def test_the_original_arguments_are_not_modified_by_a_sweep(self):
        args = self.make_args(multi={"nlive": [16, 32]})
        self.run_loop(args)
        self.assertEqual(args.nlive, 32)
        self.assertEqual(args.label, "run")

    def test_a_matrix_runs_the_cross_product(self):
        args = self.make_args(matrix={"nlive": [16, 32], "sampler": ["dynesty", "ultranest"]})
        _, mock_sampling = self.run_loop(args)
        self.assertEqual(mock_sampling.call_count, 4)
        self.assertEqual(
            [(a.nlive, a.sampler) for a in self.seen_args],
            [(16, "dynesty"), (16, "ultranest"), (32, "dynesty"), (32, "ultranest")],
        )

    def test_matrix_labels_carry_every_varied_value(self):
        args = self.make_args(matrix={"nlive": [16, 32]})
        self.run_loop(args)
        self.assertEqual([a.label for a in self.seen_args], ["run_16", "run_32"])

    def test_an_unknown_key_in_a_matrix_is_rejected(self):
        args = self.make_args(matrix={"not_an_argument": [1, 2]})
        with self.assertRaises(KeyError):
            self.run_loop(args)

    def test_a_long_matrix_value_currently_breaks_the_label_shortening(self):
        # When the formatted value exceeds 20 characters the label falls back
        # to "<key>_<index>", but it indexes args.matrix.keys()/.values()
        # directly and those dictionary views are not subscriptable. This
        # test documents that known gap rather than asserting it is correct.
        args = self.make_args(matrix={"outdir": ["a" * 40, "b" * 40]})
        with self.assertRaises(TypeError):
            self.run_loop(args)

    def test_multi_takes_precedence_over_matrix(self):
        args = self.make_args(multi={"nlive": [16]}, matrix={"nlive": [32, 64]})
        _, mock_sampling = self.run_loop(args)
        self.assertEqual(mock_sampling.call_count, 1)
        self.assertEqual(self.seen_args[0].nlive, 16)


if __name__ == "__main__":
    unittest.main()
