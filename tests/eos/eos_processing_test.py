import contextlib
import json
import pickle
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import joblib
import keras
import numpy as np

from nmma.eos import eos_processing


def write_macro_eos_file(path, scale=1.0, num=20):
    """Write a monotonic mass-radius-lambda table in the layout the EOS
    machinery expects: radius, mass, tidal deformability."""
    masses = np.linspace(0.5, 2.0 * scale, num)
    radii = 12.0 * scale - 0.3 * masses
    lambdas = 1000.0 * np.exp(-2.0 * masses)
    np.savetxt(path, np.column_stack([radii, masses, lambdas]))
    return masses, radii, lambdas


class IdentityScaler:
    """Stand-in for the scikit-learn scalers the LEC emulators are shipped
    with. Defined at module level so joblib can pickle it."""

    def transform(self, data):
        return np.asarray(data, dtype=float)

    def inverse_transform(self, data):
        return np.asarray(data, dtype=float)


class ConstantEmulator:
    """Emulator stub that returns the same row for every input."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, features):
        return np.tile(self.values, (np.shape(features)[0], 1))


class KerasEmulatorMixin:
    """Builds a tiny keras model on disk so the generator classes can be
    exercised through their real loading path."""

    n_inputs = 5
    n_mass_samples = 4

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.emulator_path = self.tmp_dir / "emulator.keras"
        n_outputs = 2 * self.n_mass_samples + 1
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.n_inputs,)),
                keras.layers.Dense(n_outputs),
            ]
        )
        model.save(self.emulator_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def metadata(self, **extra):
        meta = {
            "emulator_path": str(self.emulator_path),
            "n_mass_samples": self.n_mass_samples,
        }
        meta.update(extra)
        return meta


class TestSetupEoSGenerator(unittest.TestCase):
    """setup_eos_generator is the single dispatch point from a CLI namespace
    or a metadata dict to a concrete emulator class."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def dispatch(self, model_name, class_name, metadata=None):
        metadata = metadata if metadata is not None else {"emulator_path": "path"}
        args = Namespace(emulator_metadata=metadata, micro_eos_model=model_name)
        with patch.object(eos_processing, class_name) as generator_class:
            returned = eos_processing.setup_eos_generator(args)
        return returned, generator_class

    def test_dispatches_each_known_model_name(self):
        cases = {
            "nep": "NEPEoSGenerator",
            "nep-5": "NEP5EoSGenerator",
            "lec": "LECEoSGenerator",
            "lec-7": "LEC7EoSGenerator",
            "lec-13": "LEC13EoSGenerator",
        }
        for model_name, class_name in cases.items():
            returned, generator_class = self.dispatch(model_name, class_name)
            self.assertIs(returned, generator_class.return_value, msg=model_name)
            generator_class.assert_called_once_with({"emulator_path": "path"})

    def test_the_model_name_is_case_insensitive(self):
        returned, generator_class = self.dispatch("NEP-5", "NEP5EoSGenerator")
        self.assertIs(returned, generator_class.return_value)

    def test_an_unknown_model_name_is_reported(self):
        args = Namespace(emulator_metadata={}, micro_eos_model="unsupported")
        with self.assertRaises(ValueError) as context:
            eos_processing.setup_eos_generator(args)
        self.assertIn("unsupported", str(context.exception))

    def test_a_plain_dict_carries_the_model_name_itself(self):
        with patch.object(eos_processing, "LEC13EoSGenerator") as generator_class:
            eos_processing.setup_eos_generator(
                {"micro_eos_model": "lec-13", "emulator_path": "path"}
            )
        generator_class.assert_called_once_with(
            {"micro_eos_model": "lec-13", "emulator_path": "path"}
        )

    def test_metadata_is_read_from_a_json_file(self):
        metadata_path = self.tmp_dir / "metadata.json"
        with open(metadata_path, "w") as stream:
            json.dump({"emulator_path": "from_file"}, stream)
        args = Namespace(emulator_metadata=str(metadata_path), micro_eos_model="nep")
        with patch.object(eos_processing, "NEPEoSGenerator") as generator_class:
            eos_processing.setup_eos_generator(args)
        generator_class.assert_called_once_with({"emulator_path": "from_file"})

    def test_metadata_given_as_a_literal_string_is_evaluated(self):
        args = Namespace(
            emulator_metadata="{'emulator_path': 'inline'}", micro_eos_model="nep"
        )
        with patch.object(eos_processing, "NEPEoSGenerator") as generator_class:
            eos_processing.setup_eos_generator(args)
        generator_class.assert_called_once_with({"emulator_path": "inline"})


class TestEoSGeneratorKerasBackend(KerasEmulatorMixin, unittest.TestCase):
    """The base generator loads a keras emulator and picks a prediction
    method that matches the active backend."""

    def test_loading_selects_a_backend_specific_predict(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["a", "b"]
        )
        expected = {
            "tensorflow": generator.tensorflow_predict,
            "jax": generator.jax_predict,
        }.get(keras.backend.backend())
        if expected is not None:
            self.assertEqual(generator.predict, expected)

    def test_explicit_eos_parameters_override_the_class_attribute(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["a", "b"]
        )
        self.assertEqual(generator.eos_parameters, ["a", "b"])
        self.assertIsNone(eos_processing.EoSGenerator.eos_parameters)

    def test_default_mass_construction_is_equally_spaced(self):
        generator = eos_processing.EoSGenerator(str(self.emulator_path))
        self.assertEqual(generator.n_mass_samples, 30)
        self.assertEqual(generator.decompose_mass_data, generator.equal_distance_masses)

    def test_assemble_eos_params_follows_the_parameter_order(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["b", "a"]
        )
        assembled = generator.assemble_eos_params(
            {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "unused": 0.0}
        )
        np.testing.assert_allclose(assembled, [[3.0, 1.0], [4.0, 2.0]])

    def test_assemble_eos_params_promotes_scalars_to_one_sample(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["a", "b"]
        )
        assembled = generator.assemble_eos_params({"a": 1.0, "b": 2.0})
        self.assertEqual(assembled.shape, (1, 2))

    def test_equal_distance_masses_spans_one_solar_mass_to_mtov(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), n_mass_samples=5
        )
        masses = generator.equal_distance_masses(2.0)
        np.testing.assert_allclose(masses, [1.0, 1.25, 1.5, 1.75, 2.0])

    def test_equal_distance_masses_squeezes_a_batch_of_mtov_values(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), n_mass_samples=5
        )
        masses = generator.equal_distance_masses(np.array([[2.0], [2.4]]))
        self.assertEqual(masses.shape, (2, 5))
        np.testing.assert_allclose(masses[:, 0], [1.0, 1.0])
        np.testing.assert_allclose(masses[:, -1], [2.0, 2.4])

    def test_adjust_format_of_the_base_class_is_a_pass_through(self):
        generator = eos_processing.EoSGenerator(str(self.emulator_path))
        predictions = np.arange(6).reshape(2, 3)
        np.testing.assert_allclose(generator.adjust_format(predictions), predictions)


class TestEoSGeneratorPickleFallback(unittest.TestCase):
    """Anything keras cannot load is retried as a pickle."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.emulator_path = self.tmp_dir / "emulator.pkl"
        with open(self.emulator_path, "wb") as stream:
            pickle.dump(ConstantEmulator(np.arange(7.0)), stream)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_a_pickled_emulator_is_loaded_and_uses_pickle_predict(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["a", "b"], n_mass_samples=3
        )
        self.assertEqual(generator.predict, generator.pickle_predict)
        self.assertIsInstance(generator.emulator, ConstantEmulator)

    def test_the_pickled_emulator_is_called_through_emulate_macro_eos(self):
        generator = eos_processing.EoSGenerator(
            str(self.emulator_path), eos_parameters=["a", "b"], n_mass_samples=3
        )
        predictions = generator.emulate_macro_eos({"a": 1.0, "b": 2.0})
        np.testing.assert_allclose(predictions, [np.arange(7.0)])

    def test_a_path_that_is_neither_a_model_nor_a_pickle_fails(self):
        broken = self.tmp_dir / "broken.keras"
        broken.write_text("not a model")
        with self.assertRaises(Exception):
            eos_processing.EoSGenerator(str(broken))


class TestNEPEoSGenerator(KerasEmulatorMixin, unittest.TestCase):
    """The NEP generator adds the mass-grid handling and the reshaping of
    the emulator output into (radius, mass, lambda) triples."""

    def test_metadata_drives_the_emulator_path_and_mass_sampling(self):
        generator = eos_processing.NEPEoSGenerator(self.metadata())
        self.assertEqual(generator.n_mass_samples, self.n_mass_samples)
        self.assertEqual(generator.decompose_mass_data, generator.equal_distance_masses)

    def test_the_default_number_of_mass_samples_is_forty(self):
        metadata = {"emulator_path": str(self.emulator_path)}
        generator = eos_processing.NEPEoSGenerator(metadata)
        self.assertEqual(generator.n_mass_samples, 40)

    def test_a_matching_backend_declaration_is_accepted(self):
        generator = eos_processing.NEPEoSGenerator(
            self.metadata(backend=keras.backend.backend())
        )
        self.assertEqual(generator.n_mass_samples, self.n_mass_samples)

    def test_a_mismatched_backend_declaration_is_rejected(self):
        with self.assertRaises(AssertionError):
            eos_processing.NEPEoSGenerator(self.metadata(backend="not_a_backend"))

    def test_nep5_declares_the_five_nuclear_empirical_parameters(self):
        self.assertEqual(
            eos_processing.NEP5EoSGenerator.eos_parameters,
            ["K_sat", "L_sym", "K_sym", "3n_sat", "5n_sat"],
        )

    def test_generate_macro_eos_returns_one_triple_per_sample(self):
        generator = eos_processing.NEP5EoSGenerator(self.metadata())
        parameters = {key: np.array([1.0, 2.0]) for key in generator.eos_parameters}
        macro_eos = generator.generate_macro_eos(parameters)
        self.assertEqual(np.shape(macro_eos), (2, 3, self.n_mass_samples))

    def test_generate_macro_eos_accepts_scalar_parameters(self):
        generator = eos_processing.NEP5EoSGenerator(self.metadata())
        parameters = {key: 1.0 for key in generator.eos_parameters}
        macro_eos = generator.generate_macro_eos(parameters)
        self.assertEqual(np.shape(macro_eos), (1, 3, self.n_mass_samples))

    def test_adjust_format_splits_radii_lambdas_and_the_tov_mass(self):
        generator = eos_processing.NEP5EoSGenerator(self.metadata())
        radii = np.arange(self.n_mass_samples, dtype=float) + 11.0
        log_lambdas = np.linspace(3.0, 1.0, self.n_mass_samples)
        predictions = np.concatenate([radii, log_lambdas, [2.2]])[np.newaxis, :]

        macro_eos = generator.adjust_format(predictions)

        np.testing.assert_allclose(macro_eos[0, 0], radii)
        np.testing.assert_allclose(macro_eos[0, 2], 10.0**log_lambdas)
        np.testing.assert_allclose(macro_eos[0, 1][0], 1.0)
        np.testing.assert_allclose(macro_eos[0, 1][-1], 2.2)

    def test_a_split_mass_grid_is_configured_from_a_three_element_list(self):
        generator = eos_processing.NEP5EoSGenerator(
            self.metadata(n_mass_samples=[3, 2, 2.0])
        )
        self.assertEqual(generator.mass_samples_low, 3)
        self.assertEqual(generator.mass_samples_high, 2)
        self.assertEqual(generator.split_value, 2.0)
        self.assertEqual(generator.n_mass_samples, 5)
        self.assertEqual(generator.decompose_mass_data, generator.disjoint_masses)

    def test_a_two_element_list_falls_back_to_a_split_value_of_two(self):
        generator = eos_processing.NEP5EoSGenerator(
            self.metadata(n_mass_samples=[3, 2])
        )
        self.assertEqual(generator.split_value, 2.0)
        self.assertEqual(generator.n_mass_samples, 5)

    def test_properly_disjoint_masses_is_dense_below_the_split_value(self):
        generator = eos_processing.NEP5EoSGenerator(
            self.metadata(n_mass_samples=[3, 2, 2.0])
        )
        masses = generator.properly_disjoint_masses(np.array([2.2]))
        np.testing.assert_allclose(masses, [[1.0, 1.5, 2.0, 2.1, 2.2]])

    def test_disjoint_masses_falls_back_for_a_low_tov_mass(self):
        # An emulated TOV mass below the split value cannot be split, so the
        # equally spaced grid is used instead.
        generator = eos_processing.NEP5EoSGenerator(
            self.metadata(n_mass_samples=[3, 2, 2.0])
        )
        masses = generator.disjoint_masses(np.array([1.8]))
        np.testing.assert_allclose(masses, [[1.0, 1.2, 1.4, 1.6, 1.8]])

    def test_the_split_grid_is_ordered_and_ends_at_the_tov_mass(self):
        generator = eos_processing.NEP5EoSGenerator(
            self.metadata(n_mass_samples=[3, 2, 2.0])
        )
        masses = generator.disjoint_masses(np.array([2.4]))[0]
        self.assertTrue(np.all(np.diff(masses) > 0.0))
        self.assertAlmostEqual(masses[-1], 2.4)


class TestLECEoSGenerator(unittest.TestCase):
    """The LEC generators wrap scikit-learn emulators and scalers loaded
    with joblib, and predict mass, radius and lambda separately."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.n_mass_samples = 3
        self.radii = [12.0, 11.8, 11.5]
        self.log_lambdas = [3.0, 2.0, 1.0]
        self.metadata = {
            "feature_scaler": self.dump(IdentityScaler(), "feature_scaler"),
            "lambda_scaler": self.dump(IdentityScaler(), "lambda_scaler"),
            "radius_scaler": self.dump(IdentityScaler(), "radius_scaler"),
            "mass_emulator": self.dump(ConstantEmulator([2.1]), "mass"),
            "radius_emulator": self.dump(ConstantEmulator(self.radii), "radius"),
            "lambda_emulator": self.dump(ConstantEmulator(self.log_lambdas), "lambda"),
            "n_mass_samples": self.n_mass_samples,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def dump(self, obj, name):
        path = self.tmp_dir / f"{name}.joblib"
        joblib.dump(obj, path)
        return str(path)

    def test_lec7_declares_the_seven_low_energy_couplings(self):
        self.assertEqual(
            eos_processing.LEC7EoSGenerator.eos_parameters,
            ["d11", "d22", "d3", "d4", "d6", "d7"],
        )

    def test_lec13_extends_the_couplings_with_saturation_parameters(self):
        parameters = eos_processing.LEC13EoSGenerator.eos_parameters
        self.assertEqual(parameters[:6], eos_processing.LEC7EoSGenerator.eos_parameters)
        self.assertEqual(
            parameters[6:], ["ksat", "qsat", "zsat", "cssq1", "cssq2", "cssq3", "cssq4"]
        )

    def test_all_six_joblib_artefacts_are_loaded(self):
        generator = eos_processing.LEC7EoSGenerator(self.metadata)
        for attribute in [
            "feature_scaler",
            "lambda_scaler",
            "radius_scaler",
            "mass_emulator",
            "radius_emulator",
            "lambda_emulator",
        ]:
            self.assertTrue(hasattr(generator, attribute), msg=attribute)

    def test_predict_returns_mass_radius_and_lambda_predictions(self):
        generator = eos_processing.LEC7EoSGenerator(self.metadata)
        mass, radius, lambdas = generator.predict(np.ones((1, 6)))
        np.testing.assert_allclose(mass, [[2.1]])
        np.testing.assert_allclose(radius, [self.radii])
        np.testing.assert_allclose(lambdas, [self.log_lambdas])

    def test_generate_macro_eos_stacks_radii_masses_and_lambdas(self):
        generator = eos_processing.LEC7EoSGenerator(self.metadata)
        parameters = {key: 1.0 for key in generator.eos_parameters}
        macro_eos = generator.generate_macro_eos(parameters)
        self.assertEqual(np.shape(macro_eos), (1, 3, self.n_mass_samples))
        np.testing.assert_allclose(macro_eos[0, 0], self.radii)
        np.testing.assert_allclose(macro_eos[0, 1], [1.0, 1.55, 2.1])
        np.testing.assert_allclose(macro_eos[0, 2], 10.0 ** np.array(self.log_lambdas))

    def test_the_default_number_of_mass_samples_is_thirty(self):
        metadata = dict(self.metadata)
        metadata.pop("n_mass_samples")
        generator = eos_processing.LEC7EoSGenerator(metadata)
        self.assertEqual(generator.n_mass_samples, 30)


class TestEoSConverterMethodSelection(unittest.TestCase):
    """The converter infers which of the tabulated, emulated and
    quasi-universal-relation paths to take from the namespace it is given."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        write_macro_eos_file(self.tmp_dir / "1.dat")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_an_eos_file_selects_the_single_tabulated_eos_path(self):
        args = Namespace(eos_file=str(self.tmp_dir / "1.dat"), emulator_metadata=None)
        converter = eos_processing.EoSConverter(args)
        self.assertEqual(converter.macro_conversion, converter.single_eos_from_ram)

    def test_an_eos_directory_selects_the_tabulated_set_path(self):
        args = Namespace(
            eos_data=str(self.tmp_dir), Neos=1, eos_to_ram=True, emulator_metadata=None
        )
        converter = eos_processing.EoSConverter(args)
        self.assertEqual(converter.macro_conversion, converter.eos_from_ram)

    def test_emulator_metadata_selects_the_emulated_path(self):
        args = Namespace(emulator_metadata={"emulator_path": "path"})
        with patch.object(eos_processing, "setup_eos_generator") as setup:
            converter = eos_processing.EoSConverter(args, "emulated")
        self.assertIs(converter.tov_emulator, setup.return_value)
        self.assertEqual(
            converter.macro_conversion, setup.return_value.generate_macro_eos
        )

    def test_the_quasi_universal_relation_path_replaces_the_conversion(self):
        converter = eos_processing.EoSConverter(Namespace(), method="qur")
        self.assertIs(converter.parameter_conversion, eos_processing.radii_from_qur)

    def test_the_quasi_universal_relation_path_derives_radii_from_lambdas(self):
        converter = eos_processing.EoSConverter(Namespace(), method="qur")
        converted = converter(
            {
                "mass_1_source": 1.4,
                "mass_2_source": 1.3,
                "lambda_1": 300.0,
                "lambda_2": 400.0,
            }
        )
        for key in ["radius_1", "radius_2", "R_16"]:
            self.assertIn(key, converted)
            self.assertTrue(8.0 < float(converted[key]) < 16.0, msg=key)

    def test_an_unknown_method_is_reported(self):
        with self.assertRaises(ValueError) as context:
            eos_processing.EoSConverter(Namespace(), method="unsupported")
        self.assertIn("unsupported", str(context.exception))

    def test_a_namespace_without_any_eos_input_is_reported(self):
        with self.assertRaises(ValueError):
            eos_processing.EoSConverter(Namespace())

    def test_the_default_parameter_conversion_is_the_full_chain(self):
        args = Namespace(eos_file=str(self.tmp_dir / "1.dat"), emulator_metadata=None)
        converter = eos_processing.EoSConverter(args)
        self.assertEqual(converter.parameter_conversion, converter.full_eos_conversion)


class TestEoSConverterTabulatedSet(unittest.TestCase):
    """A set of tabulated EOSs is addressed by the integer EOS index that
    the sampler draws."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.n_eos = 3
        self.tables = [
            write_macro_eos_file(self.tmp_dir / f"{index + 1}.dat", 1.0 + 0.05 * index)
            for index in range(self.n_eos)
        ]
        self.args = Namespace(
            eos_data=str(self.tmp_dir),
            Neos=self.n_eos,
            eos_to_ram=True,
            eos_file=None,
            emulator_metadata=None,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_all_requested_files_are_loaded_into_memory(self):
        converter = eos_processing.EoSConverter(self.args)
        self.assertEqual(converter.Neos, self.n_eos)
        self.assertEqual(len(converter.eos_data), self.n_eos)
        for table in converter.eos_data:
            self.assertEqual(table.shape[0], 3)

    def test_without_neos_every_file_in_the_directory_is_used(self):
        args = Namespace(
            eos_data=str(self.tmp_dir),
            Neos=None,
            eos_to_ram=True,
            eos_file=None,
            emulator_metadata=None,
        )
        converter = eos_processing.EoSConverter(args)
        self.assertEqual(converter.Neos, self.n_eos)

    def test_the_eos_index_selects_the_table(self):
        converter = eos_processing.EoSConverter(self.args)
        selected = converter.eos_from_ram({"EOS": 2})
        self.assertEqual(len(selected), 1)
        np.testing.assert_allclose(selected[0], converter.eos_data[2])

    def test_a_float_eos_index_is_truncated_to_an_integer(self):
        converter = eos_processing.EoSConverter(self.args)
        np.testing.assert_allclose(
            converter.eos_from_ram({"EOS": 1.7})[0], converter.eos_data[1]
        )

    def test_an_array_of_eos_indices_selects_several_tables(self):
        converter = eos_processing.EoSConverter(self.args)
        selected = converter.eos_from_ram({"EOS": np.array([0, 2])})
        self.assertEqual(len(selected), 2)
        np.testing.assert_allclose(selected[1], converter.eos_data[2])

    def test_files_are_read_from_disk_when_they_are_not_kept_in_ram(self):
        args = Namespace(
            eos_data=str(self.tmp_dir),
            Neos=self.n_eos,
            eos_to_ram=False,
            eos_file=None,
            emulator_metadata=None,
        )
        converter = eos_processing.EoSConverter(args)
        self.assertEqual(converter.macro_conversion, converter.eos_direct_load)
        self.assertEqual(Path(converter.eos_data), self.tmp_dir)
        loaded = converter.eos_direct_load({"EOS": 0})
        self.assertEqual(loaded[0].shape[0], 3)

    def test_direct_loading_requires_files_that_are_already_numbered(self):
        # The renaming branch calls samefile on a target that does not exist
        # yet, so a directory of arbitrarily named files cannot be used
        # without eos_to_ram.
        other_dir = Path(tempfile.mkdtemp())
        try:
            write_macro_eos_file(other_dir / "soft.dat")
            args = Namespace(
                eos_data=str(other_dir),
                Neos=None,
                eos_to_ram=False,
                eos_file=None,
                emulator_metadata=None,
            )
            with self.assertRaises(FileNotFoundError):
                eos_processing.EoSConverter(args)
        finally:
            shutil.rmtree(other_dir)

    def glob_args(self, n_eos):
        return Namespace(
            eos_data="*.dat",
            Neos=n_eos,
            eos_to_ram=True,
            eos_file=None,
            emulator_metadata=None,
        )

    def test_a_glob_pattern_is_expanded_relative_to_the_working_directory(self):
        # Path().glob only accepts relative patterns, so a pattern reaches the
        # right files only when it is run from the directory holding them.
        with contextlib.chdir(self.tmp_dir):
            converter = eos_processing.EoSConverter(self.glob_args(self.n_eos))
        self.assertEqual(converter.Neos, self.n_eos)
        self.assertEqual(len(converter.eos_data), self.n_eos)

    def test_a_glob_that_finds_a_different_number_of_files_is_rejected(self):
        with contextlib.chdir(self.tmp_dir):
            with self.assertRaises(AssertionError):
                eos_processing.EoSConverter(self.glob_args(self.n_eos + 1))


class TestEoSConverterParameterConversion(unittest.TestCase):
    """The converter turns an EOS choice plus source-frame component masses
    into the neutron-star and binary parameters the likelihoods need."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.tables = [
            write_macro_eos_file(self.tmp_dir / f"{index + 1}.dat", 1.0 + 0.05 * index)
            for index in range(3)
        ]
        self.converter = eos_processing.EoSConverter(
            Namespace(
                eos_data=str(self.tmp_dir),
                Neos=3,
                eos_to_ram=True,
                eos_file=None,
                emulator_metadata=None,
            )
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_a_single_eos_yields_scalar_neutron_star_parameters(self):
        parameters = self.converter.compute_macro_parameters({"EOS": 0})
        masses, radii, _ = self.tables[0]
        self.assertAlmostEqual(parameters["TOV_mass"], masses[-1])
        self.assertAlmostEqual(parameters["TOV_radius"], radii[-1])
        self.assertAlmostEqual(parameters["R_14"], np.interp(1.4, masses, radii))
        self.assertAlmostEqual(parameters["R_16"], np.interp(1.6, masses, radii))

    def test_the_selected_macro_eos_is_cached_for_later_use(self):
        self.converter.compute_macro_parameters({"EOS": 1})
        cached = self.converter.macro_parameters
        self.assertEqual(sorted(cached), ["lambdas", "masses", "radii"])
        np.testing.assert_allclose(cached["masses"], self.tables[1][0])

    def test_several_eos_indices_yield_arrays_of_neutron_star_parameters(self):
        parameters = self.converter.compute_macro_parameters({"EOS": np.array([0, 2])})
        np.testing.assert_allclose(
            parameters["TOV_mass"], [self.tables[0][0][-1], self.tables[2][0][-1]]
        )
        self.assertEqual(parameters["R_14"].shape, (2,))

    def test_system_parameters_are_interpolated_at_the_component_masses(self):
        parameters = self.converter.compute_macro_parameters({"EOS": 0})
        parameters.update({"mass_1_source": 1.6, "mass_2_source": 1.2})
        converted = self.converter.system_props_from_eos(parameters)
        masses, radii, lambdas = self.tables[0]
        self.assertAlmostEqual(converted["radius_1"], np.interp(1.6, masses, radii))
        self.assertAlmostEqual(converted["radius_2"], np.interp(1.2, masses, radii))
        self.assertAlmostEqual(
            converted["lambda_1"],
            np.exp(np.interp(1.6, masses, np.log(lambdas))),
        )
        self.assertGreater(converted["lambda_2"], converted["lambda_1"])

    def test_a_component_mass_above_the_tov_mass_gives_a_vanishing_lambda(self):
        parameters = self.converter.compute_macro_parameters({"EOS": 0})
        parameters.update({"mass_1_source": 5.0, "mass_2_source": 1.2})
        converted = self.converter.system_props_from_eos(parameters)
        self.assertEqual(converted["lambda_1"], 0.0)
        self.assertEqual(converted["radius_1"], 0.0)

    def test_calling_the_converter_runs_the_full_chain(self):
        converted = self.converter(
            {
                "EOS": np.array([0, 2]),
                "mass_1_source": np.array([1.4, 1.5]),
                "mass_2_source": np.array([1.3, 1.2]),
            }
        )
        for key in ["TOV_mass", "R_14", "lambda_1", "lambda_2", "radius_1", "radius_2"]:
            self.assertEqual(np.shape(converted[key]), (2,), msg=key)

    def test_the_single_eos_path_ignores_the_eos_index(self):
        converter = eos_processing.EoSConverter(
            Namespace(eos_file=str(self.tmp_dir / "1.dat"), emulator_metadata=None)
        )
        converted = converter({"mass_1_source": 1.4, "mass_2_source": 1.3})
        masses, radii, _ = self.tables[0]
        self.assertAlmostEqual(converted["TOV_mass"], masses[-1])
        self.assertAlmostEqual(converted["radius_1"], np.interp(1.4, masses, radii))


class TestTabulatedEoSLoaders(unittest.TestCase):
    """Free functions used to read a directory of tabulated EOSs, in the
    column order radius, mass, lambda."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.tables = [
            write_macro_eos_file(self.tmp_dir / f"{index + 1}.dat", 1.0 + 0.05 * index)
            for index in range(3)
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_load_eos_files_sorts_the_directory_and_counts_the_files(self):
        files, n_eos = eos_processing.load_eos_files(str(self.tmp_dir), None)
        self.assertEqual([path.name for path in files], ["1.dat", "2.dat", "3.dat"])
        self.assertEqual(n_eos, 3)

    def test_load_eos_files_accepts_an_explicit_list(self):
        given = sorted(self.tmp_dir.glob("*.dat"))
        files, n_eos = eos_processing.load_eos_files(given, 3)
        self.assertIs(files, given)
        self.assertEqual(n_eos, 3)

    def test_load_eos_files_rejects_a_mismatched_count(self):
        with self.assertRaises(AssertionError):
            eos_processing.load_eos_files(str(self.tmp_dir), 5)

    def test_load_weights_reads_a_file_and_passes_arrays_through(self):
        weight_path = self.tmp_dir / "weights.txt"
        np.savetxt(weight_path, np.array([0.25, 0.75]))
        np.testing.assert_allclose(
            eos_processing.load_weights(str(weight_path)), [0.25, 0.75]
        )
        weights = np.array([1.0, 2.0])
        self.assertIs(eos_processing.load_weights(weights), weights)
        self.assertIsNone(eos_processing.load_weights(None))

    def test_load_to_dict_keys_the_set_by_a_one_based_index(self):
        eos_data, weights, n_eos = eos_processing.load_tabulated_macro_eos_set_to_dict(
            str(self.tmp_dir)
        )
        self.assertEqual(sorted(eos_data), [1, 2, 3])
        self.assertEqual(sorted(eos_data[1]), ["Lambda", "M", "R"])
        self.assertIsNone(weights)
        self.assertEqual(n_eos, 3)

    def test_load_to_dict_reads_the_columns_in_the_documented_order(self):
        eos_data, _, _ = eos_processing.load_tabulated_macro_eos_set_to_dict(
            str(self.tmp_dir)
        )
        masses, radii, lambdas = self.tables[0]
        np.testing.assert_allclose(eos_data[1]["M"], masses)
        np.testing.assert_allclose(eos_data[1]["R"], radii)
        np.testing.assert_allclose(eos_data[1]["Lambda"], lambdas)

    def test_load_to_dict_attaches_weights_when_they_are_given(self):
        weight_path = self.tmp_dir / "weights.txt"
        np.savetxt(weight_path, np.array([0.2, 0.3, 0.5]))
        eos_data, weights, _ = eos_processing.load_tabulated_macro_eos_set_to_dict(
            str(self.tmp_dir), str(weight_path), 3
        )
        np.testing.assert_allclose(weights, [0.2, 0.3, 0.5])
        self.assertAlmostEqual(eos_data[3]["weight"], 0.5)

    def test_load_to_list_returns_one_array_per_eos(self):
        eos_data, weights, n_eos = eos_processing.load_tabulated_macro_eos_set_to_list(
            str(self.tmp_dir)
        )
        self.assertEqual(len(eos_data), 3)
        self.assertEqual(eos_data[0].shape[0], 3)
        np.testing.assert_allclose(eos_data[0][0], self.tables[0][0])
        self.assertIsNone(weights)
        self.assertEqual(n_eos, 3)

    @unittest.expectedFailure
    def test_load_macro_characteristics_returns_the_tov_masses(self):
        # The function assembles its output list but never returns it.
        mtovs = eos_processing.load_macro_characteristics_from_tabulated_eos_set(
            str(self.tmp_dir), 3
        )
        np.testing.assert_allclose(mtovs[0], [table[0][-1] for table in self.tables])

    @unittest.expectedFailure
    def test_load_macro_characteristics_can_evaluate_characteristic_radii(self):
        # The radius buffer is allocated with np.empty_like(Neos, ...), which
        # is not a valid call, so requesting characteristic radii raises.
        eos_processing.load_macro_characteristics_from_tabulated_eos_set(
            str(self.tmp_dir), 3, masses_for_char_radii=1.4
        )


if __name__ == "__main__":
    unittest.main()
