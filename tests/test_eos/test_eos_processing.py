import shutil
from argparse import Namespace

import numpy as np
import pytest
from helpers import (
    EOS_MACRO_DIR,
    N_EOS_MACRO,
    TableEmulator,
    eos_files,
    eos_macro_generator,
    eos_macro_tables,
)
from test_tov_emulators import random_emulator_file

from nmma.core.conversion import (
    DisjointMassGrid,
    MassGrid,
    max_mass_increase_by_spin,
    rotational_mass_increase,
)
from nmma.eos import eos_processing as ep


def resampled_curves(grid: MassGrid, table: np.ndarray) -> tuple:
    "(radii, ln-lambdas, TOV mass) of a table resampled onto ``grid``"
    radius, mass, ln_lam = table
    radii, ln_lams = grid.resample(mass, radius, ln_lam)
    return radii, ln_lams, mass[-1]


def expected_properties(
    grid: MassGrid, curves: tuple, mass_1, mass_2, characteristic_masses
) -> dict:
    """Macroscopic and component properties of non-spinning NSs inside the curves
    (radii, ln-lambdas, TOV mass) of one EOS, interpolated along ``grid``; R_14 and
    R_16 at ``characteristic_masses``."""
    radii, ln_lams, m_tov = curves
    # as the single EOS of a batch
    grid_curves, tov_masses = np.stack([radii, ln_lams])[None], np.array([m_tov])
    (r_14, r_16), _ = grid.interpolate(characteristic_masses, tov_masses, grid_curves)[
        :, 0
    ]
    comp_radii, comp_ln_lams = grid.interpolate(
        [mass_1, mass_2], tov_masses, grid_curves
    )[:, 0]
    properties = {
        "TOV_mass": m_tov,
        "TOV_radius": radii[-1],
        "R_14": r_14,
        "R_16": r_16,
    }
    for j in range(2):
        properties[f"lambda_{j + 1}"] = np.exp(comp_ln_lams[j])
        properties[f"radius_{j + 1}"] = comp_radii[j]
        properties[f"is_ns_{j + 1}"] = 1
    return properties


class EoSGeneratorTestContainer:
    """Behaviour every generator provides: radius and ln Lambda curves of shape
    (n_eos, n_points) on its mass grid with the TOV masses (n_eos,), and from those
    the (radius, mass, ln Lambda) tables of shape (n_eos, 3, n_points). Subclasses
    set ``cls.generator`` in ``setup_class`` and define ``inputs``."""

    eos_indices = np.array([0, 4, 9])

    @classmethod
    def setup_class(cls):
        cls.tables = eos_macro_tables()

    def test_call_format(self) -> None:
        n_points = self.generator.mass_grid.n_points
        for eos_indices in (self.eos_indices, self.eos_indices[:1]):
            radii, ln_lams, m_tov = self.generator(self.inputs(eos_indices))
            assert np.shape(radii) == (len(eos_indices), n_points)
            assert np.shape(ln_lams) == (len(eos_indices), n_points)
            assert np.shape(m_tov) == (len(eos_indices),)

        radii, _, m_tov = self.generator(self.inputs(self.eos_indices[0]))
        assert np.shape(radii) == (1, n_points)
        assert np.shape(m_tov) == (1,)

    def test_curves_format(self) -> None:
        n_points = self.generator.mass_grid.n_points
        curves, m_tov = self.generator.curves(self.inputs(self.eos_indices))
        assert np.shape(curves) == (len(self.eos_indices), 2, n_points)
        assert np.shape(m_tov) == (len(self.eos_indices),)

    def test_batch_shape(self) -> None:
        batch = self.generator.batch_shape(self.inputs(self.eos_indices))
        assert batch == (len(self.eos_indices),)
        assert self.generator.batch_shape(self.inputs(self.eos_indices[0])) == ()

    def test_tables_from_curves(self) -> None:
        curves, m_tov = self.generator.curves(self.inputs(self.eos_indices))
        tables = self.generator.tables_from_curves(curves, m_tov)
        n_points = self.generator.mass_grid.n_points
        assert np.shape(tables) == (len(self.eos_indices), 3, n_points)
        assert tables[:, 0] == pytest.approx(curves[:, 0])
        assert tables[:, 1] == pytest.approx(self.generator.mass_grid.masses(m_tov))
        assert tables[:, 2] == pytest.approx(curves[:, 1])

    def test_parameter_dict_round_trip(self) -> None:
        params = self.generator.assemble_eos_params(self.inputs(self.eos_indices))
        parameters = self.generator.parameter_dict(params)
        assert self.generator.assemble_eos_params(parameters) == pytest.approx(params)


class TestEoSGenerator(EoSGeneratorTestContainer):
    """The default generator around an emulator with metadata, here the
    TableEmulator, whose first parameter selects a test EOS."""

    eos_parameters = ["K_sat", "L_sym", "K_sym", "3n_sat", "5n_sat"]
    stored_parameters = ["L_sym", "K_sym"]
    n_mass_samples = 30
    disjoint_n_mass_samples = (10, 15, 1.8)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.grid = MassGrid.from_n_mass_samples(cls.n_mass_samples)
        cls.emulator = TableEmulator(cls.grid)
        cls.generator = ep.EoSGenerator(cls.emulator, eos_parameters=cls.eos_parameters)

    def inputs(self, eos_indices) -> dict:
        inputs = {par: np.zeros(np.shape(eos_indices)) for par in self.eos_parameters}
        inputs[self.eos_parameters[0]] = eos_indices
        return inputs

    def test_init_sets_up(self) -> None:
        assert self.generator.emulator is self.emulator
        assert self.generator.eos_parameters == self.eos_parameters
        assert self.generator.mass_grid.config() == self.emulator.metadata["mass_grid"]
        assert self.generator.n_mass_samples == self.n_mass_samples

    def test_eos_parameters_from_metadata(self) -> None:
        emulator = TableEmulator(self.grid)
        emulator.metadata["eos_parameters"] = self.stored_parameters
        assert ep.EoSGenerator(emulator).eos_parameters == self.stored_parameters

    def test_missing_eos_parameters_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ep.EoSGenerator(self.emulator)

    def test_mass_grid_from_disjoint_n_mass_samples(self) -> None:
        generator = ep.EoSGenerator(
            self.emulator, self.eos_parameters, mass_grid=self.disjoint_n_mass_samples
        )
        assert type(generator.mass_grid) is DisjointMassGrid

    def test_predict_curves_split_the_emulator_output(self) -> None:
        params = self.generator.assemble_eos_params(self.inputs(self.eos_indices))
        curves, m_tov = self.generator.predict_curves(params)
        for i, idx in enumerate(self.eos_indices):
            radii, ln_lams, tov_mass = resampled_curves(self.grid, self.tables[idx])
            assert curves[i, 0] == pytest.approx(radii)
            assert curves[i, 1] == pytest.approx(ln_lams)
            assert m_tov[i] == pytest.approx(tov_mass)


class TestKerasEoSGenerator(EoSGeneratorTestContainer):
    """A minimal Keras model whose output does not depend on its inputs:
    constant radii, log10-lambdas and TOV mass."""

    n_mass_samples = 8
    default_n_mass_samples = 30
    radius = 12.0
    log10_lambda = 2.5
    tov_mass = 2.2
    eos_parameters = ["a", "b"]

    @classmethod
    def setup_class(cls) -> None:
        keras = pytest.importorskip("keras")
        super().setup_class()
        n = cls.n_mass_samples
        output = np.concatenate(
            [np.full(n, cls.radius), np.full(n, cls.log10_lambda), [cls.tov_mass]]
        )
        n_inputs = len(cls.eos_parameters)
        model = keras.Sequential(
            [keras.Input(shape=(n_inputs,)), keras.layers.Dense(2 * n + 1)]
        )
        model.layers[-1].set_weights([np.zeros((n_inputs, 2 * n + 1)), output])
        cls.model_path = cls.outdir / "emulator.keras"
        model.save(cls.model_path)
        cls.generator = ep.KerasEoSGenerator(
            cls.model_path,
            eos_parameters=cls.eos_parameters,
            mass_grid=cls.n_mass_samples,
        )

    def inputs(self, eos_indices) -> dict:
        return {par: np.zeros(np.shape(eos_indices)) for par in self.eos_parameters}

    def test_init_sets_up(self) -> None:
        expected = MassGrid.from_n_mass_samples(self.n_mass_samples)
        assert self.generator.mass_grid.config() == expected.config()

    def test_default_mass_grid(self) -> None:
        generator = ep.KerasEoSGenerator(
            self.model_path, eos_parameters=self.eos_parameters
        )
        assert generator.n_mass_samples == self.default_n_mass_samples

    def test_log10_lambdas_become_ln_lambdas(self) -> None:
        radii, ln_lams, m_tov = self.generator(self.inputs(self.eos_indices))
        assert radii == pytest.approx(self.radius)
        assert ln_lams == pytest.approx(np.log(10) * self.log10_lambda)
        assert m_tov == pytest.approx(self.tov_mass)


class TestTabulatedEoSGenerator(EoSGeneratorTestContainer):
    """The tabulated EOSs, resampled onto the default mass grid at initialisation."""

    grid_below_table_start = {"n_points": 50, "m_min": 0.1, "power": 2.0}
    n_prefixed = 3
    single_eos = 4

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()
        cls.generator = eos_macro_generator()

    def inputs(self, eos_indices) -> dict:
        return {"EOS": eos_indices}

    def test_init_sets_up(self) -> None:
        assert self.generator.Neos == N_EOS_MACRO
        assert self.generator.eos_parameters == ["EOS"]
        assert self.generator.eos_files == eos_files()
        assert self.generator.mass_grid.config() == MassGrid().config()

    def test_curves_select_the_resampled_tables(self) -> None:
        "The resampling itself is tested with MassGrid.interpolate on node masses"
        radii, ln_lams, m_tov = self.generator(self.inputs(self.eos_indices))
        for i, idx in enumerate(self.eos_indices):
            expected = resampled_curves(self.generator.mass_grid, self.tables[idx])
            assert radii[i] == pytest.approx(expected[0])
            assert ln_lams[i] == pytest.approx(expected[1])
            assert m_tov[i] == pytest.approx(expected[2])

    def test_tables_must_cover_the_mass_grid(self) -> None:
        args = Namespace(eos_data=EOS_MACRO_DIR)
        with pytest.raises(ValueError):
            ep.TabulatedEoSGenerator(args, mass_grid=self.grid_below_table_start)

    def test_single_table_serves_every_entry(self) -> None:
        args = Namespace(eos_file=str(eos_files()[self.single_eos]))
        generator = ep.TabulatedEoSGenerator(args)
        parameters = {"mass_1_source": self.eos_indices}
        assert generator.batch_shape(parameters) == np.shape(self.eos_indices)
        radii, ln_lams, m_tov = generator(parameters)
        expected = resampled_curves(generator.mass_grid, self.tables[self.single_eos])
        assert len(m_tov) == len(self.eos_indices)
        for i in range(len(self.eos_indices)):
            assert radii[i] == pytest.approx(expected[0])
            assert ln_lams[i] == pytest.approx(expected[1])
            assert m_tov[i] == pytest.approx(expected[2])

    def test_raw_tables_are_the_file_tables(self) -> None:
        raw_tables = self.generator.raw_tables(self.inputs(self.eos_indices))
        assert len(raw_tables) == len(self.eos_indices)
        for table, idx in zip(raw_tables, self.eos_indices):
            assert table == pytest.approx(self.tables[idx])

    def test_reference_dataset_holds_the_raw_tables(self) -> None:
        dataset = self.generator.reference_dataset()
        assert dataset.params[:, 0] == pytest.approx(np.arange(N_EOS_MACRO))
        for i, table in enumerate(self.tables):
            masses, radii, lambdas = dataset.stable_branch(i)
            assert masses == pytest.approx(table[1])
            assert radii == pytest.approx(table[0])
            assert np.log(lambdas) == pytest.approx(table[2])

    def test_read_eos_table_detects_lambda_scale(self) -> None:
        table = self.tables[self.single_eos]
        ln_lambda_file = self.outdir / "ln_lambda_eos.dat"
        np.savetxt(ln_lambda_file, table.T)
        for eos_file in (eos_files()[self.single_eos], ln_lambda_file):
            assert self.generator.read_eos_table(eos_file) == pytest.approx(table)

    def test_eos_files_from_glob(self) -> None:
        for j in range(self.n_prefixed):
            shutil.copy(eos_files()[j], self.outdir / f"eos_{j + 1}_macro.dat")
        generator = ep.TabulatedEoSGenerator(
            Namespace(eos_data=self.outdir / "eos_*.dat")
        )
        assert generator.Neos == self.n_prefixed
        assert [f.name for f in generator.eos_files] == [
            f"eos_{j + 1}_macro.dat" for j in range(self.n_prefixed)
        ]
        with pytest.raises(ValueError):
            ep.TabulatedEoSGenerator(Namespace(eos_data=self.outdir / "*.nothing"))


class EoSConverterTestContainer:
    """Shared tests for every converter setup turning EOS curves into macroscopic
    and component properties, interpolated along the mass grid of the generator.
    Subclasses define ``make_converter``, ``eos_selection`` and the
    ``expected_curves`` (radii, ln-lambdas, TOV mass) the converter sees. The
    converter keeps the curves of its last batch, so every test gets its own."""

    eos_indices = np.array([0, 4, 9])
    mass_1_source = np.array([1.6, 1.45, 1.35])
    mass_2_source = np.array([1.2, 1.3, 1.1])
    macro_keys = ["TOV_mass", "TOV_radius", "R_14", "R_16"]
    characteristic_masses = np.array([1.4, 1.6])  # of R_14 and R_16
    system_keys = ["lambda_1", "radius_1", "is_ns_1", "lambda_2", "radius_2", "is_ns_2"]
    mass_above_tov = 0.1
    mass_below_grid = 0.5  # below m_min of all grids used here
    breakup_spin = 0.5  # above the chi_lim of eos_to_system_parameters
    supported_spin = 0.3

    @classmethod
    def setup_class(cls):
        cls.tables = eos_macro_tables()

    def setup_method(self):
        self.converter = self.make_converter()

    def sample_parameters(self, eos_indices, mass_1, mass_2, a_1=0.0, a_2=0.0):
        parameters = self.eos_selection(eos_indices)
        parameters["mass_1_source"] = mass_1
        parameters["mass_2_source"] = mass_2
        parameters["a_1"] = a_1 + 0 * np.asarray(mass_1)
        parameters["a_2"] = a_2 + 0 * np.asarray(mass_2)
        return parameters

    def single_parameters(self):
        return self.sample_parameters(
            self.eos_indices[0], self.mass_1_source[0], self.mass_2_source[0]
        )

    def check_entry(self, result, eos_idx, mass_1, mass_2):
        expected = expected_properties(
            self.converter.mass_grid,
            self.expected_curves(eos_idx),
            mass_1,
            mass_2,
            self.characteristic_masses,
        )
        for key, value in expected.items():
            assert result[key] == pytest.approx(value)

    def test_init_sets_up(self):
        assert self.converter.parameter_conversion == self.converter.full_eos_conversion
        assert self.converter.mass_grid is self.converter.generator.mass_grid
        assert self.converter.spin_correction is False
        assert self.make_converter(spin_correction=True).spin_correction is True
        macro_converter = self.make_converter(macro_only=True)
        assert (
            macro_converter.parameter_conversion
            == macro_converter.compute_macro_parameters
        )

    def test_single_eos_conversion(self):
        parameters = self.single_parameters()
        result = self.converter(parameters)
        assert result is parameters
        for key in self.macro_keys + self.system_keys:
            assert np.ndim(result[key]) == 0
        assert np.ndim(self.converter.macro_parameters["radii"]) == 1
        self.check_entry(
            result, self.eos_indices[0], self.mass_1_source[0], self.mass_2_source[0]
        )

    def test_multiple_eos_conversion(self):
        parameters = self.sample_parameters(
            self.eos_indices, self.mass_1_source, self.mass_2_source
        )
        result = self.converter(parameters)
        for key in self.macro_keys + self.system_keys:
            assert np.shape(result[key]) == (len(self.eos_indices),)
        for i, idx in enumerate(self.eos_indices):
            entry = {key: value[i] for key, value in result.items()}
            self.check_entry(entry, idx, self.mass_1_source[i], self.mass_2_source[i])

    def test_macro_only_conversion(self):
        converter = self.make_converter(macro_only=True)
        result = converter(self.single_parameters())
        assert all(key in result for key in self.macro_keys)
        assert not any(key in result for key in self.system_keys)

    def test_compute_macro_parameters_stores_curves(self):
        parameters = self.sample_parameters(
            self.eos_indices, self.mass_1_source, self.mass_2_source
        )
        result = self.converter.compute_macro_parameters(parameters)
        assert "lambda_1" not in result
        stored = self.converter.macro_parameters
        assert list(stored) == ["radii", "masses", "ln_lams"]
        n_points = self.converter.mass_grid.n_points
        assert np.shape(stored["radii"]) == (len(self.eos_indices), n_points)
        for i, idx in enumerate(self.eos_indices):
            radii, ln_lams, m_tov = self.expected_curves(idx)
            masses = self.converter.mass_grid.masses(m_tov)
            assert stored["radii"][i] == pytest.approx(radii)
            assert stored["masses"][i] == pytest.approx(masses)
            assert stored["ln_lams"][i] == pytest.approx(ln_lams)

    def test_curves_at_is_zero_outside_the_curves(self):
        self.converter.compute_macro_parameters(self.single_parameters())
        m_tov = self.converter.m_tov[0]
        outside = [self.mass_below_grid, m_tov + self.mass_above_tov]
        radii, ln_lams = self.converter.curves_at(outside)
        assert np.all(radii == 0.0)
        assert np.all(ln_lams == -np.inf)

    def test_components_outside_ns_range(self):
        too_heavy = self.single_parameters()
        self.converter.compute_macro_parameters(too_heavy)
        too_heavy["mass_1_source"] = too_heavy["TOV_mass"] + self.mass_above_tov
        result = self.converter.add_system_properties(too_heavy)
        assert result["is_ns_1"] == 0
        assert result["lambda_1"] == 0.0
        assert result["radius_1"] == 0.0
        assert result["is_ns_2"] == 1

        too_fast = self.single_parameters()
        too_fast["a_2"] = self.breakup_spin
        result = self.converter(too_fast)
        assert result["is_ns_1"] == 1
        assert result["is_ns_2"] == 0
        assert result["radius_2"] == 0.0

    def test_components_below_the_mass_grid(self):
        "Below m_min the curves are undefined, as in the former raw tables"
        too_light = self.single_parameters()
        too_light["mass_2_source"] = self.mass_below_grid
        too_light["a_2"] = self.supported_spin
        for spin_correction in (False, True):
            self.converter.spin_correction = spin_correction
            result = self.converter(dict(too_light))
            assert result["is_ns_2"] == 1
            assert result["lambda_2"] == 0.0
            assert result["radius_2"] == 0.0

    def test_spin_correction(self):
        parameters = self.single_parameters()
        self.converter.compute_macro_parameters(parameters)

        # no spin -> no correction
        static = self.converter.add_system_properties(dict(parameters))
        self.converter.spin_correction = True
        result = self.converter.add_system_properties(dict(parameters))
        for key, value in static.items():
            assert result[key] == pytest.approx(value)

        # supramassive, but supported by rotation
        mass_increase = max_mass_increase_by_spin(self.supported_spin) - 1
        parameters["mass_1_source"] = parameters["TOV_mass"] * (1 + 0.5 * mass_increase)
        parameters["a_1"] = self.supported_spin
        result = self.converter.add_system_properties(dict(parameters))
        assert result["is_ns_1"] == 1
        assert result["lambda_1"] > 0.0
        assert result["radius_1"] > 0.0

        self.converter.spin_correction = False
        result = self.converter.add_system_properties(dict(parameters))
        assert result["is_ns_1"] == 0


class TestEoSConverter(EoSConverterTestContainer):
    """EoSConverter on top of the TableEmulator, which provides the test EOS of its
    first parameter, resampled onto its mass grid. Also covers how the converter
    sets up its generator from arguments, see ``build_generator``."""

    eos_parameters = ["K_sat", "L_sym", "K_sym", "3n_sat", "5n_sat"]
    n_mass_samples = 30
    seed = 42
    model_name = "LEC-7"  # must be case-insensitive
    model_parameters = ["d11", "d22", "d3", "d4", "d6", "d7"]
    emulator_layout = {"units": 8, "n_points": 6, "n_components": 2}
    metadata_n_mass_samples = 6

    @classmethod
    def setup_class(cls) -> None:
        super().setup_class()
        cls.grid = MassGrid.from_n_mass_samples(cls.n_mass_samples)
        cls.generator = ep.EoSGenerator(
            TableEmulator(cls.grid), eos_parameters=cls.eos_parameters
        )
        emulator_file = random_emulator_file(
            np.random.default_rng(cls.seed), cls.eos_parameters, **cls.emulator_layout
        )
        cls.emulator_path = emulator_file.write(cls.outdir / "emulator")

    def make_converter(self, macro_only=False, **kwargs):
        return ep.EoSConverter(
            Namespace(**kwargs), generator=self.generator, macro_only=macro_only
        )

    def eos_selection(self, eos_indices):
        selection = {
            par: np.zeros(np.shape(eos_indices)) for par in self.eos_parameters
        }
        selection[self.eos_parameters[0]] = eos_indices
        return selection

    def expected_curves(self, eos_idx):
        return resampled_curves(self.grid, self.tables[eos_idx])

    def emulator_args(self, **kwargs) -> Namespace:
        "Emulator arguments as the CLI parser provides them, see eos_parsing"
        args = dict(emulator_path=None, emulator_metadata={}, micro_eos_model=None)
        return Namespace(**(args | kwargs))

    def test_generator_from_emulator_file(self) -> None:
        args = self.emulator_args(emulator_path=self.emulator_path)
        converter = ep.EoSConverter(args)
        assert type(converter.generator) is ep.EoSGenerator
        assert converter.generator.eos_parameters == self.eos_parameters

    def test_micro_eos_model_sets_eos_parameters(self) -> None:
        args = self.emulator_args(
            emulator_path=self.emulator_path, micro_eos_model=self.model_name
        )
        assert ep.EoSConverter(args).generator.eos_parameters == self.model_parameters

    def test_n_mass_samples_sets_the_mass_grid(self) -> None:
        metadata = {"n_mass_samples": self.metadata_n_mass_samples}
        args = self.emulator_args(
            emulator_path=self.emulator_path, emulator_metadata=metadata
        )
        expected = MassGrid.from_n_mass_samples(self.metadata_n_mass_samples)
        assert ep.EoSConverter(args).mass_grid.config() == expected.config()
        assert metadata == {"n_mass_samples": self.metadata_n_mass_samples}

    def test_unsupported_emulator_files_are_rejected(self) -> None:
        args = self.emulator_args(emulator_path=self.outdir / "emulator.pkl")
        with pytest.raises(ValueError):
            ep.EoSConverter(args)
        with pytest.raises(ValueError):
            ep.EoSConverter(self.emulator_args(emulator_path=None))

    def test_tabulated_generator_from_arguments(self) -> None:
        converter = ep.EoSConverter(
            Namespace(eos_data=str(EOS_MACRO_DIR), Neos=N_EOS_MACRO)
        )
        assert type(converter.generator) is ep.TabulatedEoSGenerator
        assert converter.generator.Neos == N_EOS_MACRO


class TestEoSConverterWithTables(EoSConverterTestContainer):
    """EoSConverter on tabulated EOSs, resampled onto the default mass grid. Being
    deterministic, it also tests the non-rotating masses of spinning components."""

    spins = np.array([0.0, 0.1, 0.35])
    supramassive_factor = 1.02  # times M_tov
    # fixed point tolerance of non_rotating_mass (1e-10), times 1 + dm_bar chi**2
    mass_tolerance = 1e-9

    def setup_method(self):
        super().setup_method()
        self.m_source = np.stack([self.mass_1_source, self.mass_2_source], axis=-1)
        self.chi = np.stack([self.spins, self.spins[::-1]], axis=-1)

    def make_converter(self, macro_only=False, **kwargs):
        return ep.EoSConverter(
            Namespace(**kwargs), generator=eos_macro_generator(), macro_only=macro_only
        )

    def eos_selection(self, eos_indices):
        return {"EOS": eos_indices}

    def expected_curves(self, eos_idx):
        return resampled_curves(self.converter.mass_grid, self.tables[eos_idx])

    def non_rotating_masses(self, eos_indices, m_source, chi):
        "Non-rotating masses on the curves of ``eos_indices``"
        self.converter.update_curves({"EOS": eos_indices})
        m_tov = self.converter.m_tov[:, None]
        return self.converter.non_rotating_mass(m_source, chi, m_tov)

    def test_full_conversion_matches_separate_steps(self):
        """full_eos_conversion interpolates once for both steps, with and without
        spin correction"""
        parameters = self.sample_parameters(
            self.eos_indices,
            self.mass_1_source,
            self.mass_2_source,
            a_1=self.supported_spin,
        )
        for spin_correction in (False, True):
            self.converter.spin_correction = spin_correction
            full = self.converter.full_eos_conversion(dict(parameters))
            separate = self.converter.add_system_properties(
                self.converter.compute_macro_parameters(dict(parameters))
            )
            for key in self.macro_keys + self.system_keys:
                np.testing.assert_array_equal(full[key], separate[key])

    def test_non_rotating_mass_solves_its_defining_equation(self) -> None:
        m_0 = self.non_rotating_masses(self.eos_indices, self.m_source, self.chi)
        ln_lam = self.converter.ln_lam_at(m_0)
        rotating = m_0 * (1.0 + rotational_mass_increase(ln_lam) * self.chi**2)
        assert rotating == pytest.approx(self.m_source, abs=self.mass_tolerance)
        static = self.chi == 0.0
        assert np.all(m_0[static] == self.m_source[static])

    def test_non_rotating_mass_batch_matches_single_samples(self) -> None:
        batch = self.non_rotating_masses(self.eos_indices, self.m_source, self.chi)
        for i, eos_idx in enumerate(self.eos_indices):
            single = self.non_rotating_masses(eos_idx, self.m_source[i], self.chi[i])
            assert single[0] == pytest.approx(batch[i], abs=self.mass_tolerance)

    def test_non_rotating_mass_is_capped_at_tov_mass(self) -> None:
        self.converter.update_curves({"EOS": self.eos_indices[0]})
        m_tov = self.converter.m_tov[:, None]
        m_source = self.supramassive_factor * m_tov
        m_0 = self.converter.non_rotating_mass(m_source, np.zeros(1), m_tov)
        assert m_0 == pytest.approx(m_tov)

    def test_non_rotating_mass_below_the_grid_stays_undefined(self) -> None:
        "The spin correction must not pull masses below the grid into it"
        m_source = np.array([self.mass_below_grid, self.mass_2_source[0]])
        chi = np.full(2, self.spins[-1])
        m_0 = self.non_rotating_masses(self.eos_indices[0], m_source, chi)
        ln_lam = self.converter.ln_lam_at(m_0)
        assert ln_lam[0, 0] == -np.inf
        assert np.isfinite(ln_lam[0, 1])


class TestEoSConverterWithSingleTable(TestEoSConverterWithTables):
    """A single table serves every entry, without an "EOS" parameter."""

    eos_indices = np.array([4, 4, 4])

    def make_converter(self, macro_only=False, **kwargs):
        args = Namespace(eos_file=str(eos_files()[self.eos_indices[0]]))
        generator = ep.TabulatedEoSGenerator(args)
        return ep.EoSConverter(
            Namespace(**kwargs), generator=generator, macro_only=macro_only
        )

    def eos_selection(self, eos_indices):
        return {}


class TestEoSConverterVsDataset:
    """test_vs_dataset of a converter on resampled tables against its own tables.
    The resampling error must stay well below the precision of the TOV solver
    (about 4e-4 relative), see ``max_relative_error``."""

    max_relative_error = 1e-4
    ref_masses = (1.4, 1.6, 1.8, 2.0)

    def test_vs_own_tables(self) -> None:
        converter = ep.EoSConverter(generator=eos_macro_generator())
        characteristics = converter.test_vs_dataset(ref_masses=self.ref_masses)
        summary = characteristics["summary"]
        for mass in self.ref_masses:
            for obs in ("rad", "lam"):
                model = summary[str(mass)][obs]["model"]
                assert model["n"] > 0
                assert model["median"] < self.max_relative_error
        assert characteristics["m_tov_pred"] == pytest.approx(characteristics["m_tov"])
