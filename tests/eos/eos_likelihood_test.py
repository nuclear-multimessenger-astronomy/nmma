import json
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest
from bilby.core.prior import PriorDict, WeightedCategorical
from scipy.stats import norm

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from nmma.eos import eos_likelihood  # noqa: E402
from nmma.eos.eos_processing import EoSConverter  # noqa: E402


def write_macro_eos_file(path, scale=1.0, num=20):
    """Write a monotonic mass-radius-lambda table in the column order the
    EOS machinery expects: radius, mass, tidal deformability."""
    masses = np.linspace(0.5, 2.0 * scale, num)
    radii = 12.0 * scale - 0.3 * masses
    lambdas = 1000.0 * np.exp(-2.0 * masses)
    np.savetxt(path, np.column_stack([radii, masses, lambdas]))
    return masses, radii, lambdas


def write_mass_radius_posterior(path, columns="mass_first", num=20000, seed=7):
    """Write a synthetic mass-radius posterior, in either column order."""
    generator = np.random.default_rng(seed)
    masses = generator.normal(1.4, 0.1, num)
    radii = generator.normal(12.0, 0.5, num)
    if columns == "mass_first":
        data = np.column_stack([masses, radii])
    elif columns == "radius_first":
        data = np.column_stack([radii, masses])
    else:
        data = np.column_stack([masses, radii, np.ones(num)])
    np.savetxt(path, data)
    return masses, radii


class MacroEoSSetMixin:
    """A directory of tabulated EOSs named the way the samplers index them."""

    n_eos = 4

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.eos_dir = self.tmp_dir / "eos"
        self.eos_dir.mkdir()
        self.tables = [
            write_macro_eos_file(self.eos_dir / f"{index + 1}.dat", 1.0 + 0.06 * index)
            for index in range(self.n_eos)
        ]

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def tabulated_converter(self):
        converter = EoSConverter(
            Namespace(
                eos_data=str(self.eos_dir),
                Neos=self.n_eos,
                eos_to_ram=True,
                eos_file=None,
                emulator_metadata=None,
            ),
            "tabulated",
        )
        converter.parameter_conversion = converter.compute_macro_parameters
        return converter


class PlotRcParamsMixin:
    """Importing the plotting helpers switches LaTeX rendering on outside
    CI, which needs a LaTeX installation a developer machine may not have.
    Each test renders with mathtext and restores the global rcParams."""

    def setup_method(self):
        self.original_rc = matplotlib.rcParams.copy()
        matplotlib.rcParams["text.usetex"] = False
        getattr(super(), "setup_method", lambda: None)()

    def teardown_method(self):
        plt.close("all")
        matplotlib.rcParams.update(self.original_rc)
        getattr(super(), "teardown_method", lambda: None)()


class TestSetupTabulatedEoSPriors(MacroEoSSetMixin):
    """Sampling over a set of precomputed EOSs means one categorical prior
    over the EOS index."""

    def test_neos_sets_the_number_of_categories(self):
        priors = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(eos_data=str(self.eos_dir), Neos=3, eos_weight=None), PriorDict()
        )
        assert isinstance(priors["EOS"], WeightedCategorical)
        assert priors["EOS"].ncategories == 3
        assert priors["EOS"].name == "EOS"

    def test_without_neos_the_directory_is_counted(self):
        priors = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(eos_data=str(self.eos_dir), Neos=None, eos_weight=None),
            PriorDict(),
        )
        assert priors["EOS"].ncategories == self.n_eos

    def test_an_unweighted_prior_is_uniform_over_the_set(self):
        priors = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(eos_data=str(self.eos_dir), Neos=None, eos_weight=None),
            PriorDict(),
        )
        for index in range(self.n_eos):
            assert priors["EOS"].prob(index) == pytest.approx(1.0 / self.n_eos)

    def test_weights_are_read_from_file_and_set_the_prior(self):
        weight_path = self.tmp_dir / "weights.dat"
        np.savetxt(weight_path, np.array([0.1, 0.2, 0.3, 0.4]))
        priors = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(
                eos_data=str(self.eos_dir), Neos=None, eos_weight=str(weight_path)
            ),
            PriorDict(),
        )
        assert priors["EOS"].ncategories == 4
        assert priors["EOS"].prob(0) == pytest.approx(0.1)
        assert priors["EOS"].prob(3) == pytest.approx(0.4)

    def test_the_weight_file_alone_determines_the_number_of_categories(self):
        weight_path = self.tmp_dir / "weights.dat"
        np.savetxt(weight_path, np.array([0.5, 0.5]))
        priors = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(
                eos_data=str(self.eos_dir), Neos=None, eos_weight=str(weight_path)
            ),
            PriorDict(),
        )
        assert priors["EOS"].ncategories == 2

    def test_existing_priors_are_kept(self):
        priors = PriorDict()
        priors["other"] = 1.0
        returned = eos_likelihood.setup_tabulated_eos_priors(
            Namespace(eos_data=str(self.eos_dir), Neos=2, eos_weight=None), priors
        )
        assert returned is priors
        assert "other" in returned

    def test_a_logger_is_told_about_the_sampling_mode(self):
        logger = MagicMock()
        eos_likelihood.setup_tabulated_eos_priors(
            Namespace(eos_data=str(self.eos_dir), Neos=2, eos_weight=None),
            PriorDict(),
            logger=logger,
        )
        logger.info.assert_called_once()


class TestReadConstraintFromArgs:
    """Constraints may arrive as a ready-made dict or as parallel lists of
    names, masses, errors and references."""

    def test_a_prepared_dict_is_returned_unchanged(self):
        prepared = {"J0740": {"mass": 2.08, "error": 0.07}}
        args = Namespace(lower_mtov=prepared)
        assert eos_likelihood.read_constraint_from_args(args, "lower_mtov") is prepared

    def test_a_dict_given_as_a_string_is_evaluated(self):
        args = Namespace(lower_mtov="{'J0740': {'mass': 2.08}}")
        assert eos_likelihood.read_constraint_from_args(args, "lower_mtov") == {
            "J0740": {"mass": 2.08}
        }

    def test_parallel_lists_are_zipped_into_one_dict_per_name(self):
        args = Namespace(
            lower_mtov=None,
            lower_mtov_name=["J0740", "J0348"],
            lower_mtov_mass=[2.08, 2.01],
            lower_mtov_error=[0.07, 0.04],
            lower_mtov_arxiv=None,
        )
        assert eos_likelihood.read_constraint_from_args(args, "lower_mtov") == {
            "J0740": {"mass": 2.08, "error": 0.07},
            "J0348": {"mass": 2.01, "error": 0.04},
        }

    def test_optional_properties_are_carried_along(self):
        args = Namespace(
            lower_mtov=None,
            lower_mtov_name=["J0740"],
            lower_mtov_mass=[2.08],
            lower_mtov_arxiv=["2104.00880"],
        )
        parsed = eos_likelihood.read_constraint_from_args(args, "lower_mtov")
        assert parsed["J0740"]["arxiv"] == "2104.00880"

    def test_properties_that_are_not_set_are_dropped(self):
        args = Namespace(
            lower_mtov=None,
            lower_mtov_name=["J0740"],
            lower_mtov_mass=[2.08],
            lower_mtov_error=None,
        )
        parsed = eos_likelihood.read_constraint_from_args(args, "lower_mtov")
        assert parsed == {"J0740": {"mass": 2.08}}

    def test_a_property_list_of_the_wrong_length_is_reported(self):
        args = Namespace(
            lower_mtov=None,
            lower_mtov_name=["J0740", "J0348"],
            lower_mtov_mass=[2.08],
        )
        with pytest.raises(ValueError) as context:
            eos_likelihood.read_constraint_from_args(args, "lower_mtov")
        assert "lower_mtov" in str(context.value)

    def test_without_names_nothing_is_built(self):
        args = Namespace(lower_mtov=None, lower_mtov_mass=[2.08])
        assert eos_likelihood.read_constraint_from_args(args, "lower_mtov") is None

    def test_an_absent_constraint_kind_gives_none(self):
        assert eos_likelihood.read_constraint_from_args(Namespace(), "mass_radius") is None

    def test_mass_radius_file_paths_are_read_the_same_way(self):
        args = Namespace(
            mass_radius=None,
            mass_radius_name=["NICER"],
            mass_radius_file_path=["posterior.dat"],
        )
        assert eos_likelihood.read_constraint_from_args(args, "mass_radius") == {
            "NICER": {"file_path": "posterior.dat"}
        }


class TestComposeEoSConstraints:
    """compose_eos_constraints merges a stored constraint file with whatever
    the command line adds and writes the merged set back."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.json_path = self.tmp_dir / "constraints.json"

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def write_json(self, content):
        with open(self.json_path, "w") as stream:
            json.dump(content, stream)

    def base_args(self, **kwargs):
        args = Namespace(
            eos_constraint_json=str(self.json_path),
            lower_mtov=None,
            upper_mtov=None,
            mass_radius=None,
        )
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_a_stored_constraint_file_is_read(self):
        self.write_json({"upper_mtov": {"GW170817": {"mass": 2.3, "error": 0.1}}})
        composed = eos_likelihood.compose_eos_constraints(self.base_args())
        assert composed == {"upper_mtov": {"GW170817": {"mass": 2.3, "error": 0.1}}}

    def test_command_line_constraints_are_merged_into_the_stored_ones(self):
        self.write_json({"upper_mtov": {"GW170817": {"mass": 2.3}}})
        args = self.base_args(lower_mtov={"J0740": {"mass": 2.08}})
        composed = eos_likelihood.compose_eos_constraints(args)
        assert sorted(composed) == ["lower_mtov", "upper_mtov"]

    def test_a_new_constraint_of_a_stored_kind_extends_it(self):
        self.write_json({"lower_mtov": {"J0348": {"mass": 2.01}}})
        args = self.base_args(lower_mtov={"J0740": {"mass": 2.08}})
        composed = eos_likelihood.compose_eos_constraints(args)
        assert sorted(composed["lower_mtov"]) == ["J0348", "J0740"]

    def test_the_merged_set_is_written_back_to_the_file(self):
        self.write_json({})
        args = self.base_args(lower_mtov={"J0740": {"mass": 2.08}})
        eos_likelihood.compose_eos_constraints(args)
        with open(self.json_path) as stream:
            assert json.load(stream) == {"lower_mtov": {"J0740": {"mass": 2.08}}}

    def test_a_missing_file_is_not_an_error(self):
        args = Namespace(
            eos_constraint_json=None,
            lower_mtov={"J0740": {"mass": 2.08}},
            upper_mtov=None,
            mass_radius=None,
        )
        composed = eos_likelihood.compose_eos_constraints(args)
        assert composed == {"lower_mtov": {"J0740": {"mass": 2.08}}}

    def test_no_constraints_at_all_gives_an_empty_dict(self):
        assert (
            eos_likelihood.compose_eos_constraints(Namespace(eos_constraint_json=None))
            == {}
        )

    def test_only_the_requested_kinds_are_considered(self):
        args = self.base_args(lower_mtov={"J0740": {"mass": 2.08}})
        composed = eos_likelihood.compose_eos_constraints(
            args, constraint_kinds=["upper_mtov"]
        )
        assert composed == {}


class TestEoSConstraintBase:
    """The base class only carries identification and plotting metadata."""

    def test_a_named_constraint_describes_itself_by_its_source(self):
        constraint = eos_likelihood.EoSConstraint(name="J0740")
        assert constraint.name == "J0740"
        assert repr(constraint).strip() == "EoSConstraint based on J0740"

    def test_an_arxiv_reference_is_appended_to_the_representation(self):
        constraint = eos_likelihood.EoSConstraint(name="J0740", arxiv_ref="2104.00880")
        assert "arxiv:2104.00880" in repr(constraint)

    def test_an_unnamed_constraint_falls_back_to_its_class_name(self):
        constraint = eos_likelihood.EoSConstraint()
        assert constraint.name == "EoSConstraint"

    def test_the_type_marks_the_constraint_as_macroscopic(self):
        assert eos_likelihood.EoSConstraint().type == "macro"

    def test_plot_keywords_default_to_an_empty_dict(self):
        assert eos_likelihood.EoSConstraint().plot_kwargs == {}

    def test_plot_keywords_are_kept(self):
        constraint = eos_likelihood.EoSConstraint(plot_kwargs={"color": "red"})
        assert constraint.plot_kwargs == {"color": "red"}


class TestMassConstraints:
    """A maximum-mass measurement enters as a one-sided Gaussian: a lower
    limit through the normal CDF and an upper limit through its survival
    function."""

    def test_a_lower_limit_uses_the_cumulative_distribution(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.04, name="J0740")
        assert constraint.lognorm_method == norm.logcdf
        assert constraint.log_likelihood({"TOV_mass": 2.2}) == pytest.approx(
            norm.logcdf(2.2, loc=2.0, scale=0.04)
        )

    def test_an_upper_limit_uses_the_survival_function(self):
        constraint = eos_likelihood.UpperMTOVConstraint(2.3, 0.1)
        assert constraint.lognorm_method == norm.logsf
        assert constraint.log_likelihood({"TOV_mass": 2.2}) == pytest.approx(
            norm.logsf(2.2, loc=2.3, scale=0.1)
        )

    def test_a_lower_limit_rewards_a_stiffer_equation_of_state(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.05)
        soft = constraint.log_likelihood({"TOV_mass": 1.8})
        stiff = constraint.log_likelihood({"TOV_mass": 2.3})
        assert soft < stiff

    def test_an_upper_limit_rewards_a_softer_equation_of_state(self):
        constraint = eos_likelihood.UpperMTOVConstraint(2.3, 0.1)
        soft = constraint.log_likelihood({"TOV_mass": 1.8})
        stiff = constraint.log_likelihood({"TOV_mass": 2.8})
        assert soft > stiff

    def test_the_mass_and_error_are_stored_and_shown(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.08, 0.07, name="J0740")
        assert constraint.mass == 2.08
        assert constraint.error == 0.07
        assert repr(constraint) == "LowerMTOVConstraint of 2.08+-0.07 M_sun based on J0740"

    def test_the_two_limits_are_drawn_with_different_line_styles(self):
        assert eos_likelihood.LowerMTOVConstraint(2.0, 0.1).linestyle == "--"
        assert eos_likelihood.UpperMTOVConstraint(2.3, 0.1).linestyle == ":"

    def test_the_tov_mass_is_taken_from_the_macro_eos_when_not_supplied(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.05)
        masses = np.linspace(1.0, 2.1, 12)
        assert constraint.log_likelihood({}, {"masses": masses}) == pytest.approx(
            norm.logcdf(2.1, loc=2.0, scale=0.05)
        )

    def test_a_list_of_macro_eos_masses_gives_one_value_per_eos(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.05)
        masses = [np.linspace(1.0, 2.1, 12), np.linspace(1.0, 1.9, 12)]
        log_likelihood = constraint.log_likelihood({}, {"masses": masses})
        assert np.shape(log_likelihood) == (2,)
        assert log_likelihood[0] > log_likelihood[1]

    def test_an_array_of_tov_masses_is_evaluated_elementwise(self):
        constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.05)
        log_likelihood = constraint.log_likelihood({"TOV_mass": np.array([1.9, 2.1])})
        np.testing.assert_allclose(
            log_likelihood, norm.logcdf(np.array([1.9, 2.1]), loc=2.0, scale=0.05)
        )

    def test_the_legacy_names_still_resolve_to_the_general_constraints(self):
        assert issubclass(
            eos_likelihood.PulsarConstraint, eos_likelihood.LowerMTOVConstraint
        )
        assert issubclass(
            eos_likelihood.MTOVUpperConstraint, eos_likelihood.UpperMTOVConstraint
        )
        assert issubclass(
            eos_likelihood.JointConstraint, eos_likelihood.JointEoSConstraint
        )


class TestMassConstraintPlot(PlotRcParamsMixin):
    """The constraint draws itself as a horizontal line with a label into an
    existing mass-radius figure."""

    def setup_method(self):
        super().setup_method()
        self.figure, self.axes = plt.subplots()
        self.axes.set_xlim(9.0, 15.0)
        self.axes.set_ylim(1.0, 2.5)
        self.constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.05, name="J0740")

    def test_the_axes_are_returned(self):
        assert self.constraint.plot(self.axes) is self.axes

    def test_a_line_is_drawn_at_the_measured_mass(self):
        self.constraint.plot(self.axes)
        assert len(self.axes.collections) == 1

    def test_the_name_is_written_next_to_the_line(self):
        self.constraint.plot(self.axes)
        assert [text.get_text() for text in self.axes.texts] == ["J0740"]

    def test_an_explicit_colour_overrides_the_cycle(self):
        self.constraint.plot(self.axes, color="red")
        assert self.axes.texts[0].get_color() == "red"

    def test_stored_plot_keywords_are_used(self):
        constraint = eos_likelihood.LowerMTOVConstraint(
            2.0, 0.05, name="J0740", plot_kwargs={"color": "green"}
        )
        constraint.plot(self.axes)
        assert self.axes.texts[0].get_color() == "green"

    def test_keyword_arguments_win_over_stored_plot_keywords(self):
        constraint = eos_likelihood.LowerMTOVConstraint(
            2.0, 0.05, name="J0740", plot_kwargs={"color": "green"}
        )
        constraint.plot(self.axes, color="blue")
        assert self.axes.texts[0].get_color() == "blue"

    def test_the_x_limits_are_left_untouched(self):
        limits = self.axes.get_xlim()
        self.constraint.plot(self.axes)
        assert self.axes.get_xlim() == limits


class TestMassRadiusConstraintDataReading:
    """A mass-radius posterior is read from file and its two columns are
    identified by their value ranges rather than by their order."""

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.mass_first = cls.tmp_dir / "mass_first.dat"
        cls.radius_first = cls.tmp_dir / "radius_first.dat"
        cls.weighted = cls.tmp_dir / "weighted.dat"
        cls.masses, cls.radii = write_mass_radius_posterior(cls.mass_first)
        write_mass_radius_posterior(cls.radius_first, columns="radius_first")
        write_mass_radius_posterior(cls.weighted, columns="weighted")

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_masses_and_radii_are_identified_when_mass_comes_first(self):
        constraint = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        masses, radii, weights = constraint.read_data(str(self.mass_first))
        np.testing.assert_allclose(masses, self.masses)
        np.testing.assert_allclose(radii, self.radii)
        assert weights is None

    def test_the_column_order_does_not_matter(self):
        mass_first = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        radius_first = eos_likelihood.MassRadiusConstraint(
            file_path=str(self.radius_first)
        )
        np.testing.assert_allclose(mass_first.histogram, radius_first.histogram)

    def test_a_third_column_is_read_as_sample_weights(self):
        _, _, weights = eos_likelihood.MassRadiusConstraint(
            file_path=str(self.weighted)
        ).read_data(str(self.weighted))
        np.testing.assert_allclose(weights, 1.0)

    def test_uniform_weights_reproduce_the_unweighted_histogram(self):
        unweighted = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        weighted = eos_likelihood.MassRadiusConstraint(file_path=str(self.weighted))
        np.testing.assert_allclose(weighted.histogram, unweighted.histogram)

    def test_arrays_can_be_passed_instead_of_a_file(self):
        from_arrays = eos_likelihood.MassRadiusConstraint(
            mass_array=self.masses, radius_array=self.radii, name="NICER"
        )
        from_file = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        np.testing.assert_allclose(from_arrays.histogram, from_file.histogram)

    def test_a_file_path_takes_precedence_over_arrays(self):
        constraint = eos_likelihood.MassRadiusConstraint(
            mass_array=np.zeros(10),
            radius_array=np.zeros(10),
            file_path=str(self.mass_first),
        )
        from_file = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        np.testing.assert_allclose(constraint.histogram, from_file.histogram)

    def test_no_data_at_all_is_reported(self):
        with pytest.raises(ValueError):
            eos_likelihood.MassRadiusConstraint()

    def test_only_one_of_the_two_arrays_is_reported(self):
        with pytest.raises(ValueError):
            eos_likelihood.MassRadiusConstraint(mass_array=self.masses)

    def test_a_file_with_too_many_columns_is_reported(self):
        path = self.tmp_dir / "four_columns.dat"
        np.savetxt(path, np.tile(self.masses[:100, np.newaxis], (1, 4)))
        with pytest.raises(ValueError) as context:
            eos_likelihood.MassRadiusConstraint(file_path=str(path))
        assert "two or three columns" in str(context.value)

    def test_a_transposed_file_is_recognised(self):
        path = self.tmp_dir / "transposed.dat"
        np.savetxt(path, np.column_stack([self.masses, self.radii]).T)
        constraint = eos_likelihood.MassRadiusConstraint(file_path=str(path))
        from_file = eos_likelihood.MassRadiusConstraint(file_path=str(self.mass_first))
        np.testing.assert_allclose(constraint.histogram, from_file.histogram)

    def test_unphysical_masses_are_reported(self):
        # The range check fires when the first column is not all positive but
        # still looks like a mass column next to plausible radii.
        path = self.tmp_dir / "negative_masses.dat"
        masses = np.linspace(-0.5, 2.0, 200)
        np.savetxt(path, np.column_stack([masses, np.full_like(masses, 12.0)]))
        with pytest.raises(ValueError) as context:
            eos_likelihood.MassRadiusConstraint(file_path=str(path))
        assert "Failed to properly identify" in str(context.value)


class TestMassRadiusConstraintGrid:
    """The posterior is turned into a smoothed 2D histogram that acts as the
    likelihood surface."""

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        cls.path = cls.tmp_dir / "posterior.dat"
        cls.masses, cls.radii = write_mass_radius_posterior(cls.path)
        cls.constraint = eos_likelihood.MassRadiusConstraint(
            file_path=str(cls.path), name="NICER"
        )

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_the_histogram_is_indexed_by_radius_then_mass(self):
        assert self.constraint.histogram.shape == (
            len(self.constraint.rad_edges) - 1,
            len(self.constraint.mass_edges) - 1,
        )

    def test_the_histogram_is_a_normalised_probability(self):
        assert self.constraint.histogram.sum() == pytest.approx(1.0, abs=1.5 * 10**-3)
        assert np.all(self.constraint.histogram >= 0.0)

    def test_the_grid_brackets_the_samples(self):
        assert self.constraint.mass_edges[0] < np.median(self.masses)
        assert self.constraint.mass_edges[-1] > np.median(self.masses)
        assert self.constraint.rad_edges[0] < np.median(self.radii)
        assert self.constraint.rad_edges[-1] > np.median(self.radii)

    def test_the_default_step_sizes_set_the_grid_spacing(self):
        assert np.diff(self.constraint.mass_edges)[0] == pytest.approx(0.01)
        assert np.diff(self.constraint.rad_edges)[0] == pytest.approx(0.03)

    def test_the_step_sizes_can_be_overridden(self):
        constraint = eos_likelihood.MassRadiusConstraint(
            file_path=str(self.path), name="NICER"
        )
        constraint.set_grid(
            self.masses, self.radii, None, mass_step=0.02, radius_step=0.06
        )
        assert np.diff(constraint.mass_edges)[0] == pytest.approx(0.02)
        assert np.diff(constraint.rad_edges)[0] == pytest.approx(0.06)

    def test_set_bins_trims_the_tails_of_the_sample(self):
        bins = self.constraint.set_bins(self.masses, 0.01)
        low, high = np.quantile(self.masses, [0.001, 0.999])
        assert bins[0] == pytest.approx(0.95 * low)
        assert bins[-1] <= 1.05 * high
        assert np.diff(bins)[0] == pytest.approx(0.01)

    def test_set_bins_sensitivity_controls_how_much_is_trimmed(self):
        wide = self.constraint.set_bins(self.masses, 0.01, sensitivity=0.0)
        narrow = self.constraint.set_bins(self.masses, 0.01, sensitivity=0.05)
        assert len(wide) > len(narrow)

    def test_the_test_mass_grid_covers_the_neutron_star_range(self):
        np.testing.assert_allclose(self.constraint.test_masses[0], 1.2)
        np.testing.assert_allclose(self.constraint.test_masses[-1], 2.5)
        assert len(self.constraint.test_masses) == 151

    def test_a_sparse_posterior_is_flagged(self):
        with patch("builtins.print") as printed:
            eos_likelihood.MassRadiusConstraint(
                mass_array=np.random.default_rng(1).normal(1.4, 0.1, 200),
                radius_array=np.random.default_rng(2).normal(12.0, 0.5, 200),
            )
        printed.assert_called_once()
        assert "sparsely populated" in printed.call_args[0][0]


class TestMassRadiusConstraintLikelihood:
    """The log likelihood integrates the smoothed posterior along the
    mass-radius curve of the proposed EOS."""

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        path = cls.tmp_dir / "posterior.dat"
        write_mass_radius_posterior(path)
        cls.constraint = eos_likelihood.MassRadiusConstraint(
            file_path=str(path), name="NICER"
        )
        cls.masses = np.linspace(0.5, 2.2, 50)
        cls.radii = 12.0 - 0.1 * cls.masses

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_a_curve_through_the_posterior_is_preferred(self):
        matching = self.constraint.single_logl(2.2, self.masses, self.radii)
        offset = self.constraint.single_logl(2.2, self.masses, self.radii - 3.0)
        assert matching > offset
        assert matching < 0.0

    def test_the_tov_mass_limits_the_part_of_the_curve_that_is_used(self):
        full = self.constraint.single_logl(2.2, self.masses, self.radii)
        truncated = self.constraint.single_logl(1.3, self.masses, self.radii)
        assert full > truncated

    def test_log_likelihood_uses_the_supplied_tov_mass(self):
        assert self.constraint.log_likelihood(
            {"TOV_mass": 2.2}, {"masses": self.masses, "radii": self.radii}
        ) == pytest.approx(self.constraint.single_logl(2.2, self.masses, self.radii))

    def test_log_likelihood_falls_back_to_the_last_tabulated_mass(self):
        assert self.constraint.log_likelihood(
            {}, {"masses": self.masses, "radii": self.radii}
        ) == pytest.approx(
            self.constraint.single_logl(self.masses[-1], self.masses, self.radii)
        )

    @pytest.mark.xfail(strict=True)
    def test_several_macro_eos_curves_give_one_value_each(self):
        # The multi-EOS branch is only reached from the except clause, where
        # it calls single_logl with the same arguments that raised, so the
        # original ValueError escapes instead of a list of log likelihoods.
        log_likelihoods = self.constraint.log_likelihood(
            {},
            {
                "masses": [self.masses, self.masses],
                "radii": [self.radii, self.radii - 3.0],
            },
        )
        assert len(log_likelihoods) == 2


class TestMassRadiusConstraintPlot(PlotRcParamsMixin):
    """The constraint is drawn as labelled credible-region contours."""

    @classmethod
    def setup_class(cls):
        cls.tmp_dir = Path(tempfile.mkdtemp())
        path = cls.tmp_dir / "posterior.dat"
        write_mass_radius_posterior(path)
        cls.path = path

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp_dir)

    def setup_method(self):
        super().setup_method()
        self.constraint = eos_likelihood.MassRadiusConstraint(
            file_path=str(self.path), name="NICER"
        )
        self.figure, self.axes = plt.subplots()
        self.axes.set_xlim(9.0, 15.0)
        self.axes.set_ylim(1.0, 2.5)

    def test_the_axes_are_returned(self):
        assert self.constraint.plot(self.axes) is self.axes

    def test_two_credible_contours_are_drawn(self):
        self.constraint.plot(self.axes)
        assert len(self.axes.collections) == 1
        assert len(self.axes.texts) == 1
        assert self.axes.texts[0].get_text() == "NICER"

    def test_an_explicit_colour_is_used(self):
        assert self.constraint.plot(self.axes, color="red") is self.axes

    def test_a_manual_label_position_is_accepted_as_a_pair(self):
        assert self.constraint.plot(self.axes, manual=(12.0, 1.4)) is self.axes

    def test_a_manual_label_position_is_accepted_as_a_list_of_pairs(self):
        assert self.constraint.plot(self.axes, manual=[(12.0, 1.4)]) is self.axes

    def test_a_manual_position_of_the_wrong_shape_is_reported(self):
        with pytest.raises(AssertionError):
            self.constraint.plot(self.axes, manual=[(12.0, 1.4, 0.0)])


class TestJointEoSConstraint:
    """Constraints are combined by summing their log likelihoods; the joint
    object also owns the EOS converter that produces the macro parameters."""

    def setup_method(self):
        self.lower = eos_likelihood.LowerMTOVConstraint(2.0, 0.05, name="J0740")
        self.upper = eos_likelihood.UpperMTOVConstraint(2.3, 0.1, name="GW170817")

    def test_a_single_constraint_is_wrapped(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower)
        assert joint.constraints == [self.lower]

    def test_constraints_are_collected_in_order(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower, self.upper)
        assert joint.constraints == [self.lower, self.upper]

    def test_a_nested_joint_constraint_is_flattened(self):
        inner = eos_likelihood.JointEoSConstraint(self.lower, self.upper)
        outer = eos_likelihood.JointEoSConstraint(inner, self.lower)
        assert outer.constraints == [self.lower, self.upper, self.lower]

    def test_the_log_likelihood_is_the_sum_of_the_parts(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower, self.upper)
        parameters = {"TOV_mass": 2.2}
        assert joint.log_likelihood(parameters) == pytest.approx(
            self.lower.log_likelihood(parameters)
            + self.upper.log_likelihood(parameters)
        )

    def test_without_a_converter_an_empty_macro_parameter_set_is_used(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower)
        assert joint.eos_converter.macro_parameters == {}

    def test_the_parameter_conversion_is_delegated_to_the_converter(self):
        converter = MagicMock()
        joint = eos_likelihood.JointEoSConstraint(self.lower, eos_converter=converter)
        parameters = {"EOS": 1}
        assert joint.parameter_conversion(parameters) is converter.parameter_conversion.return_value
        converter.parameter_conversion.assert_called_once_with(parameters)

    def test_one_constraint_represents_itself(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower)
        assert repr(joint) == repr(self.lower)

    def test_two_constraints_are_joined_by_and(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower, self.upper)
        assert repr(joint) == f"{self.lower!r} and {self.upper!r}"

    def test_more_constraints_are_listed(self):
        joint = eos_likelihood.JointEoSConstraint(self.lower, self.upper, self.lower)
        representation = repr(joint)
        assert representation.startswith("JointEoSConstraint of")
        assert f", {self.upper!r} and {self.lower!r}" in representation

    def test_constraints_are_built_from_a_constraint_dict(self):
        joint = eos_likelihood.JointEoSConstraint(
            {
                "lower_mtov": {"J0740": {"mass": 2.08, "error": 0.07, "arxiv": "1"}},
                "upper_mtov": {"GW170817": {"mass": 2.3, "error": 0.1}},
            }
        )
        assert [type(constraint).__name__ for constraint in joint.constraints] == [
            "LowerMTOVConstraint",
            "UpperMTOVConstraint",
        ]
        assert joint.constraints[0].mass == 2.08
        assert joint.constraints[0].arxiv_ref == "1"
        assert joint.constraints[1].error == 0.1

    def test_a_missing_error_in_the_dict_defaults_to_zero(self):
        joint = eos_likelihood.JointEoSConstraint(
            {"lower_mtov": {"J0740": {"mass": 2.08}}}
        )
        assert joint.constraints[0].error == 0.0

    def test_plot_keywords_are_forwarded_from_the_dict(self):
        joint = eos_likelihood.JointEoSConstraint(
            {"lower_mtov": {"J0740": {"mass": 2.08, "plot_kwargs": {"color": "red"}}}}
        )
        assert joint.constraints[0].plot_kwargs == {"color": "red"}

    def test_mass_radius_constraints_are_built_from_a_file_path(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            path = tmp_dir / "posterior.dat"
            write_mass_radius_posterior(path, num=5000)
            joint = eos_likelihood.JointEoSConstraint(
                {"mass_radius": {"NICER": {"file_path": str(path)}}}
            )
            assert isinstance(
                joint.constraints[0], eos_likelihood.MassRadiusConstraint
            )
            assert joint.constraints[0].name == "NICER"
        finally:
            shutil.rmtree(tmp_dir)

    def test_the_legacy_posterior_key_is_still_accepted(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            path = tmp_dir / "posterior.dat"
            write_mass_radius_posterior(path, num=5000)
            joint = eos_likelihood.JointEoSConstraint(
                {"mass_radius": {"NICER": {"posterior": str(path)}}}
            )
            assert isinstance(
                joint.constraints[0], eos_likelihood.MassRadiusConstraint
            )
        finally:
            shutil.rmtree(tmp_dir)

    def test_an_empty_dict_gives_no_constraints(self):
        assert eos_likelihood.JointEoSConstraint({}).constraints == []


class TestJointEoSConstraintTabulation(MacroEoSSetMixin):
    """tabulate_weighted_eos scores a whole set of tabulated EOSs and stores
    them sorted by their prior weight."""

    def setup_method(self):
        super().setup_method()
        self.out_dir = self.tmp_dir / "out"
        self.joint = eos_likelihood.JointEoSConstraint(
            {"lower_mtov": {"J0740": {"mass": 2.0, "error": 0.1}}},
            eos_converter=self.tabulated_converter(),
        )

    def tabulate(self, parameters, **kwargs):
        # process_map spawns worker processes, which a unit test should not
        # depend on; the serial equivalent exercises the same code.
        with patch.object(
            eos_likelihood,
            "process_map",
            lambda function, iterable, **kw: [function(item) for item in iterable],
        ):
            return self.joint.tabulate_weighted_eos(
                parameters, str(self.out_dir), **kwargs
            )

    def test_every_eos_is_written_out_and_counted(self):
        weight_path, sorted_dir, n_eos = self.tabulate(self.n_eos)
        assert n_eos == self.n_eos
        assert sorted(path.name for path in sorted_dir.iterdir()) == [
            f"{index + 1}.dat" for index in range(self.n_eos)
        ]
        assert Path(weight_path).is_file()

    def test_the_weights_are_normalised_and_sorted_ascending(self):
        weight_path, _, _ = self.tabulate(self.n_eos)
        weights = np.loadtxt(weight_path)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(np.diff(weights) >= 0.0)

    def test_the_stiffest_equation_of_state_gets_the_largest_weight(self):
        weight_path, sorted_dir, _ = self.tabulate(self.n_eos)
        weights = np.loadtxt(weight_path)
        heaviest = np.loadtxt(sorted_dir / f"{len(weights)}.dat")
        assert heaviest[:, 1].max() == pytest.approx(self.tables[-1][0][-1])

    def test_normalisation_can_be_switched_off(self):
        weight_path, _, _ = self.tabulate(self.n_eos, normalise=False)
        weights = np.loadtxt(weight_path)
        assert np.all(weights <= 1.0)
        assert weights.sum() != pytest.approx(1.0)

    def test_an_existing_tabulation_is_reused(self):
        first = self.tabulate(self.n_eos)
        with patch.object(eos_likelihood, "process_map") as process_map:
            second = self.joint.tabulate_weighted_eos(None, str(self.out_dir))
        process_map.assert_not_called()
        assert first[2] == second[2]

    def test_the_whole_set_is_scored_when_no_parameters_are_given(self):
        _, _, n_eos = self.tabulate(None)
        assert n_eos == self.n_eos

    def test_previous_weights_are_folded_in(self):
        previous_path = self.tmp_dir / "previous.dat"
        np.savetxt(previous_path, np.full(self.n_eos, 0.5))
        weight_path, _, _ = self.tabulate(
            self.n_eos, weight_path=str(previous_path), normalise=False
        )
        plain_dir = self.tmp_dir / "plain"
        with patch.object(
            eos_likelihood,
            "process_map",
            lambda function, iterable, **kw: [function(item) for item in iterable],
        ):
            plain_path, _, _ = self.joint.tabulate_weighted_eos(
                self.n_eos, str(plain_dir), normalise=False
            )
        np.testing.assert_allclose(
            np.loadtxt(weight_path), 0.5 * np.loadtxt(plain_path)
        )

    def test_eval_eos_data_scores_one_macro_eos(self):
        radii, masses = np.array([12.0, 11.5]), np.array([1.0, 2.1])
        assert self.joint.eval_eos_data((radii, masses, None)) == pytest.approx(
            norm.logcdf(2.1, loc=2.0, scale=0.1)
        )

    def test_eval_eos_data_rejects_data_it_cannot_unpack(self):
        assert self.joint.eval_eos_data((1.0, 2.0)) is None


class TestEquationofStateLikelihood(MacroEoSSetMixin):
    """The likelihood is a thin NMMA wrapper around the joint constraint,
    with the EOS conversion registered as a parameter conversion."""

    def setup_method(self):
        super().setup_method()
        self.priors = PriorDict()
        self.priors["EOS"] = WeightedCategorical(self.n_eos, name="EOS")
        self.converter = self.tabulated_converter()
        self.likelihood = eos_likelihood.EquationofStateLikelihood(
            self.priors,
            {"lower_mtov": {"J0740": {"mass": 2.0, "error": 0.05}}},
            self.converter,
        )

    def test_the_constraint_dict_becomes_a_joint_constraint(self):
        assert isinstance(
            self.likelihood.sub_model, eos_likelihood.JointEoSConstraint
        )
        assert len(self.likelihood.sub_model.constraints) == 1

    def test_the_representation_names_the_constraint(self):
        assert repr(self.likelihood) == (
            "EquationofStateLikelihood with "
            "LowerMTOVConstraint of 2.0+-0.05 M_sun based on J0740"
        )

    def test_the_eos_conversion_is_registered(self):
        assert self.likelihood.conv_functions == [
            self.likelihood.sub_model.parameter_conversion
        ]

    def test_the_log_likelihood_scores_the_drawn_equation_of_state(self):
        log_likelihood = self.likelihood.log_likelihood({"EOS": 0})
        assert np.isfinite(log_likelihood)
        assert log_likelihood < 0.0

    def test_a_stiffer_equation_of_state_is_preferred_by_a_lower_limit(self):
        soft = self.likelihood.log_likelihood({"EOS": 0})
        stiff = self.likelihood.log_likelihood({"EOS": self.n_eos - 1})
        assert soft < stiff

    def test_the_conversion_populates_the_neutron_star_parameters(self):
        parameters = self.likelihood.parameter_conversion({"EOS": 1})
        for key in ["TOV_mass", "TOV_radius", "R_14", "R_16"]:
            assert key in parameters

    def test_there_is_no_noise_log_likelihood_to_subtract(self):
        assert self.likelihood.noise_log_likelihood() == 0.0

    def test_the_priors_carry_no_constraints(self):
        assert self.likelihood.constraints == {}


class TestTabulatedEoSSetup(MacroEoSSetMixin):
    """tabulated_eos_setup wires priors, converter and likelihood together
    for a standalone EOS run over precomputed tables."""

    def setup_method(self):
        super().setup_method()
        self.args = Namespace(
            eos_data=str(self.eos_dir),
            Neos=self.n_eos,
            eos_weight=None,
            eos_to_ram=True,
            eos_file=None,
            emulator_metadata=None,
            eos_constraint_json=None,
            lower_mtov={"J0740": {"mass": 2.0, "error": 0.05}},
            upper_mtov=None,
            mass_radius=None,
        )

    def test_the_prior_the_likelihood_and_no_extra_data_are_returned(self):
        priors, likelihood, extra = eos_likelihood.tabulated_eos_setup(self.args)
        assert priors["EOS"].ncategories == self.n_eos
        assert isinstance(likelihood, eos_likelihood.EquationofStateLikelihood)
        assert extra is None

    def test_neos_is_taken_from_the_prior(self):
        args = Namespace(**vars(self.args))
        args.Neos = None
        eos_likelihood.tabulated_eos_setup(args)
        assert args.Neos == self.n_eos

    def test_the_conversion_stops_at_the_macro_parameters(self):
        # A standalone EOS run has no binary, so the system parameters that
        # need component masses are not computed.
        _, likelihood, _ = eos_likelihood.tabulated_eos_setup(self.args)
        converter = likelihood.sub_model.eos_converter
        assert converter.parameter_conversion == converter.compute_macro_parameters

    def test_the_likelihood_can_be_evaluated_on_a_prior_sample(self):
        priors, likelihood, _ = eos_likelihood.tabulated_eos_setup(self.args)
        sample = priors.sample()
        assert np.isfinite(likelihood.log_likelihood(sample))

    def test_the_constraints_from_the_arguments_are_used(self):
        args = Namespace(**vars(self.args))
        args.upper_mtov = {"GW170817": {"mass": 2.3, "error": 0.1}}
        _, likelihood, _ = eos_likelihood.tabulated_eos_setup(args)
        assert len(likelihood.sub_model.constraints) == 2


class TestSetupEoSKwargs:
    """setup_eos_kwargs is the generation-stage helper that turns a data
    dump plus arguments into the likelihood keyword arguments."""

    def test_the_constraints_and_an_emulated_converter_are_returned(self):
        args = Namespace(emulator_metadata={"emulator_path": "path"})
        data_dump = {"eos_constraint_dict": {"lower_mtov": {"J0740": {"mass": 2.0}}}}
        with patch(
            "nmma.eos.eos_processing.setup_eos_generator"
        ) as setup_eos_generator:
            kwargs = eos_likelihood.setup_eos_kwargs(data_dump, args, MagicMock())
        assert kwargs["constraint_dict"] == data_dump["eos_constraint_dict"]
        assert kwargs["eos_converter"].tov_emulator is setup_eos_generator.return_value

    def test_the_returned_kwargs_build_a_likelihood(self):
        args = Namespace(emulator_metadata={"emulator_path": "path"})
        data_dump = {
            "eos_constraint_dict": {
                "lower_mtov": {"J0740": {"mass": 2.0, "error": 0.05}}
            }
        }
        with patch("nmma.eos.eos_processing.setup_eos_generator"):
            kwargs = eos_likelihood.setup_eos_kwargs(data_dump, args, MagicMock())
        likelihood = eos_likelihood.EquationofStateLikelihood(PriorDict(), **kwargs)
        assert len(likelihood.sub_model.constraints) == 1


class TestFinalDiagnostics(PlotRcParamsMixin, MacroEoSSetMixin):
    """The diagnostic plot draws the best-fit mass-radius curve together
    with every constraint, and optionally the posterior band."""

    def setup_method(self):
        super().setup_method()
        posterior_path = self.tmp_dir / "posterior.dat"
        write_mass_radius_posterior(posterior_path, num=5000)
        self.out_dir = self.tmp_dir / "out"
        self.out_dir.mkdir()
        self.args = Namespace(
            eos_data=str(self.eos_dir),
            Neos=self.n_eos,
            eos_weight=None,
            eos_to_ram=True,
            eos_file=None,
            emulator_metadata=None,
            eos_constraint_json=None,
            lower_mtov={"J0740": {"mass": 2.0, "error": 0.05}},
            upper_mtov={"GW170817": {"mass": 2.3, "error": 0.1}},
            mass_radius={"NICER": {"file_path": str(posterior_path)}},
        )
        _, self.likelihood, _ = eos_likelihood.tabulated_eos_setup(self.args)
        self.plot_args = Namespace(outdir=str(self.out_dir), label="run", fig=None)

    def test_a_figure_is_returned_and_saved(self):
        figure = self.likelihood.final_diagnostics({"EOS": 2}, self.plot_args)
        assert isinstance(figure, plt.Figure)
        assert (self.out_dir / "run_mr_curve.png").is_file()

    def test_every_constraint_is_labelled_in_the_figure(self):
        figure = self.likelihood.final_diagnostics({"EOS": 2}, self.plot_args)
        labels = [text.get_text() for text in figure.axes[0].texts]
        assert sorted(labels) == ["GW170817", "J0740", "NICER"]

    def test_the_axes_cover_the_mass_radius_curve(self):
        figure = self.likelihood.final_diagnostics({"EOS": 2}, self.plot_args)
        radii, masses, _ = (
            self.likelihood.sub_model.eos_converter.macro_parameters.values()
        )
        axes = figure.axes[0]
        assert axes.get_xlim()[0] <= np.min(radii)
        assert axes.get_ylim()[1] >= masses[-1]

    def test_an_existing_figure_is_reused_rather_than_replaced(self):
        first = self.likelihood.final_diagnostics({"EOS": 2}, self.plot_args)
        reuse_args = Namespace(outdir=str(self.out_dir), label="second", fig=first)
        second = self.likelihood.final_diagnostics({"EOS": 3}, reuse_args)
        assert second is first
        assert (self.out_dir / "second_mr_curve.png").is_file()

    def test_constraints_are_not_drawn_twice_into_a_reused_figure(self):
        first = self.likelihood.final_diagnostics({"EOS": 2}, self.plot_args)
        labels = [text.get_text() for text in first.axes[0].texts]
        reuse_args = Namespace(outdir=str(self.out_dir), label="second", fig=first)
        second = self.likelihood.final_diagnostics({"EOS": 3}, reuse_args)
        assert [text.get_text() for text in second.axes[0].texts] == labels

    def test_a_posterior_adds_credible_bands_and_the_injection(self):
        result = Namespace(
            posterior={"EOS": np.arange(self.n_eos)},
            injection_parameters={"EOS": 1},
        )
        figure = self.likelihood.final_diagnostics(
            {"EOS": 2}, self.plot_args, result=result
        )
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert "Injection" in labels
        assert "run" in labels

    def test_a_posterior_without_an_injection_is_accepted(self):
        result = Namespace(
            posterior={"EOS": np.arange(self.n_eos)}, injection_parameters=None
        )
        figure = self.likelihood.final_diagnostics(
            {"EOS": 2}, self.plot_args, result=result
        )
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert "Injection" not in labels


class TestLegacyTabulatedWeighting:
    """The legacy free functions score a directory of tabulated EOSs without
    going through the converter."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.macro_dir = self.tmp_dir / "macro"
        self.micro_dir = self.tmp_dir / "micro"
        self.macro_dir.mkdir()
        self.micro_dir.mkdir()
        self.tables = [
            write_macro_eos_file(self.macro_dir / f"eos{index}.dat", 1.0 + 0.05 * index)
            for index in range(3)
        ]
        number_density = np.linspace(0.05, 1.0, 10)
        for index in range(3):
            np.savetxt(
                self.micro_dir / f"eos{index}.dat",
                np.column_stack(
                    [
                        number_density,
                        10.0 * number_density**2,
                        number_density * 939.0,
                    ]
                ),
            )
        self.constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.1, name="J0740")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_macroscopic_weights_are_normalised_and_ordered_by_stiffness(self):
        weights, files = eos_likelihood.weights_for_tabulated_eos_from_constraints(
            macro_constraints=self.constraint,
            macro_eos_path=str(self.macro_dir),
            eos_identifier="eos",
        )
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(np.diff(weights) > 0.0)
        assert [path.name for path in files] == ["eos0.dat", "eos1.dat", "eos2.dat"]

    def test_unnormalised_weights_are_plain_likelihoods(self):
        weights, _ = eos_likelihood.weights_for_tabulated_eos_from_constraints(
            macro_constraints=self.constraint,
            macro_eos_path=str(self.macro_dir),
            eos_identifier="eos",
            normalise=False,
        )
        masses = self.tables[0][0]
        assert weights[0] == pytest.approx(
            np.exp(norm.logcdf(masses[-1], loc=2.0, scale=0.1))
        )

    def test_the_weights_can_be_written_to_file(self):
        save_path = self.tmp_dir / "weights.dat"
        weights, _ = eos_likelihood.weights_for_tabulated_eos_from_constraints(
            macro_constraints=self.constraint,
            macro_eos_path=str(self.macro_dir),
            eos_identifier="eos",
            save_path=str(save_path),
        )
        np.testing.assert_allclose(np.loadtxt(save_path), weights)

    def test_microscopic_and_macroscopic_log_weights_are_added(self):
        micro_constraint = MagicMock()
        micro_constraint.log_likelihood.side_effect = [-1.0, -2.0, -3.0]
        combined, files = eos_likelihood.weights_for_tabulated_eos_from_constraints(
            macro_constraints=self.constraint,
            micro_constraints=micro_constraint,
            macro_eos_path=str(self.macro_dir),
            micro_eos_path=str(self.micro_dir),
            eos_identifier="eos",
            normalise=False,
        )
        macro_only, _ = eos_likelihood.weights_for_tabulated_eos_from_constraints(
            macro_constraints=self.constraint,
            macro_eos_path=str(self.macro_dir),
            eos_identifier="eos",
            normalise=False,
        )
        np.testing.assert_allclose(combined, macro_only * np.exp([-1.0, -2.0, -3.0]))
        assert files[0].parent.name == "micro"

    def test_unequal_numbers_of_micro_and_macro_files_are_reported(self):
        write_macro_eos_file(self.macro_dir / "eos3.dat")
        with pytest.raises(ValueError) as context:
            eos_likelihood.weights_for_tabulated_eos_from_constraints(
                macro_constraints=self.constraint,
                micro_constraints=MagicMock(),
                macro_eos_path=str(self.macro_dir),
                micro_eos_path=str(self.micro_dir),
                eos_identifier="eos",
            )
        assert "unequal numbers" in str(context.value)

    def test_the_default_identifier_matches_the_directory_itself(self):
        # An empty identifier becomes the pattern "**", which yields the
        # directory before its contents, so an identifier is mandatory in
        # practice.
        with pytest.raises(IsADirectoryError):
            eos_likelihood.weights_for_tabulated_eos_from_constraints(
                macro_constraints=self.constraint,
                macro_eos_path=str(self.macro_dir),
            )

    @pytest.mark.xfail(strict=True)
    def test_microscopic_constraints_can_be_used_on_their_own(self):
        # The micro-only branch never assigns eos_files, so returning the
        # file list raises UnboundLocalError.
        eos_likelihood.weights_for_tabulated_eos_from_constraints(
            micro_constraints=MagicMock(),
            micro_eos_path=str(self.micro_dir),
            eos_identifier="eos",
        )

    def test_a_macro_file_is_scored_on_its_maximum_mass(self):
        log_weight = eos_likelihood.constraint_weight_from_macro_eos_file(
            self.macro_dir / "eos0.dat", self.constraint
        )
        assert log_weight == pytest.approx(
            norm.logcdf(self.tables[0][0][-1], loc=2.0, scale=0.1)
        )

    def test_a_micro_file_is_passed_on_as_number_density_and_pressure(self):
        constraint = MagicMock()
        constraint.log_likelihood.return_value = -1.0
        eos_likelihood.constraint_weight_from_micro_eos_file(
            self.micro_dir / "eos0.dat", constraint
        )
        passed = constraint.log_likelihood.call_args[0][0]
        assert sorted(passed) == ["energy_density", "number_density", "pressur"]

    def test_a_single_constraint_object_is_evaluated(self):
        assert eos_likelihood.eos_weight_from_constraints(
            {"TOV_mass": 2.1}, self.constraint
        ) == pytest.approx(self.constraint.log_likelihood({"TOV_mass": 2.1}))

    def test_several_constraint_objects_are_summed(self):
        upper = eos_likelihood.UpperMTOVConstraint(2.3, 0.1)
        assert eos_likelihood.eos_weight_from_constraints(
            {"TOV_mass": 2.1}, self.constraint, upper
        ) == pytest.approx(
            self.constraint.log_likelihood({"TOV_mass": 2.1})
            + upper.log_likelihood({"TOV_mass": 2.1})
        )

    def test_a_list_of_constraints_is_not_accepted(self):
        # The callers pass their constraints as one positional argument, so a
        # list arrives as a single "constraint" without a log_likelihood.
        with pytest.raises(AttributeError):
            eos_likelihood.eos_weight_from_constraints(
                {"TOV_mass": 2.1}, [self.constraint, self.constraint]
            )


class TestEOSSorting:
    """EOSSorting copies EOS files into a directory named by their rank in
    the sorting quantity."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.tmp_dir / "source"
        self.out_dir = self.tmp_dir / "sorted"
        self.source_dir.mkdir()
        self.out_dir.mkdir()
        self.tables = [
            write_macro_eos_file(
                self.source_dir / f"eos{index}.dat", 1.0 + 0.05 * index
            )
            for index in range(3)
        ]
        # The routine appends the suffix itself, so the stems are passed in.
        self.stems = [str(self.source_dir / f"eos{index}") for index in range(3)]

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_every_file_is_copied_under_a_one_based_index(self):
        eos_likelihood.EOSSorting(self.stems, str(self.out_dir), [0.2, 0.5, 0.3])
        assert sorted(path.name for path in self.out_dir.iterdir()) == [
            "1.dat",
            "2.dat",
            "3.dat",
        ]

    def test_the_index_follows_the_rank_of_the_sorting_quantity(self):
        eos_likelihood.EOSSorting(self.stems, str(self.out_dir), [0.2, 0.5, 0.3])
        np.testing.assert_allclose(
            np.loadtxt(self.out_dir / "1.dat"),
            np.loadtxt(self.source_dir / "eos0.dat"),
        )
        np.testing.assert_allclose(
            np.loadtxt(self.out_dir / "3.dat"),
            np.loadtxt(self.source_dir / "eos1.dat"),
        )

    def test_file_names_that_already_carry_the_suffix_are_not_found(self):
        with pytest.raises(FileNotFoundError):
            eos_likelihood.EOSSorting(
                [f"{stem}.dat" for stem in self.stems],
                str(self.out_dir),
                [0.2, 0.5, 0.3],
            )


class TestEOSConstraints2Prior:
    """The legacy one-shot helper combines weighting, sorting and prior
    construction."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.macro_dir = self.tmp_dir / "macro"
        self.out_dir = self.tmp_dir / "sorted"
        self.macro_dir.mkdir()
        self.out_dir.mkdir()
        for index in range(3):
            write_macro_eos_file(
                self.macro_dir / f"{index + 1}.dat", 1.0 + 0.05 * index
            )
        self.constraint = eos_likelihood.LowerMTOVConstraint(2.0, 0.1, name="J0740")

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    @pytest.mark.xfail(strict=True)
    def test_a_weighted_categorical_prior_is_returned(self):
        # The helper calls the weighting routine without an EOS identifier,
        # whose "**" pattern picks up the directory itself before any file.
        prior, log_norm = eos_likelihood.EOSConstraints2Prior(
            str(self.macro_dir), str(self.out_dir), self.constraint
        )
        assert isinstance(prior, WeightedCategorical)
        assert prior.ncategories == 3
        assert np.isfinite(log_norm)
