import argparse
import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from nmma.post_processing import plotting_routines as pr  # noqa: E402
from nmma.post_processing.parser import corner_plot_parser  # noqa: E402


def posterior_frame(size=300, seed=11, with_likelihood=True):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "log10_mej_dyn": generator.normal(-2.5, 0.2, size),
            "log10_mej_wind": generator.normal(-1.8, 0.2, size),
            "luminosity_distance": generator.normal(40.0, 3.0, size),
        }
    )
    if with_likelihood:
        frame["log_likelihood"] = generator.normal(-10.0, 1.0, size)
        frame["log_prior"] = np.zeros(size)
    return frame


class FigureMixin:
    """Importing the plotting helpers switches LaTeX rendering on outside CI,
    which needs an installation a developer machine may not have. Each test
    renders with mathtext and restores the global settings."""

    def setup_method(self):
        self.original_rc = matplotlib.rcParams.copy()
        matplotlib.rcParams["text.usetex"] = False

    def teardown_method(self):
        plt.close("all")
        matplotlib.rcParams.update(self.original_rc)


class TestSetupPlotQuantities(FigureMixin):
    """Collects everything a corner or histogram plot needs from a posterior
    table: the samples, the axis labels, the titles and the ranges."""

    def build(self, samples=None, **kwargs):
        settings = dict(limits=None, plot_keys=None, injection=None)
        settings.update(kwargs)
        return pr.setup_plot_quantities(
            posterior_frame() if samples is None else samples, **settings
        )

    def test_every_piece_the_plotters_need_is_returned(self):
        quantities = self.build()
        for key in ["best_fit", "labels", "titles", "samples", "limits", "keys"]:
            assert key in quantities, key

    def test_all_columns_are_plotted_when_no_keys_are_given(self):
        quantities = self.build()
        assert "log10_mej_dyn" in quantities["keys"]
        assert "luminosity_distance" in quantities["keys"]

    def test_the_likelihood_columns_are_left_out_of_the_plot(self):
        # They are bookkeeping rather than inferred parameters.
        quantities = self.build()
        assert "log_likelihood" not in quantities["keys"]
        assert "log_prior" not in quantities["keys"]

    def test_only_the_requested_keys_are_plotted(self):
        quantities = self.build(plot_keys=["log10_mej_dyn"])
        assert quantities["keys"] == ["log10_mej_dyn"]
        assert quantities["samples"].shape[1] == 1

    def test_the_samples_are_stacked_one_column_per_parameter(self):
        quantities = self.build(plot_keys=["log10_mej_dyn", "log10_mej_wind"])
        assert quantities["samples"].shape == (300, 2)

    def test_the_best_fit_sample_is_the_most_likely_one(self):
        samples = posterior_frame()
        quantities = self.build(samples)
        expected = samples.iloc[samples["log_likelihood"].idxmax()]
        assert quantities["best_fit"]["log10_mej_dyn"] == pytest.approx(
            expected["log10_mej_dyn"]
        )

    def test_without_a_likelihood_column_there_is_no_best_fit(self):
        quantities = self.build(posterior_frame(with_likelihood=False))
        assert quantities["best_fit"]["log10_mej_dyn"] is None

    def test_the_limits_are_widened_to_hold_the_samples(self):
        samples = posterior_frame()
        quantities = self.build(samples, plot_keys=["log10_mej_dyn"])
        low, high = quantities["limits"][0]
        assert low <= samples["log10_mej_dyn"].min()
        assert high >= samples["log10_mej_dyn"].max()

    def test_supplied_limits_are_only_ever_widened(self):
        quantities = self.build(plot_keys=["log10_mej_dyn"], limits=[(-10.0, 10.0)])
        low, high = quantities["limits"][0]
        assert low == pytest.approx(-10.0)
        assert high == pytest.approx(10.0)

    def test_a_known_parameter_gets_its_published_label(self):
        quantities = self.build(plot_keys=["luminosity_distance"])
        assert "d_L" in quantities["labels"][0]

    def test_an_unknown_parameter_falls_back_to_its_own_name(self):
        samples = posterior_frame()
        samples["my_parameter"] = np.linspace(0.0, 1.0, len(samples))
        quantities = self.build(samples, plot_keys=["my_parameter"])
        assert quantities["labels"][0] == "my_parameter"

    def test_a_parameter_with_no_spread_cannot_be_titled(self):
        # The title is built from the significant figures of the credible
        # interval, and a fixed parameter has none, so the rounding overflows.
        # A posterior holding a delta-function parameter cannot be plotted.
        samples = posterior_frame()
        samples["fixed"] = 1.0
        with pytest.raises(OverflowError):
            self.build(samples, plot_keys=["fixed"])

    def test_a_supplied_label_beats_the_parameter_name(self):
        samples = posterior_frame()
        samples["my_parameter"] = np.linspace(0, 1, len(samples))
        quantities = self.build(
            samples,
            plot_keys=["my_parameter"],
            default_labels={"my_parameter": "$x$"},
        )
        assert quantities["labels"][0] == "$x$"

    def test_a_missing_parameter_gets_a_blank_off_screen_placeholder(self):
        # Several posteriors are overlaid on one figure, so a parameter only
        # present in some of them still needs a column in the grid.
        quantities = self.build(plot_keys=["log10_mej_dyn", "absent"])
        assert quantities["labels"][1] == ""
        assert quantities["titles"][1] == ""
        assert quantities["samples"].shape[1] == 2

    def test_a_title_is_produced_for_each_present_parameter(self):
        quantities = self.build(plot_keys=["log10_mej_dyn"])
        assert quantities["titles"][0]

    def test_no_injection_means_no_truth_markers(self):
        assert self.build()["truths"] is None

    def test_an_injection_dictionary_becomes_the_truth_markers(self):
        quantities = self.build(
            plot_keys=["log10_mej_dyn"], injection={"log10_mej_dyn": -2.4}
        )
        assert quantities["truths"][0] == pytest.approx(-2.4)

    def test_an_injection_table_is_reduced_to_its_first_row(self):
        injection = pd.DataFrame({"log10_mej_dyn": [-2.4, -9.9]})
        quantities = self.build(plot_keys=["log10_mej_dyn"], injection=injection)
        assert quantities["truths"][0] == pytest.approx(-2.4)

    def test_a_parameter_absent_from_the_injection_gets_no_marker(self):
        quantities = self.build(
            plot_keys=["log10_mej_dyn", "luminosity_distance"],
            injection={"log10_mej_dyn": -2.4},
        )
        assert quantities["truths"][1] is None


class TestCornerPlot(FigureMixin):
    """Draws the corner figure itself, over a set of NMMA defaults that the
    caller can override."""

    def setup_method(self):
        super().setup_method()
        self.samples = np.random.default_rng(0).normal(size=(400, 2))
        self.labels = ["$a$", "$b$"]
        self.limits = [(-4.0, 4.0), (-4.0, 4.0)]

    def test_a_figure_is_returned(self):
        figure = pr.corner_plot(self.samples, self.labels, self.limits)
        assert isinstance(figure, plt.Figure)

    def test_one_panel_is_drawn_per_parameter_pair(self):
        figure = pr.corner_plot(self.samples, self.labels, self.limits)
        assert len(figure.axes) == 4

    def test_the_defaults_can_be_overridden(self):
        figure = pr.corner_plot(
            self.samples, self.labels, self.limits, bins=10, color="C3"
        )
        assert isinstance(figure, plt.Figure)

    def test_an_existing_figure_is_drawn_onto_so_posteriors_can_overlay(self):
        first = pr.corner_plot(self.samples, self.labels, self.limits)
        second = pr.corner_plot(self.samples, self.labels, self.limits, fig=first)
        assert second is first

    def test_the_figure_can_be_saved(self):
        directory = Path(tempfile.mkdtemp())
        try:
            target = directory / "corner.pdf"
            pr.corner_plot(self.samples, self.labels, self.limits, save=str(target))
            assert target.is_file()
        finally:
            shutil.rmtree(directory)

    def test_the_quantiles_default_to_the_one_sigma_band_and_median(self):
        with patch.object(pr.corner, "corner") as corner:
            pr.corner_plot(self.samples, self.labels, self.limits)
        assert corner.call_args.kwargs["quantiles"] == [0.16, 0.5, 0.84]

    def test_the_ranges_are_passed_to_corner_as_its_range_argument(self):
        with patch.object(pr.corner, "corner") as corner:
            pr.corner_plot(self.samples, self.labels, self.limits)
        assert corner.call_args.kwargs["range"] == self.limits


class TestSetupCornerPlot(FigureMixin):
    """Joins the quantity collection to the drawing, which is the path the
    corner-plot script takes for each posterior file."""

    def test_the_figure_and_the_widened_limits_are_returned(self):
        figure, limits = pr.setup_corner_plot(
            posterior_frame(), plot_keys=["log10_mej_dyn", "log10_mej_wind"]
        )
        assert isinstance(figure, plt.Figure)
        assert len(limits) == 2

    def test_the_truth_markers_reach_the_drawing(self):
        # The routine post-processes the real figure afterwards, so the
        # recording has to sit at the corner call and still hand back a
        # genuine figure.
        real_corner = pr.corner.corner
        recorded = {}

        def recording_corner(*args, **kwargs):
            recorded.update(kwargs)
            return real_corner(*args, **kwargs)

        with patch.object(pr.corner, "corner", side_effect=recording_corner):
            pr.setup_corner_plot(
                posterior_frame(),
                plot_keys=["log10_mej_dyn", "log10_mej_wind"],
                injection={"log10_mej_dyn": -2.4, "log10_mej_wind": -1.9},
            )
        assert recorded["truths"][0] == pytest.approx(-2.4)
        assert recorded["truths"][1] == pytest.approx(-1.9)

    def test_an_existing_figure_is_reused_for_overlays(self):
        first, _ = pr.setup_corner_plot(posterior_frame(), plot_keys=["log10_mej_dyn"])
        second, _ = pr.setup_corner_plot(
            posterior_frame(seed=12), plot_keys=["log10_mej_dyn"], fig=first
        )
        assert second is first

    def test_the_limits_come_back_widened_so_overlays_share_a_scale(self):
        # Each posterior is drawn onto the same figure, so the caller feeds
        # the returned limits into the next call.
        _, limits = pr.setup_corner_plot(posterior_frame(), plot_keys=["log10_mej_dyn"])
        low, high = limits[0]
        assert low < high


class TestPlotHistogramsOnly(FigureMixin):
    """A cheaper one-dimensional summary, used when the full corner grid
    would be unreadable."""

    def test_the_figure_and_the_widened_limits_are_returned(self):
        figure, limits = pr.plot_histograms_only(
            posterior_frame(), plot_keys=["log10_mej_dyn", "log10_mej_wind"]
        )
        assert isinstance(figure, plt.Figure)
        assert len(limits) == 2

    def test_one_panel_is_drawn_per_parameter(self):
        figure, _ = pr.plot_histograms_only(
            posterior_frame(), plot_keys=["log10_mej_dyn", "log10_mej_wind"]
        )
        assert len(figure.axes) >= 2

    def test_the_column_count_can_be_chosen(self):
        figure, _ = pr.plot_histograms_only(
            posterior_frame(),
            plot_keys=["log10_mej_dyn", "log10_mej_wind", "luminosity_distance"],
            ncols=1,
        )
        assert isinstance(figure, plt.Figure)


class TestPlotR14Trend(FigureMixin):
    """The figure the combine-EOS script ends with: the recovered radius as
    events accumulate, with and without the electromagnetic counterpart."""

    def setup_method(self):
        super().setup_method()
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.gw_dir = self.tmp_dir / "gw"
        self.gw_dir.mkdir()
        self.write_trend(self.tmp_dir / "GW_EM_R14trend_run.dat")
        self.write_trend(self.gw_dir / "GW_R14trend.dat", spread=0.6)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)
        super().teardown_method()

    def write_trend(self, path, events=5, spread=0.3):
        pd.DataFrame(
            {
                "R14_med": np.full(events, 11.6),
                "R14_uperr": np.linspace(spread, spread / 2, events),
                "R14_lowerr": np.linspace(spread, spread / 2, events),
            }
        ).to_csv(path, sep=" ", index=False)

    def args(self):
        return Namespace(
            outdir=str(self.tmp_dir),
            label="run",
            gwR14trend=str(self.gw_dir),
            R14_true=11.55,
        )

    def test_the_figure_is_written(self):
        pr.plot_R14_trend(self.args())
        assert (self.tmp_dir / "R14_trend_GW_EM_run.pdf").is_file()

    def test_both_trends_are_drawn(self):
        pr.plot_R14_trend(self.args())
        figure = plt.gcf()
        assert len(figure.axes) == 2

    def test_the_injected_value_is_marked(self):
        pr.plot_R14_trend(self.args())
        upper_axis = plt.gcf().axes[0]
        assert any(
            np.allclose(line.get_ydata(), 11.55) for line in upper_axis.get_lines()
        )

    def test_the_relative_error_panel_is_logarithmic(self):
        pr.plot_R14_trend(self.args())
        assert plt.gcf().axes[1].get_yscale() == "log"

    def test_a_missing_trend_file_is_reported(self):
        (self.tmp_dir / "GW_EM_R14trend_run.dat").unlink()
        with pytest.raises(FileNotFoundError):
            pr.plot_R14_trend(self.args())

    def test_the_file_names_match_what_the_trend_script_writes(self):
        # The script writes GW_EM_R14trend_<label>.dat under its output
        # directory, and this reader has to look for exactly that.
        source = Path(pr.__file__).read_text()
        assert "GW_EM_R14trend_{args.label}.dat" in source


class TestPlotMultiCorner(FigureMixin):
    """The entry point behind the corner-plot console script. Four attribute
    names do not match the parser it is fed by, so it cannot run."""

    def args(self, *extra):
        return corner_plot_parser(argparse.ArgumentParser()).parse_args(
            ["-f", "a.csv"] + list(extra)
        )

    def test_the_prior_file_is_read_under_the_wrong_name(self):
        # The parser defines --prior-filename, giving args.prior_filename,
        # but the routine reads args.prior.
        args = self.args()
        assert hasattr(args, "prior_filename")
        assert not hasattr(args, "prior")
        with pytest.raises(AttributeError) as caught:
            pr.plot_multi_corner(args)
        assert "prior" in str(caught.value)

    def test_the_best_fit_file_is_also_read_under_the_wrong_name(self):
        # The parser defines --bestfit-params; the routine reads
        # args.bestfit_json when deciding what to draw as truth.
        args = self.args()
        assert not hasattr(args, "bestfit_json")
        source = Path(pr.__file__).read_text()
        assert "args.bestfit_json" in source

    def test_the_verbose_flag_is_not_provided_by_the_parser(self):
        assert not hasattr(self.args(), "verbose")
        assert "args.verbose" in Path(pr.__file__).read_text()

    def test_the_plot_keys_and_labels_cannot_be_unpacked_from_one_mapping(self):
        # The routine writes "plot_keys, plot_labels = mapping.items()",
        # which only works for a mapping of exactly two parameters. Any
        # realistic prior has more and raises. Taking .keys() and .values()
        # separately is the fix.
        args = self.args()
        args.prior = "ignored"
        args.verbose = False
        with patch.object(
            pr.corepu,
            "plotting_parameters_from_priors",
            return_value={"a": "$a$", "b": "$b$", "c": "$c$"},
        ):
            with pytest.raises(ValueError) as caught:
                pr.plot_multi_corner(args)
        assert "unpack" in str(caught.value)

    def test_a_two_parameter_prior_slips_through_the_unpacking(self):
        # With exactly two parameters the unpacking succeeds but binds the
        # two key-value pairs, not the keys and the labels, so the plot keys
        # become tuples.
        args = self.args()
        args.prior = "ignored"
        args.verbose = False
        with patch.object(
            pr.corepu,
            "plotting_parameters_from_priors",
            return_value={"a": "$a$", "b": "$b$"},
        ):
            with patch.object(pr, "setup_corner_plot") as setup:
                pr.plot_multi_corner(args)
        assert setup.call_args.kwargs["plot_keys"] == ("a", "$a$")

    def test_the_legend_falls_back_to_the_file_names(self):
        args = self.args()
        assert args.label_name is None
        source = Path(pr.__file__).read_text()
        assert "for f in args.posterior_files" in source


class TestResamplingCornerPlot(FigureMixin):
    """The figure the resampling script ends with, drawn from the sampler
    solution rather than from a file."""

    def samples(self, withNSBH=False):
        generator = np.random.default_rng(9)
        frame = pd.DataFrame(
            {
                "chirp_mass": generator.normal(1.2, 0.01, 300),
                "mass_ratio": generator.uniform(0.8, 1.0, 300),
                "EOS": generator.uniform(0, 5, 300),
                "alpha": generator.uniform(0.0, 0.2, 300),
                "zeta": generator.uniform(0.0, 1.0, 300),
            }
        )
        if withNSBH:
            frame["chi_1"] = generator.uniform(-0.05, 0.05, 300)
            frame["chi_2"] = generator.uniform(-0.05, 0.05, 300)
        return frame

    def solution(self):
        """The sampler object, which the binary-neutron-star branch queries
        for the equation-of-state tables it loaded."""
        solution = MagicMock()
        masses = np.linspace(0.5, 2.3, 40)
        solution.EOS_masses_dict = {index: masses for index in range(1, 7)}
        solution.EOS_lambda_dict = {
            index: 1000.0 * np.exp(-2.0 * masses) for index in range(1, 7)
        }
        return solution

    def test_the_output_directory_is_handed_to_the_figure_argument(self):
        # corner_plot takes the figure to draw onto as its fourth positional
        # argument and the save path as a keyword. The directory is passed
        # positionally, so corner treats it as an existing figure and the
        # plot is never written. Passing save=<path> is the fix.
        directory = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(AttributeError) as caught:
                pr.resampling_corner_plot(
                    self.samples(withNSBH=True), self.solution(), str(directory), True
                )
            assert "axes" in str(caught.value)
            assert list(directory.iterdir()) == []
        finally:
            shutil.rmtree(directory)

    def test_the_same_mistake_affects_the_binary_neutron_star_branch(self):
        directory = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(AttributeError):
                pr.resampling_corner_plot(
                    self.samples(), self.solution(), str(directory), False
                )
        finally:
            shutil.rmtree(directory)

    def test_the_mixed_binary_branch_plots_four_parameters(self):
        with patch.object(pr, "corner_plot") as draw:
            pr.resampling_corner_plot(
                self.samples(withNSBH=True), self.solution(), "out", True
            )
        assert draw.call_args.args[0].shape[1] == 4

    def test_the_binary_neutron_star_branch_adds_tidal_and_maximum_mass(self):
        with patch.object(pr, "corner_plot") as draw:
            pr.resampling_corner_plot(self.samples(), self.solution(), "out", False)
        assert draw.call_args.args[0].shape[1] == 6
        assert r"$\tilde{\Lambda}$" in draw.call_args.args[1]

    def test_the_mass_ratio_is_reported_as_the_larger_than_one_convention(self):
        with patch.object(pr, "corner_plot") as draw:
            pr.resampling_corner_plot(
                self.samples(withNSBH=True), self.solution(), "out", True
            )
        mass_ratio = draw.call_args.args[0][:, 1]
        assert np.all(mass_ratio >= 1.0)
