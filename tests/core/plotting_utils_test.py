import itertools

import matplotlib
import numpy as np
import pytest
from bilby.core.prior import DeltaFunction, PriorDict, Uniform
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402

from nmma.core import plotting_utils  # noqa: E402


class RcParamsMixin:
    """Importing the module switches LaTeX rendering on outside CI, which
    needs a LaTeX installation that a developer machine may not have. Each
    test therefore renders with mathtext and puts the global rcParams back
    afterwards, so no test leaks its settings into the rest of the suite."""

    def setup_method(self):
        self.original_rc = matplotlib.rcParams.copy()
        matplotlib.rcParams["text.usetex"] = False

    def teardown_method(self):
        plt.close("all")
        matplotlib.rcParams.update(self.original_rc)


class TestFigSetup(RcParamsMixin):
    def test_a_colour_cycle_is_returned(self):
        colors = plotting_utils.fig_setup()
        assert isinstance(colors, itertools.cycle)
        first = next(colors)
        assert first.startswith("#")

    def test_the_colour_cycle_repeats(self):
        colors = plotting_utils.fig_setup()
        drawn = [next(colors) for _ in range(11)]
        assert len(set(drawn)) == 11
        assert next(colors) == drawn[0]

    def test_the_figure_style_is_applied(self):
        plotting_utils.fig_setup()
        assert matplotlib.rcParams["axes.labelsize"] == 18
        assert matplotlib.rcParams["font.family"] == ["serif"]

    def test_the_figure_size_follows_the_golden_ratio(self):
        plotting_utils.fig_setup()
        width, height = matplotlib.rcParams["figure.figsize"]
        golden_mean = (np.sqrt(5) - 1.0) / 2.0
        assert height / width == pytest.approx(0.9 * golden_mean)


class TestPlottingParametersFromPriors(RcParamsMixin):
    def make_priors(self):
        priors = PriorDict()
        priors["log10_mej"] = Uniform(-3, -1, "log10_mej", latex_label=r"$\log M$")
        priors["log10_vej"] = Uniform(-2, -0.5, "log10_vej", latex_label=r"$\log v$")
        return priors

    def test_every_sampled_parameter_gets_its_latex_label(self):
        labels = plotting_utils.plotting_parameters_from_priors(self.make_priors())
        assert labels == {"log10_mej": r"$\log M$", "log10_vej": r"$\log v$"}

    def test_fixed_parameters_are_left_out(self):
        priors = self.make_priors()
        priors["beta"] = DeltaFunction(3.0, name="beta")
        labels = plotting_utils.plotting_parameters_from_priors(priors)
        assert "beta" not in labels

    def test_a_float_entry_is_treated_as_fixed(self):
        priors = self.make_priors()
        priors["beta"] = 3.0
        labels = plotting_utils.plotting_parameters_from_priors(priors)
        assert "beta" not in labels

    def test_the_keys_can_be_restricted(self):
        labels = plotting_utils.plotting_parameters_from_priors(
            self.make_priors(), keys=["log10_mej"]
        )
        assert list(labels) == ["log10_mej"]

    def test_a_prior_file_path_is_read(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            prior_file = Path(tmp_dir) / "test.prior"
            prior_file.write_text(
                "log10_mej = Uniform(minimum=-3, maximum=-1, name='log10_mej')\n"
            )
            labels = plotting_utils.plotting_parameters_from_priors(str(prior_file))
        assert "log10_mej" in labels


class TestSetupMultiAxes(RcParamsMixin):
    def test_enough_axes_are_created(self):
        fig, axes = plotting_utils.setup_multi_axes(6)
        assert len(axes) >= 6

    def test_the_grid_is_roughly_square_by_default(self):
        _, axes = plotting_utils.setup_multi_axes(9)
        assert len(axes) == 9

    def test_the_column_count_can_be_fixed(self):
        _, axes = plotting_utils.setup_multi_axes(6, ncols=2)
        assert len(axes) == 6

    def test_the_grid_never_grows_beyond_five_columns(self):
        fig, axes = plotting_utils.setup_multi_axes(36)
        assert len(axes) == 40  # 5 columns, 8 rows

    def test_a_single_panel_still_yields_something_indexable(self):
        _, axes = plotting_utils.setup_multi_axes(1, ncols=1)
        assert axes is not None

    def test_extra_figure_arguments_are_forwarded(self):
        fig, _ = plotting_utils.setup_multi_axes(4, figsize=(4.0, 3.0))
        np.testing.assert_allclose(fig.get_size_inches(), [4.0, 3.0])

    def test_shared_axes_are_honoured(self):
        _, axes = plotting_utils.setup_multi_axes(4, sharex=True, ncols=2)
        assert axes[0].get_shared_x_axes().joined(axes[0], axes[2])


class TestFadingCmap(RcParamsMixin):
    def test_a_colormap_is_returned(self):
        cmap = plotting_utils.fading_cmap("#22ADFC")
        assert isinstance(cmap, LinearSegmentedColormap)

    def test_the_map_fades_in_from_fully_transparent(self):
        cmap = plotting_utils.fading_cmap("#22ADFC")
        assert cmap(0.0)[3] == pytest.approx(0.0)
        assert cmap(1.0)[3] == pytest.approx(1.0)

    def test_the_opaque_end_is_the_requested_colour(self):
        cmap = plotting_utils.fading_cmap("#22ADFC")
        np.testing.assert_allclose(
            cmap(1.0)[:3], (0x22 / 255, 0xAD / 255, 0xFC / 255), atol=1e-6
        )

    def test_a_named_colour_is_accepted(self):
        cmap = plotting_utils.fading_cmap("red")
        np.testing.assert_allclose(cmap(1.0)[:3], (1.0, 0.0, 0.0), atol=1e-6)

    def test_the_alpha_channel_increases_monotonically(self):
        cmap = plotting_utils.fading_cmap("#22ADFC")
        alphas = [cmap(x)[3] for x in np.linspace(0, 1, 20)]
        assert np.all(np.diff(alphas) >= 0)


class TestGetOffset:
    def test_a_pure_addition_has_no_multiplier(self):
        assert plotting_utils.get_offset("+1000") == (1.0, 1000.0)

    def test_a_negative_addition(self):
        assert plotting_utils.get_offset("-0.5") == (1.0, -0.5)

    def test_a_unicode_minus_is_normalised(self):
        # matplotlib writes its offsets with a typographic minus sign
        assert plotting_utils.get_offset("−1000") == (1.0, -1000.0)

    def test_a_pure_multiplier_has_no_addend(self):
        assert plotting_utils.get_offset("1e6") == (1e6, 0.0)

    def test_a_multiplier_and_an_addend_are_separated(self):
        assert plotting_utils.get_offset("1e6+500") == (1e6, 500.0)

    def test_a_negative_exponent_is_not_mistaken_for_the_addend(self):
        multiplier, addend = plotting_utils.get_offset("1e-3-20")
        assert multiplier == pytest.approx(1e-3)
        assert addend == pytest.approx(-20.0)

    def test_a_negative_exponent_without_an_addend(self):
        assert plotting_utils.get_offset("1e-3") == (1e-3, 0.0)


class TestFormatTitleOffset(RcParamsMixin):
    def make_axis_with_offset(self):
        fig, ax = plt.subplots()
        ax.plot([1e6 + 1, 1e6 + 2, 1e6 + 3], [1, 2, 3])
        fig.canvas.draw()
        return ax

    def test_a_title_is_returned_unchanged_when_there_is_no_offset(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        fig.canvas.draw()
        title = r"$x = ${1.0}_{-0.1}^{+0.1}$"
        assert plotting_utils.format_title_offset(ax, title) == title

    def test_the_offset_is_folded_into_the_quoted_value(self):
        ax = self.make_axis_with_offset()
        formatter = ax.xaxis.get_major_formatter()
        if not formatter.get_offset():
            pytest.skip("this matplotlib backend did not produce an axis offset")
        title = "$x = ${1000002.0}_{-0.5}^{+0.5}$"
        formatted = plotting_utils.format_title_offset(ax, title)
        assert formatted != title
        assert "^{+" in formatted


class TestArangeTitles(RcParamsMixin):
    def make_axis(self, n_texts):
        fig, ax = plt.subplots()
        ax.set_title("title")
        for i in range(n_texts):
            ax.text(0.5, 1.0, f"text {i}")
        return ax

    def test_the_axis_title_is_hidden(self):
        ax = plotting_utils.arange_titles(self.make_axis(2))
        assert not ax.title.get_visible()

    def test_two_texts_are_pushed_to_the_left_and_right_edges(self):
        ax = plotting_utils.arange_titles(self.make_axis(2))
        assert ax.texts[0].get_ha() == "left"
        assert ax.texts[1].get_ha() == "right"
        assert ax.texts[0].get_position()[0] == pytest.approx(0.0)
        assert ax.texts[1].get_position()[0] == pytest.approx(1.0)

    def test_two_texts_can_be_moved_up(self):
        ax = plotting_utils.arange_titles(self.make_axis(2), move=True)
        assert ax.texts[0].get_position()[1] > 1.07

    def test_three_texts_get_a_centred_middle_entry(self):
        ax = plotting_utils.arange_titles(self.make_axis(3))
        assert ax.texts[1].get_ha() == "center"
        assert ax.texts[2].get_ha() == "right"

    def test_four_texts_are_arranged_over_two_rows(self):
        ax = plotting_utils.arange_titles(self.make_axis(4))
        assert ax.texts[1].get_ha() == "left"
        assert ax.texts[2].get_ha() == "right"
        assert ax.texts[3].get_ha() == "right"
        assert ax.texts[0].get_position()[1] > 1.07

    def test_an_unexpected_number_of_texts_hides_all_of_them(self):
        ax = plotting_utils.arange_titles(self.make_axis(5))
        for text in ax.texts:
            assert not text.get_visible()


class TestTextHelpers(RcParamsMixin):
    def test_move_up_shifts_only_the_vertical_position(self):
        fig, ax = plt.subplots()
        text = ax.text(0.3, 1.0, "label")
        plotting_utils.move_up(text)
        assert text.get_position()[0] == pytest.approx(0.3)
        assert text.get_position()[1] == pytest.approx(1.22)

    def test_move_up_takes_an_explicit_amount(self):
        fig, ax = plt.subplots()
        text = ax.text(0.3, 1.0, "label")
        plotting_utils.move_up(text, amount=0.5)
        assert text.get_position()[1] == pytest.approx(1.5)

    def test_well_separated_texts_do_not_overlap(self):
        fig, ax = plt.subplots()
        left = ax.text(0.0, 0.5, "left")
        right = ax.text(0.9, 0.5, "right")
        fig.canvas.draw()
        assert not plotting_utils.texts_overlap(left, right)

    def test_texts_at_the_same_place_overlap(self):
        fig, ax = plt.subplots()
        left = ax.text(0.4, 0.5, "a long piece of text")
        right = ax.text(0.4, 0.5, "another long piece of text")
        fig.canvas.draw()
        assert plotting_utils.texts_overlap(left, right)

