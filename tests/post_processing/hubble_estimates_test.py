import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from nmma.post_processing import hubble_estimates


class ConstantDensity:
    """A stand-in for one event's Hubble-constant posterior. A constant log
    density makes the accumulation arithmetic checkable by hand."""

    def __init__(self, value):
        self.value = value

    def logpdf(self, samples):
        return np.full_like(np.asarray(samples, dtype=float), self.value)


def hubble_grid(size=8):
    return np.linspace(60.0, 80.0, size)


class TestH0Resampling:
    """Each accumulated weighting is turned back into samples by building a
    weighted density over the Hubble prior and drawing from it."""

    def setup_method(self):
        self.prior = np.random.default_rng(0).uniform(5, 120, 500)
        self.weights = np.ones(500) / 500

    def test_the_requested_number_of_samples_is_returned(self):
        samples = hubble_estimates.H0_resampling(self.prior, self.weights, 321)
        assert samples.shape == (321,)

    def test_a_one_dimensional_array_is_returned_not_a_row_matrix(self):
        samples = hubble_estimates.H0_resampling(self.prior, self.weights, 50)
        assert samples.ndim == 1

    def test_the_draws_follow_the_weighting(self):
        # Concentrating the weight on the low end has to pull the draws down.
        low_weights = np.where(self.prior < 40, 1.0, 1e-12)
        low_weights /= low_weights.sum()
        low = hubble_estimates.H0_resampling(self.prior, low_weights, 2000)
        flat = hubble_estimates.H0_resampling(self.prior, self.weights, 2000)
        assert np.median(low) < np.median(flat)

    def test_the_draws_stay_in_a_plausible_range(self):
        samples = hubble_estimates.H0_resampling(self.prior, self.weights, 2000)
        assert samples.min() > -50
        assert samples.max() < 250


class TestGenerateLogprob:
    """Accumulates one event's posterior at a time, so each row is the joint
    constraint after that many events. This is what the trend plots show."""

    def setup_method(self):
        self.samples = hubble_grid()
        self.probs = {
            0: ConstantDensity(-1.0),
            1: ConstantDensity(-2.0),
            2: ConstantDensity(-4.0),
        }

    def test_one_row_is_produced_per_event(self):
        rows = hubble_estimates.generate_logprob(self.probs, self.samples, [0, 1, 2])
        assert rows.shape == (3, len(self.samples))

    def test_every_row_is_the_same_array_so_the_trend_is_lost(self):
        # The accumulator is updated in place and the same object is appended
        # on every pass, so all rows end up holding the final value. The
        # trend over events is flattened to its endpoint, which defeats the
        # purpose of the whole routine. Appending a copy fixes it.
        rows = hubble_estimates.generate_logprob(self.probs, self.samples, [0, 1, 2])
        np.testing.assert_allclose(rows[0], rows[1])
        np.testing.assert_allclose(rows[0], rows[2])

    def test_a_single_event_is_still_correct(self):
        # With one event there is nothing to accumulate, so the aliasing
        # cannot bite and the row is the normalised density.
        rows = hubble_estimates.generate_logprob(self.probs, self.samples, [0])
        expected = np.full(len(self.samples), -1.0)
        expected = expected - scipy.special.logsumexp(expected)
        np.testing.assert_allclose(rows[0], expected)

    def test_each_row_is_normalised_to_sum_to_one(self):
        rows = hubble_estimates.generate_logprob(self.probs, self.samples, [0, 1])
        for row in rows:
            assert np.exp(row).sum() == pytest.approx(1.0, abs=1.5e-6)

    def test_a_volume_weighting_is_applied_from_the_second_event_onwards(self):
        # The first event keeps the uniform-in-volume prior; every later one
        # divides the prior back out, which shows up as a tilt in the Hubble
        # constant.
        single = hubble_estimates.generate_logprob(self.probs, self.samples, [0])
        pair = hubble_estimates.generate_logprob(self.probs, self.samples, [0, 1])
        assert not np.allclose(single[0], pair[-1])

    def test_no_events_gives_an_empty_result(self):
        rows = hubble_estimates.generate_logprob(self.probs, self.samples, [])
        assert rows.shape == (0,)

    def test_the_final_constraint_does_not_depend_on_the_event_order(self):
        # Accumulation is a sum in the logarithm and the volume weighting is
        # applied once per event after the first, so the endpoint is the same
        # whichever order the events arrive in. Only the intermediate steps
        # differ, which is what averaging over orderings is for.
        forward = hubble_estimates.generate_logprob(self.probs, self.samples, [0, 2])
        backward = hubble_estimates.generate_logprob(self.probs, self.samples, [2, 0])
        np.testing.assert_allclose(forward[-1], backward[-1])


class TestGetCumprodRowwise:
    def test_the_logarithms_are_exponentiated_row_by_row(self):
        logprob = np.log(np.array([[0.25, 0.75], [0.5, 0.5]]))
        np.testing.assert_allclose(
            hubble_estimates.get_cumprod_rowwise(logprob),
            [[0.25, 0.75], [0.5, 0.5]],
        )

    def test_the_shape_is_preserved(self):
        logprob = np.log(np.full((3, 4), 0.25))
        assert hubble_estimates.get_cumprod_rowwise(logprob).shape == (3, 4)

    def test_a_minus_infinite_logarithm_becomes_zero_weight(self):
        logprob = np.array([[-np.inf, 0.0]])
        np.testing.assert_allclose(
            hubble_estimates.get_cumprod_rowwise(logprob), [[0.0, 1.0]]
        )


class TestGenerateCumprods:
    """Produces three weightings per step: from gravitational waves alone,
    from the electromagnetic counterpart alone, and from both combined."""

    def setup_method(self):
        self.samples = hubble_grid()
        self.gw = {0: ConstantDensity(-1.0), 1: ConstantDensity(-2.0)}
        self.em = {0: ConstantDensity(-3.0), 1: ConstantDensity(-4.0)}

    def test_three_sets_of_weightings_are_returned(self):
        result = hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        )
        assert len(result) == 3

    def test_each_set_has_one_weighting_per_event(self):
        gw, em, total = hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        )
        for weighting in [gw, em, total]:
            assert weighting.shape == (2, len(self.samples))

    def test_the_combined_weighting_is_normalised(self):
        _, _, total = hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        )
        for row in total:
            assert row.sum() == pytest.approx(1.0, abs=1.5e-6)

    def test_the_single_messenger_weightings_are_normalised(self):
        gw, em, _ = hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        )
        for weighting in [gw, em]:
            for row in weighting:
                assert row.sum() == pytest.approx(1.0, abs=1.5e-6)

    def test_the_combined_weighting_differs_from_either_messenger(self):
        gw, em, total = hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        )
        assert not np.allclose(total[-1], gw[-1])
        assert not np.allclose(total[-1], em[-1])

    def test_every_weighting_is_non_negative(self):
        for weighting in hubble_estimates.generate_cumprods(
            self.gw, self.em, self.samples, [0, 1]
        ):
            assert np.all(weighting >= 0.0)


class TestH0MeansFromProbs:
    """Averages the trend over many random event orderings, so the reported
    curve does not depend on which event happened to come first."""

    def setup_method(self):
        self.samples = hubble_grid(size=40)
        self.gw = {0: ConstantDensity(-1.0), 1: ConstantDensity(-2.0)}
        self.em = {0: ConstantDensity(-3.0), 1: ConstantDensity(-4.0)}
        self.args = Namespace(
            N_reordering=3,
            N_posterior_samples=50,
            cred_interval=0.95,
            rng=np.random.default_rng(0),
        )

    def spread(self, *_args, **_kwargs):
        return np.array([70.0, 71.0]), np.array([75.0, 76.0]), np.array([65.0, 66.0])

    def test_three_sets_of_estimates_are_returned(self):
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ):
            result = hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        assert len(result) == 3

    def test_each_set_holds_a_median_and_two_bounds(self):
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ):
            gw, em, total = hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        for estimate in [gw, em, total]:
            assert len(estimate) == 3

    def test_the_median_over_the_orderings_is_reported(self):
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ):
            gw, _, _ = hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        np.testing.assert_allclose(gw[0], [70.0, 71.0])

    def test_one_spread_is_computed_per_messenger_per_ordering(self):
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ) as spread:
            hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        assert spread.call_count == 3 * 3

    def test_the_event_order_is_reshuffled_for_every_realisation(self):
        shuffler = MagicMock()
        self.args.rng = shuffler
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ):
            hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        assert shuffler.shuffle.call_count == 3

    def test_the_credible_interval_is_passed_through(self):
        self.args.cred_interval = 0.68
        with patch.object(
            hubble_estimates, "find_spread_from_resampling", side_effect=self.spread
        ) as spread:
            hubble_estimates.H0_means_from_probs(
                self.gw, self.em, self.samples, self.args, np.array([0, 1])
            )
        assert spread.call_args.args[4] == pytest.approx(0.68)


class LoadPosteriorsMixin:
    """Both trend scripts read one posterior file per event from a directory,
    and tolerate gaps so a partially finished run can be analysed."""

    n_events = 3
    n_samples = 200

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.gw_dir = self.tmp_dir / "gw"
        self.em_dir = self.tmp_dir / "em"
        for directory in [self.gw_dir, self.em_dir]:
            directory.mkdir()

        generator = np.random.default_rng(2)
        self.injection = pd.DataFrame(
            {"luminosity_distance": np.linspace(40.0, 80.0, self.n_events)}
        )
        for event in range(self.n_events):
            self.write_event(event, generator)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def write_event(self, event, generator, directories=None):
        distance = self.injection.luminosity_distance[event]
        for directory in directories or [self.gw_dir, self.em_dir]:
            frame = pd.DataFrame(
                {
                    "luminosity_distance": generator.normal(
                        distance, 2.0, self.n_samples
                    ),
                    "mass_1": generator.normal(1.6, 0.05, self.n_samples),
                    "mass_2": generator.normal(1.4, 0.05, self.n_samples),
                    "chirp_mass": generator.normal(1.2, 0.01, self.n_samples),
                    "mass_ratio": generator.uniform(0.7, 1.0, self.n_samples),
                }
            )
            frame.to_csv(
                directory / f"posterior_samples_{event}.dat", sep=" ", index=False
            )

    def args(self, **kwargs):
        defaults = dict(
            GWsamples=str(self.gw_dir),
            EMsamples=str(self.em_dir),
            inject_Hubble=70.0,
            p_value_threshold=None,
            rng=np.random.default_rng(4),
        )
        defaults.update(kwargs)
        return Namespace(**defaults)


class TestLoadInPosteriors(LoadPosteriorsMixin):
    def test_one_density_per_event_is_built_for_each_messenger(self):
        em, gw = hubble_estimates.load_in_posteriors(
            self.injection, range(self.n_events), self.args()
        )
        assert sorted(em) == [0, 1, 2]
        assert sorted(gw) == [0, 1, 2]

    def test_the_densities_are_over_the_hubble_constant(self):
        em, _ = hubble_estimates.load_in_posteriors(
            self.injection, range(self.n_events), self.args()
        )
        assert isinstance(em[0], scipy.stats.gaussian_kde)
        assert em[0].dataset.mean() > 10.0
        assert em[0].dataset.mean() < 200.0

    def test_a_missing_event_file_is_skipped_rather_than_fatal(self):
        # The scripts are meant to run against an incomplete set of results.
        (self.gw_dir / "posterior_samples_1.dat").unlink()
        em, gw = hubble_estimates.load_in_posteriors(
            self.injection, range(self.n_events), self.args()
        )
        assert sorted(gw) == [0, 2]
        assert sorted(em) == [0, 2]

    def test_the_electromagnetic_density_is_reweighted_by_distance_squared(self):
        # Reweighting back to uniform in volume is what makes the selection
        # effect on the Hubble constant a known power law.
        em, gw = hubble_estimates.load_in_posteriors(self.injection, [0], self.args())
        assert em[0].weights is not None
        assert not np.allclose(em[0].weights, em[0].weights[0])

    def test_the_gravitational_wave_density_is_left_unweighted(self):
        _, gw = hubble_estimates.load_in_posteriors(self.injection, [0], self.args())
        np.testing.assert_allclose(gw[0].weights, gw[0].weights[0])

    def test_a_well_recovered_event_survives_the_p_value_cut(self):
        em, gw = hubble_estimates.load_in_posteriors(
            self.injection, [0], self.args(p_value_threshold=1e-6)
        )
        assert sorted(gw) == [0]

    def test_a_badly_recovered_event_is_discarded_by_the_p_value_cut(self):
        # The injected distance sitting in the far tail of the recovered
        # posterior means the gravitational-wave run did not converge, so the
        # event is dropped rather than contributing a wrong constraint.
        with patch.object(
            hubble_estimates.scipy.stats, "percentileofscore", return_value=99.9
        ):
            em, gw = hubble_estimates.load_in_posteriors(
                self.injection,
                range(self.n_events),
                self.args(p_value_threshold=0.05),
            )
        assert len(gw) == 0
        assert len(em) == 0

    def test_an_event_right_at_the_threshold_is_kept(self):
        # The comparison is strictly less than, so a p-value equal to the
        # threshold survives.
        with patch.object(
            hubble_estimates.scipy.stats, "percentileofscore", return_value=2.5
        ):
            _, gw = hubble_estimates.load_in_posteriors(
                self.injection, [0], self.args(p_value_threshold=0.05)
            )
        assert sorted(gw) == [0]

    def test_the_two_messengers_are_read_from_their_own_directories(self):
        args = self.args(GWsamples=str(self.tmp_dir / "absent"))
        em, gw = hubble_estimates.load_in_posteriors(self.injection, [0], args)
        assert len(gw) == 0
        assert len(em) == 0


class TestMain(LoadPosteriorsMixin):
    """The console script wires the pieces together and writes one row per
    accumulated event."""

    def setup_method(self):
        super().setup_method()
        self.injection_file = self.tmp_dir / "injection.json"
        self.injection.to_json(self.injection_file)

    def parsed_args(self, **kwargs):
        defaults = dict(
            GWsamples=str(self.gw_dir),
            EMsamples=str(self.em_dir),
            injection=str(self.injection_file),
            inject_Hubble=70.0,
            p_value_threshold=None,
            detections_file=None,
            Nevent=self.n_events,
            seed=42,
            N_prior_samples=200,
            N_posterior_samples=50,
            N_reordering=2,
            cred_interval=0.95,
            outdir=str(self.tmp_dir),
            output_label="run",
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def run_main(self, **kwargs):
        args = self.parsed_args(**kwargs)
        spread = (
            np.array([70.0, 71.0]),
            np.array([75.0, 76.0]),
            np.array([65.0, 66.0]),
        )
        with patch.object(hubble_estimates, "nmma_base_parsing", return_value=args):
            with patch.object(
                hubble_estimates, "read_injection_file", return_value=self.injection
            ):
                with patch.object(
                    hubble_estimates,
                    "find_spread_from_resampling",
                    return_value=spread,
                ):
                    hubble_estimates.main()
        return args

    def test_the_trend_file_is_written(self):
        self.run_main()
        assert Path("GW_EM_H0_trend_run.dat").is_file()
        Path("GW_EM_H0_trend_run.dat").unlink()

    def test_the_trend_file_lands_in_the_working_directory_not_the_outdir(self):
        # Every other script writes under its output directory; this one
        # ignores the setting, so the result appears wherever it was started.
        self.run_main()
        assert not (self.tmp_dir / "GW_EM_H0_trend_run.dat").is_file()
        assert Path("GW_EM_H0_trend_run.dat").is_file()
        Path("GW_EM_H0_trend_run.dat").unlink()

    def test_the_trend_file_holds_a_column_per_messenger_and_bound(self):
        self.run_main()
        written = pd.read_csv("GW_EM_H0_trend_run.dat", sep=" ")
        for column in [
            "GW_med",
            "GW_uperr",
            "GW_lowerr",
            "EM_med",
            "EM_uperr",
            "EM_lowerr",
            "total_med",
            "total_uperr",
            "total_lowerr",
        ]:
            assert column in written.columns, column
        Path("GW_EM_H0_trend_run.dat").unlink()

    def test_the_errors_are_stored_as_offsets_from_the_median(self):
        self.run_main()
        written = pd.read_csv("GW_EM_H0_trend_run.dat", sep=" ")
        np.testing.assert_allclose(written["GW_uperr"], [5.0, 5.0])
        np.testing.assert_allclose(written["GW_lowerr"], [5.0, 5.0])
        Path("GW_EM_H0_trend_run.dat").unlink()

    def test_the_events_can_come_from_a_detections_file(self):
        detections = self.tmp_dir / "detections.dat"
        np.savetxt(detections, np.array([0, 2]))
        self.run_main(detections_file=str(detections))
        assert Path("GW_EM_H0_trend_run.dat").is_file()
        Path("GW_EM_H0_trend_run.dat").unlink()

    def test_neither_an_event_count_nor_a_detections_file_is_refused(self):
        with pytest.raises(ValueError):
            self.run_main(detections_file=None, Nevent=None)


class TestScipySpecialImport:
    def test_the_special_submodule_is_reachable_without_being_imported(self):
        # The module normalises with scipy.special.logsumexp but only imports
        # scipy.stats. It works because scipy.stats pulls scipy.special in as
        # a side effect, which is worth importing explicitly rather than
        # relying on.
        source = Path(hubble_estimates.__file__).read_text()
        assert "scipy.special.logsumexp" in source
        assert "import scipy.special" not in source
        assert hasattr(scipy, "special")
