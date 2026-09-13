import shutil
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nmma.post_processing import ns_characteristics


class TestGenerateEOSCumprods:
    """Accumulates one event's equation-of-state posterior at a time, each
    time dividing the common prior back out so it is only counted once."""

    def setup_method(self):
        self.prior = np.array([0.25, 0.25, 0.25, 0.25])

    def test_one_weighting_is_produced_per_event(self):
        probs = [np.array([0.4, 0.3, 0.2, 0.1])] * 3
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        assert len(result) == 3

    def test_each_weighting_is_normalised(self):
        probs = [np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.1, 0.2, 0.3, 0.4])]
        for weighting in ns_characteristics.generate_EOS_cumprods(probs, self.prior):
            assert weighting.sum() == pytest.approx(1.0)

    def test_the_first_weighting_is_the_first_posterior(self):
        # With the prior divided out, one event leaves its own posterior.
        probs = [np.array([0.4, 0.3, 0.2, 0.1])]
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        np.testing.assert_allclose(result[0], [0.4, 0.3, 0.2, 0.1])

    def test_the_weightings_accumulate_across_events(self):
        probs = [np.array([0.4, 0.3, 0.2, 0.1])] * 2
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        expected = np.array([0.4, 0.3, 0.2, 0.1]) ** 2 / 0.25
        expected /= expected.sum()
        np.testing.assert_allclose(result[-1], expected)

    def test_each_step_is_kept_separately_so_the_trend_survives(self):
        # Unlike the Hubble accumulation, this routine rebinds rather than
        # mutating in place, so the intermediate steps are preserved.
        probs = [np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.1, 0.7, 0.1, 0.1])]
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        assert not np.allclose(result[0], result[1])

    def test_agreeing_events_sharpen_the_weighting(self):
        probs = [np.array([0.7, 0.1, 0.1, 0.1])] * 3
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        assert result[-1][0] > result[0][0]

    def test_an_equation_of_state_excluded_by_one_event_stays_excluded(self):
        probs = [np.array([0.0, 0.4, 0.3, 0.3]), np.array([0.7, 0.1, 0.1, 0.1])]
        result = ns_characteristics.generate_EOS_cumprods(probs, self.prior)
        assert result[-1][0] == pytest.approx(0.0)

    def test_a_non_uniform_prior_is_respected(self):
        prior = np.array([0.7, 0.1, 0.1, 0.1])
        probs = [np.array([0.25, 0.25, 0.25, 0.25])]
        result = ns_characteristics.generate_EOS_cumprods(probs, prior)
        np.testing.assert_allclose(result[0], [0.25, 0.25, 0.25, 0.25])

    def test_no_events_gives_no_weightings(self):
        assert ns_characteristics.generate_EOS_cumprods([], self.prior) == []


class LoadPosteriorsMixin:
    n_eos = 4
    n_samples = 400

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.samples_dir = self.tmp_dir / "samples"
        self.samples_dir.mkdir()
        generator = np.random.default_rng(6)
        for event in range(3):
            self.write_event(event, generator)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def write_event(self, event, generator, eos_values=None):
        directory = self.samples_dir / str(event)
        directory.mkdir(parents=True, exist_ok=True)
        if eos_values is None:
            eos_values = generator.integers(0, self.n_eos, self.n_samples)
        frame = pd.DataFrame(
            {
                "EOS": np.asarray(eos_values, dtype=float),
                "mass_1": generator.normal(1.6, 0.05, len(eos_values)),
                "mass_2": generator.normal(1.4, 0.05, len(eos_values)),
                "chirp_mass": generator.normal(1.2, 0.01, len(eos_values)),
                "mass_ratio": generator.uniform(0.7, 1.0, len(eos_values)),
            }
        )
        frame.to_csv(directory / "posterior_samples.dat", sep=" ", index=False)

    def args(self, **kwargs):
        defaults = dict(GWEMsamples=str(self.samples_dir), Neos=self.n_eos)
        defaults.update(kwargs)
        return Namespace(**defaults)


class TestLoadInPosteriors(LoadPosteriorsMixin):
    """Each event's posterior over equation-of-state index is turned into a
    discrete probability by counting the samples in each bin."""

    def test_one_probability_vector_is_returned_per_event(self):
        probs = ns_characteristics.load_in_posteriors([0, 1, 2], self.args())
        assert len(probs) == 3

    def test_each_vector_has_one_entry_per_equation_of_state(self):
        probs = ns_characteristics.load_in_posteriors([0], self.args())
        assert len(probs[0]) == self.n_eos

    def test_each_vector_is_normalised(self):
        for prob in ns_characteristics.load_in_posteriors([0, 1], self.args()):
            assert prob.sum() == pytest.approx(1.0)

    def unweighted(self):
        """Reweighting to a flat mass prior resamples the table, so it is
        bypassed when the counting itself is under test."""
        return patch.object(
            ns_characteristics,
            "reweight_to_flat_mass_prior",
            side_effect=lambda frame: frame,
        )

    def test_the_counting_reflects_where_the_samples_fell(self):
        generator = np.random.default_rng(0)
        self.write_event(9, generator, eos_values=[0] * 30 + [1] * 10)
        with self.unweighted():
            probs = ns_characteristics.load_in_posteriors([9], self.args())
        np.testing.assert_allclose(probs[0], [0.75, 0.25, 0.0, 0.0])

    def test_the_continuous_index_is_floored_then_shifted_to_match_the_tables(self):
        # The sampler treats the index as continuous; counting happens on the
        # one-based integer the table files are named with.
        generator = np.random.default_rng(0)
        self.write_event(8, generator, eos_values=[0.2, 0.9, 1.4, 1.8])
        with self.unweighted():
            probs = ns_characteristics.load_in_posteriors([8], self.args())
        np.testing.assert_allclose(probs[0], [0.5, 0.5, 0.0, 0.0])

    def test_a_missing_event_directory_is_skipped(self):
        # The routine is meant to run against a partially finished set.
        probs = ns_characteristics.load_in_posteriors([0, 42, 2], self.args())
        assert len(probs) == 2

    def test_the_samples_are_reweighted_to_a_flat_mass_prior_first(self):
        with patch.object(
            ns_characteristics,
            "reweight_to_flat_mass_prior",
            side_effect=lambda frame: frame,
        ) as reweight:
            ns_characteristics.load_in_posteriors([0], self.args())
        reweight.assert_called_once()

    def test_asking_for_more_equations_of_state_pads_with_zeros(self):
        probs = ns_characteristics.load_in_posteriors([0], self.args(Neos=10))
        assert len(probs[0]) == 10
        assert probs[0][-1] == pytest.approx(0.0)


class TestEstimateObservableTrend:
    """Averages the radius trend over many random event orderings."""

    def setup_method(self):
        self.prior = np.linspace(10.0, 14.0, 50)
        self.probs = [np.full(4, 0.25), np.array([0.4, 0.3, 0.2, 0.1])]
        self.prior_prob = np.full(4, 0.25)
        self.args = Namespace(
            seed=42, N_reordering=3, N_posterior_samples=50, cred_interval=0.95
        )

    def spread(self, *_args, **_kwargs):
        return np.array([11.5, 11.6]), np.array([12.0, 12.1]), np.array([11.0, 11.1])

    def run_trend(self):
        with patch.object(
            ns_characteristics, "find_spread_from_resampling", side_effect=self.spread
        ) as spread:
            result = ns_characteristics.estimate_observable_trend(
                self.prior, self.probs, self.prior_prob, self.args
            )
        return result, spread

    def test_a_median_and_two_bounds_are_returned(self):
        result, _ = self.run_trend()
        assert len(result) == 3

    def test_each_estimate_has_one_entry_per_event(self):
        result, _ = self.run_trend()
        for estimate in result:
            assert len(estimate) == 2

    def test_the_median_over_the_orderings_is_reported(self):
        result, _ = self.run_trend()
        np.testing.assert_allclose(result[0], [11.5, 11.6])

    def test_one_spread_is_computed_per_ordering(self):
        _, spread = self.run_trend()
        assert spread.call_count == 3

    def test_the_estimates_are_ordered_median_upper_lower(self):
        result, _ = self.run_trend()
        median, upper, lower = result
        assert np.all(upper > median)
        assert np.all(lower < median)

    def test_the_resampling_draws_from_the_radius_prior(self):
        _, spread = self.run_trend()
        method = spread.call_args.args[0]
        drawn = method(self.prior, np.full(len(self.prior), 1 / len(self.prior)), 20)
        assert len(drawn) == 20
        assert np.all(np.isin(drawn, self.prior))

    def test_the_seed_makes_the_resampling_reproducible(self):
        first, _ = self.run_trend()
        second, _ = self.run_trend()
        np.testing.assert_allclose(first[0], second[0])

    def test_the_event_list_is_reshuffled_in_place_for_each_ordering(self):
        # The shuffle uses the global random module rather than the seeded
        # generator, so the orderings are not reproducible from args.seed.
        with patch.object(ns_characteristics.random, "shuffle") as shuffle:
            with patch.object(
                ns_characteristics,
                "find_spread_from_resampling",
                side_effect=self.spread,
            ):
                ns_characteristics.estimate_observable_trend(
                    self.prior, self.probs, self.prior_prob, self.args
                )
        assert shuffle.call_count == 3


class TestMain:
    """The combine-EOS console script. Two naming mistakes stop it before it
    reaches any of the science."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.detections = self.tmp_dir / "detections.dat"
        np.savetxt(self.detections, np.array([[0], [1]]))
        self.eos_prior = self.tmp_dir / "eos.prior"
        np.savetxt(self.eos_prior, np.full(4, 0.25))
        self.pdet = self.tmp_dir / "pdet.dat"
        np.savetxt(
            self.pdet,
            np.column_stack([np.linspace(1.9, 2.5, 10), np.linspace(0.2, 0.9, 10)]),
        )

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def args(self, **kwargs):
        defaults = dict(
            detections_file=str(self.detections),
            EOS_prior=str(self.eos_prior),
            EOSpath=str(self.tmp_dir / "eos"),
            pdet=str(self.pdet),
            Neos=4,
            outdir=str(self.tmp_dir),
            label="run",
            seed=42,
            N_reordering=2,
            N_posterior_samples=20,
            cred_interval=0.95,
            GWEMsamples=str(self.tmp_dir / "samples"),
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_the_equation_of_state_directory_is_read_under_the_wrong_name(self):
        # The parser defines --EOSpath, giving args.EOSpath, but the script
        # reads args.EOSPath with a capital P. The run dies here, before any
        # posterior is loaded. One of the two spellings has to change.
        args = self.args()
        with patch.object(ns_characteristics, "nmma_base_parsing", return_value=args):
            with pytest.raises(AttributeError) as caught:
                ns_characteristics.main()
        assert "EOSPath" in str(caught.value)

    def test_the_trend_file_cannot_be_written_with_a_regular_expression_separator(self):
        # Past the attribute error, the output is written with a separator of
        # "\s+". That is a valid separator for reading, where it is treated
        # as a regular expression, but writing needs a single character.
        args = self.args()
        args.EOSPath = args.EOSpath
        trend = (np.array([11.5, 11.6]), np.array([12.0, 12.1]), np.array([11.0, 11.1]))
        with patch.object(ns_characteristics, "nmma_base_parsing", return_value=args):
            with patch.object(
                ns_characteristics,
                "load_macro_characteristics_from_tabulated_eos_set",
                return_value=(np.linspace(2.0, 2.4, 4), np.linspace(11.0, 12.0, 4)),
            ):
                with patch.object(
                    ns_characteristics, "load_in_posteriors", return_value=[]
                ):
                    with patch.object(
                        ns_characteristics,
                        "estimate_observable_trend",
                        return_value=trend,
                    ):
                        with patch.object(ns_characteristics, "plot_R14_trend"):
                            with pytest.raises(TypeError):
                                ns_characteristics.main()

    def test_the_source_still_carries_the_invalid_escape_sequences(self):
        # Both separators are written as "\s+" in a plain string, which
        # python reports as an invalid escape sequence at import time. Raw
        # strings would silence that.
        source = Path(ns_characteristics.__file__).read_text()
        assert 'sep="\\s+"' in source
