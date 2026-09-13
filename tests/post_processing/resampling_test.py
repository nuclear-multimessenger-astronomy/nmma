import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.stats
from bilby.core.prior import Uniform
from bilby.gw.prior import PriorDict

from nmma.post_processing import resampling


def write_eos_set(directory, count=3, maximum_mass=2.2):
    """A directory of tabulated macroscopic equations of state, named the way
    the resampler indexes them, in radius-mass-lambda column order."""
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        masses = np.linspace(0.5, maximum_mass + 0.05 * index, 60)
        radii = 12.0 + 0.1 * index - 0.3 * masses
        lambdas = 1000.0 * np.exp(-2.0 * masses)
        np.savetxt(
            directory / f"{index}.dat", np.column_stack([radii, masses, lambdas])
        )


def gw_samples(size=200, seed=3, with_spins=False):
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "chirp_mass": generator.normal(1.2, 0.01, size),
            "mass_ratio": generator.uniform(0.7, 1.0, size),
            "luminosity_distance": generator.uniform(30.0, 60.0, size),
            "EOS": generator.integers(0, 3, size).astype(float),
        }
    )
    if with_spins:
        frame["chi_1"] = generator.uniform(-0.05, 0.05, size)
        frame["chi_2"] = generator.uniform(-0.05, 0.05, size)
    return frame


def em_samples(size=200, seed=5, combined=False):
    generator = np.random.default_rng(seed)
    if combined:
        return pd.DataFrame({"log10_mej": generator.normal(-2.0, 0.2, size)})
    return pd.DataFrame(
        {
            "log10_mej_dyn": generator.normal(-2.5, 0.2, size),
            "log10_mej_wind": generator.normal(-1.8, 0.2, size),
        }
    )


class TestFindSpreadFromResampling:
    """Turns a sequence of accumulated weightings into a median and credible
    interval for each step, which is what the trend plots show."""

    def setup_method(self):
        self.prior = np.linspace(10.0, 14.0, 500)
        self.weights = [np.ones(500) / 500, np.ones(500) / 500]

    def method(self, prior, weight, size):
        self.seen = (prior, weight, size)
        return np.random.default_rng(1).normal(12.0, 0.5, size)

    def test_the_credible_interval_keyword_no_longer_matches_arviz(self):
        # The interval is requested as hdi_prob, which arviz renamed to prob
        # in its 1.0 release, and the project requires arviz 1.2 or newer.
        # Every caller of this function therefore raises, which takes out
        # both the gwem-Hubble-estimate and combine-EOS console scripts.
        # Renaming the keyword to prob is the whole fix.
        with pytest.raises(TypeError) as caught:
            resampling.find_spread_from_resampling(
                self.method, self.weights, self.prior, 200, 0.95
            )
        assert "hdi_prob" in str(caught.value)

    def test_the_resampling_method_is_called_twice_per_weighting(self):
        # The call is written out twice with a comment between, and the first
        # result is thrown away, so every trend costs double.
        calls = []

        def counting_method(prior, weight, size):
            calls.append(weight)
            return np.random.default_rng(1).normal(12.0, 0.5, size)

        with patch.object(resampling, "hdi", return_value=(11.0, 13.0)):
            resampling.find_spread_from_resampling(
                counting_method, self.weights[:1], self.prior, 200, 0.95
            )
        assert len(calls) == 2

    def test_one_estimate_is_produced_per_weighting(self):
        with patch.object(resampling, "hdi", return_value=(11.0, 13.0)):
            median, upper, lower = resampling.find_spread_from_resampling(
                self.method, self.weights, self.prior, 200, 0.95
            )
        for estimate in [median, upper, lower]:
            assert estimate.shape == (2,)

    def test_the_median_of_the_resampled_draws_is_reported(self):
        with patch.object(resampling, "hdi", return_value=(11.0, 13.0)):
            median, _, _ = resampling.find_spread_from_resampling(
                self.method, self.weights[:1], self.prior, 200, 0.95
            )
        expected = np.median(np.random.default_rng(1).normal(12.0, 0.5, 200))
        assert median[0] == pytest.approx(expected)

    def test_the_interval_bounds_are_returned_lower_then_upper(self):
        with patch.object(resampling, "hdi", return_value=(11.0, 13.0)):
            _, upper, lower = resampling.find_spread_from_resampling(
                self.method, self.weights[:1], self.prior, 200, 0.95
            )
        assert upper[0] == pytest.approx(13.0)
        assert lower[0] == pytest.approx(11.0)

    def test_the_prior_and_sample_size_are_passed_through(self):
        with patch.object(resampling, "hdi", return_value=(11.0, 13.0)):
            resampling.find_spread_from_resampling(
                self.method, self.weights[:1], self.prior, 321, 0.95
            )
        prior, _, size = self.seen
        assert prior is self.prior
        assert size == 321

    def test_no_weightings_gives_empty_estimates(self):
        median, upper, lower = resampling.find_spread_from_resampling(
            self.method, [], self.prior, 200, 0.95
        )
        for estimate in [median, upper, lower]:
            assert len(estimate) == 0


class TestConstructEMKDE:
    """The electromagnetic posterior enters the resampling as a density over
    ejecta mass, built from whichever ejecta columns are present."""

    def test_a_single_total_ejecta_column_is_used_when_present(self):
        kde = resampling.construct_EM_KDE(em_samples(combined=True), False)
        assert isinstance(kde, scipy.stats.gaussian_kde)
        assert kde.d == 1

    def test_the_total_column_is_converted_out_of_the_logarithm(self):
        samples = pd.DataFrame({"log10_mej": np.full(200, -2.0)})
        kde = resampling.construct_EM_KDE(samples, False)
        np.testing.assert_allclose(kde.dataset[0], 1e-2)

    def test_the_two_ejecta_give_a_two_dimensional_density(self):
        kde = resampling.construct_EM_KDE(em_samples(), False)
        assert kde.d == 2

    def test_the_two_ejecta_can_be_summed_into_one_dimension(self):
        kde = resampling.construct_EM_KDE(em_samples(), True)
        assert kde.d == 1

    def test_summing_adds_the_two_masses_not_their_logarithms(self):
        samples = pd.DataFrame(
            {"log10_mej_dyn": np.full(200, -2.0), "log10_mej_wind": np.full(200, -2.0)}
        )
        kde = resampling.construct_EM_KDE(samples, True)
        np.testing.assert_allclose(kde.dataset[0], 2e-2)

    def test_the_single_total_column_wins_over_the_split_columns(self):
        samples = em_samples()
        samples["log10_mej"] = -2.0
        assert resampling.construct_EM_KDE(samples, False).d == 1

    def test_samples_without_any_ejecta_column_are_refused(self):
        with pytest.raises(ValueError):
            resampling.construct_EM_KDE(pd.DataFrame({"mass_1": [1.4]}), False)

    def test_only_one_of_the_two_split_columns_is_not_enough(self):
        with pytest.raises(ValueError):
            resampling.construct_EM_KDE(
                pd.DataFrame({"log10_mej_dyn": np.full(10, -2.0)}), False
            )


class StandaloneResampler(resampling.EjectaResamplerMixIn):
    """The mixin is designed to be paired with a sampler class, and the
    pipeline pairs it with pymultinest's solver. Pairing it with a plain
    object instead exercises the prior and the likelihood without needing
    the MultiNest library installed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ResamplerMixin:
    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.eos_dir = self.tmp_dir / "eos"
        write_eos_set(self.eos_dir, count=3)

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def priors(self, with_spins=False):
        gw = PriorDict(
            {
                "chirp_mass": Uniform(1.0, 1.5, "chirp_mass"),
                "mass_ratio": Uniform(0.5, 1.0, "mass_ratio"),
            }
        )
        if with_spins:
            gw["chi_1"] = Uniform(-0.1, 0.1, "chi_1")
            gw["chi_2"] = Uniform(-0.1, 0.1, "chi_2")
        em = PriorDict(
            {"alpha": Uniform(0.0, 0.2, "alpha"), "zeta": Uniform(0.0, 1.0, "zeta")}
        )
        return gw, em

    def build(self, withNSBH=False, combine_ejecta_mass=False):
        gw_prior, em_prior = self.priors(with_spins=withNSBH)
        return StandaloneResampler(
            gw_samples(with_spins=withNSBH),
            em_samples(combined=combine_ejecta_mass),
            gw_prior,
            em_prior,
            3,
            str(self.eos_dir),
            withNSBH,
            combine_ejecta_mass,
        )


class TestEjectaResamplerSetup(ResamplerMixin):
    """Setting up the resampler turns both posteriors into densities and
    loads the tabulated equations of state into memory."""

    def test_the_sampled_parameters_are_the_five_binary_neutron_star_ones(self):
        resampler = self.build()
        assert list(resampler._search_parameter_keys) == [
            "chirp_mass",
            "mass_ratio",
            "EOS",
            "alpha",
            "zeta",
        ]

    def test_a_neutron_star_black_hole_run_adds_the_two_spins(self):
        resampler = self.build(withNSBH=True)
        assert "chi_1" in resampler._search_parameter_keys
        assert "chi_2" in resampler._search_parameter_keys

    def test_the_equation_of_state_index_spans_one_past_the_count(self):
        # The index is sampled as a continuous value and floored later, so
        # the upper edge is one above the number of tables.
        resampler = self.build()
        assert resampler.priors["EOS"].maximum == 4

    def test_every_equation_of_state_table_is_loaded(self):
        resampler = self.build()
        assert sorted(resampler.EOS_masses_dict) == [1, 2, 3]
        assert sorted(resampler.EOS_radius_dict) == [1, 2, 3]
        assert sorted(resampler.EOS_lambda_dict) == [1, 2, 3]

    def test_the_tables_are_indexed_from_one_to_match_the_file_names(self):
        resampler = self.build()
        assert 0 not in resampler.EOS_masses_dict

    def test_the_posterior_equation_of_state_index_is_shifted_to_match(self):
        # The samples hold a zero-based continuous index, the tables are
        # named from one.
        resampler = self.build()
        assert resampler.EOSsamples.min() == 1
        assert np.issubdtype(resampler.EOSsamples.dtype, np.integer)

    def test_the_chirp_mass_density_is_built_in_the_source_frame(self):
        # The samples are in the detector frame, so they are divided by one
        # plus the redshift before the density is built.
        resampler = self.build()
        samples = gw_samples()
        assert resampler.mcKDE.dataset.mean() < samples.chirp_mass.to_numpy().mean()

    def test_the_inverse_mass_ratio_density_is_built(self):
        resampler = self.build()
        assert resampler.invqKDE.dataset.mean() > 1.0

    def test_the_spin_densities_are_only_built_for_a_mixed_binary(self):
        assert not hasattr(self.build(), "chi_1KDE")
        assert hasattr(self.build(withNSBH=True), "chi_1KDE")

    def test_both_ejecta_fitting_formulae_are_prepared(self):
        resampler = self.build()
        assert resampler.BNSEjectaFitting is not None
        assert resampler.NSBHEjectaFitting is not None

    def test_a_missing_equation_of_state_table_is_reported(self):
        gw_prior, em_prior = self.priors()
        with pytest.raises(OSError):
            StandaloneResampler(
                gw_samples(),
                em_samples(),
                gw_prior,
                em_prior,
                99,
                str(self.eos_dir),
                False,
            )


class TestEjectaResamplerPrior(ResamplerMixin):
    def test_the_unit_cube_is_rescaled_onto_the_priors(self):
        resampler = self.build()
        values = resampler.Prior(np.full(5, 0.5))
        assert len(values) == 5

    def test_the_chirp_mass_lands_inside_its_prior_range(self):
        resampler = self.build()
        chirp_mass = resampler.Prior(np.full(5, 0.5))[0]
        assert chirp_mass >= 1.0
        assert chirp_mass <= 1.5

    def test_the_cube_edges_map_to_the_prior_edges(self):
        resampler = self.build()
        low = resampler.Prior(np.full(5, 0.0))
        high = resampler.Prior(np.full(5, 1.0))
        assert low[0] == pytest.approx(1.0)
        assert high[0] == pytest.approx(1.5)

    def test_a_mixed_binary_rescales_seven_values(self):
        resampler = self.build(withNSBH=True)
        assert len(resampler.Prior(np.full(7, 0.5))) == 7


class TestEjectaResamplerLogLikelihood(ResamplerMixin):
    """The likelihood maps binary parameters through an equation of state and
    an ejecta formula, then scores the result against the electromagnetic
    density."""

    def point(self, chirp_mass=1.2, mass_ratio=0.9, eos=0.0, alpha=0.05, zeta=0.2):
        return [chirp_mass, mass_ratio, eos, alpha, zeta]

    def test_a_plausible_binary_gets_a_finite_value(self):
        resampler = self.build()
        value = resampler.LogLikelihood(self.point())
        assert np.isfinite(value)

    def test_the_value_is_a_plain_float_not_an_array(self):
        resampler = self.build()
        assert isinstance(resampler.LogLikelihood(self.point()), float)

    def test_an_equation_of_state_absent_from_the_posterior_is_excluded(self):
        resampler = self.build()
        resampler.EOSsamples = np.full_like(resampler.EOSsamples, 1)
        value = resampler.LogLikelihood(self.point(eos=2.0))
        assert value < -1e300

    def test_a_negative_dynamical_ejecta_mass_is_excluded(self):
        resampler = self.build()
        with patch.object(
            resampler.BNSEjectaFitting, "dynamic_mass_fitting_KrFo", return_value=-1.0
        ):
            value = resampler.LogLikelihood(self.point(alpha=0.0))
        assert value < -1e300

    def test_the_combined_ejecta_mode_scores_a_single_total(self):
        resampler = self.build(combine_ejecta_mass=True)
        with patch.object(
            resampler.EMKDE, "logpdf", return_value=np.array([-1.0])
        ) as logpdf:
            resampler.LogLikelihood(self.point())
        assert np.isscalar(logpdf.call_args.args[0])

    def test_the_separate_ejecta_mode_scores_both_masses(self):
        resampler = self.build(combine_ejecta_mass=False)
        with patch.object(
            resampler.EMKDE, "logpdf", return_value=np.array([-1.0])
        ) as logpdf:
            resampler.LogLikelihood(self.point())
        assert len(logpdf.call_args.args[0]) == 2

    def test_a_mixed_binary_uses_the_neutron_star_black_hole_formula(self):
        resampler = self.build(withNSBH=True)
        with patch.object(
            resampler.NSBHEjectaFitting, "dynamic_mass_fitting", return_value=0.01
        ) as fitting:
            resampler.LogLikelihood(self.point() + [0.0, 0.0])
        fitting.assert_called_once()

    def test_a_binary_neutron_star_uses_the_binary_neutron_star_formula(self):
        resampler = self.build()
        with patch.object(
            resampler.BNSEjectaFitting,
            "dynamic_mass_fitting_KrFo",
            return_value=0.01,
        ) as fitting:
            resampler.LogLikelihood(self.point())
        fitting.assert_called_once()

    def test_a_secondary_above_the_maximum_mass_is_not_excluded_as_intended(self):
        # The radius interpolation returns zero above the table's maximum
        # mass, and the compactness is then guarded by catching
        # ZeroDivisionError. numpy division by a zero float yields infinity
        # with a warning instead of raising, so that guard never fires and
        # an unsupportable secondary is scored rather than rejected. The
        # check has to test the radius directly.
        resampler = self.build()
        heavy = self.point(chirp_mass=1.49, mass_ratio=0.99)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = resampler.LogLikelihood(heavy)
        assert isinstance(value, float)

    def test_the_equation_of_state_index_is_floored_then_shifted(self):
        resampler = self.build()
        first = resampler.LogLikelihood(self.point(eos=0.2))
        second = resampler.LogLikelihood(self.point(eos=0.8))
        assert first == pytest.approx(second)

    def test_a_different_equation_of_state_changes_the_value(self):
        resampler = self.build()
        first = resampler.LogLikelihood(self.point(eos=0.0))
        second = resampler.LogLikelihood(self.point(eos=1.0))
        assert first != pytest.approx(second)


class TestMainResampling:
    """The console script reads both posteriors, builds the sampler and
    writes the combined samples out."""

    def setup_method(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_multinest_is_only_imported_when_the_script_runs(self):
        # The library needs a separately built shared object, so importing
        # the module must not require it.
        import importlib

        module = importlib.import_module("nmma.post_processing.resampling")
        assert not hasattr(module, "Solver")

    def test_the_sampler_dimension_follows_the_source_type(self):
        # Five parameters for a binary neutron star, seven once the two
        # spins are added for a mixed binary.
        source = Path(resampling.__file__).read_text()
        assert "n_dims=5" in source
        assert 'pymulti_kwargs["n_dims"] = 7' in source
