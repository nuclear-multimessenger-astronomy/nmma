import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.stats
from bilby.core.prior import Uniform
from bilby.gw.prior import PriorDict

from nmma.post_processing import maximum_mass_constraint as mmc


def write_macro_eos(path, maximum_mass=2.2, num=80):
    """A macroscopic table in the radius, mass, tidal deformability, central
    pressure column order the constraint expects."""
    masses = np.linspace(0.5, maximum_mass, num)
    radii = 12.0 - 0.3 * masses
    lambdas = 1000.0 * np.exp(-2.0 * masses)
    central_pressure = np.linspace(10.0, 400.0, num)
    np.savetxt(path, np.column_stack([radii, masses, lambdas, central_pressure]))


def write_micro_eos(path, num=200):
    """A microscopic table in the number density, energy density, pressure,
    sound speed squared column order."""
    number_density = np.linspace(0.05, 1.2, num)
    energy_density = 150.0 + 900.0 * number_density
    pressure = np.linspace(1.0, 500.0, num)
    sound_speed = np.full(num, 0.3)
    np.savetxt(
        path, np.column_stack([number_density, energy_density, pressure, sound_speed])
    )


class TestBaryonicKeplerMass(unittest.TestCase):
    """The Kepler limit is the largest mass a uniformly rotating remnant can
    hold, reached through a quasi-universal relation rather than a solve."""

    def test_the_limit_exceeds_the_non_rotating_maximum_mass(self):
        self.assertGreater(mmc.baryonic_Kepler_mass(2.0, 12.0, 1.2, 0.0), 2.0)

    def test_a_larger_rotation_ratio_raises_the_limit(self):
        low = mmc.baryonic_Kepler_mass(2.0, 12.0, 1.1, 0.0)
        high = mmc.baryonic_Kepler_mass(2.0, 12.0, 1.3, 0.0)
        self.assertGreater(high, low)

    def test_the_correction_term_scales_the_whole_result(self):
        base = mmc.baryonic_Kepler_mass(2.0, 12.0, 1.2, 0.0)
        corrected = mmc.baryonic_Kepler_mass(2.0, 12.0, 1.2, 0.1)
        self.assertAlmostEqual(corrected, base * 1.1)

    def test_the_baryonic_correction_follows_the_known_relation(self):
        # The baryonic mass exceeds the gravitational one by a term that
        # grows with compactness, so a smaller radius binds more tightly.
        compact = mmc.baryonic_Kepler_mass(2.0, 10.0, 1.0, 0.0)
        extended = mmc.baryonic_Kepler_mass(2.0, 14.0, 1.0, 0.0)
        self.assertGreater(compact, extended)

    def test_the_formula_is_evaluated_exactly_as_published(self):
        m_max = 1.2 * 2.0
        expected = (m_max + 0.78 / 12.0 * m_max**2) * 1.05
        self.assertAlmostEqual(mmc.baryonic_Kepler_mass(2.0, 12.0, 1.2, 0.05), expected)

    def test_it_works_elementwise_over_arrays(self):
        result = mmc.baryonic_Kepler_mass(
            np.array([1.9, 2.1]), np.array([12.0, 12.0]), 1.2, 0.0
        )
        self.assertEqual(result.shape, (2,))
        self.assertGreater(result[1], result[0])

    def test_a_heavier_non_rotating_maximum_gives_a_heavier_limit(self):
        self.assertGreater(
            mmc.baryonic_Kepler_mass(2.3, 12.0, 1.2, 0.0),
            mmc.baryonic_Kepler_mass(1.9, 12.0, 1.2, 0.0),
        )


class StandalonePostmergerInference(mmc.PostmergerInferenceMixIn):
    """The mixin is paired with pymultinest's solver in the pipeline. Pairing
    it with a plain object exercises the prior and the likelihood without the
    MultiNest library being installed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PostmergerMixin:
    n_eos = 3
    n_samples = 300

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.macro_dir = self.tmp_dir / "macro"
        self.micro_dir = self.tmp_dir / "micro"
        for directory in [self.macro_dir, self.micro_dir]:
            directory.mkdir()
        for index in range(1, self.n_eos + 1):
            write_macro_eos(self.macro_dir / f"{index}.dat", 2.0 + 0.1 * index)
            write_micro_eos(self.micro_dir / f"{index}.dat")
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def prior(self, with_kepler=False):
        priors = {
            "chirp_mass": Uniform(1.0, 1.5, "chirp_mass"),
            "eta_star": Uniform(-6.0, -2.0, "eta_star"),
            "log10_mdisk": Uniform(-3.0, -1.0, "log10_mdisk"),
            "log10_mej_dyn": Uniform(-3.0, -1.5, "log10_mej_dyn"),
        }
        if with_kepler:
            priors["ratio_R"] = Uniform(1.1, 1.3, "ratio_R")
            priors["delta"] = Uniform(0.0, 0.1, "delta")
        return PriorDict(priors)

    def posterior(self):
        generator = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "chirp_mass": generator.normal(1.2, 0.01, self.n_samples),
                "eta_star": generator.normal(-4.0, 0.3, self.n_samples),
                "EOS": generator.uniform(0, self.n_eos, self.n_samples),
                "log10_mdisk": generator.normal(-2.0, 0.2, self.n_samples),
                "log10_mej_dyn": generator.normal(-2.5, 0.2, self.n_samples),
            }
        )

    def build(self, use_M_max=False):
        return StandalonePostmergerInference(
            self.prior(with_kepler=use_M_max),
            self.posterior(),
            self.n_eos,
            str(self.macro_dir),
            str(self.micro_dir),
            use_M_max,
        )


class TestPostmergerSetup(PostmergerMixin, unittest.TestCase):
    def test_the_sampled_parameters_are_the_five_joint_ones(self):
        inference = self.build()
        self.assertEqual(
            list(inference._search_parameter_keys),
            ["chirp_mass", "eta_star", "EOS", "log10_mdisk", "log10_mej_dyn"],
        )

    def test_the_kepler_limit_adds_the_two_relation_parameters(self):
        inference = self.build(use_M_max=True)
        self.assertIn("ratio_R", inference._search_parameter_keys)
        self.assertIn("delta", inference._search_parameter_keys)

    def test_the_equation_of_state_index_spans_the_table_count(self):
        # Unlike the ejecta resampler, this prior stops at the count rather
        # than one past it.
        self.assertEqual(self.build().priors["EOS"].maximum, self.n_eos)

    def test_the_joint_posterior_becomes_a_five_dimensional_density(self):
        inference = self.build()
        self.assertIsInstance(inference.KDE, scipy.stats.gaussian_kde)
        self.assertEqual(inference.KDE.d, 5)

    def test_both_table_directories_are_kept_as_paths(self):
        inference = self.build()
        self.assertIsInstance(inference.eos_path_macro, Path)
        self.assertIsInstance(inference.eos_path_micro, Path)

    def test_a_posterior_missing_a_required_column_is_reported(self):
        posterior = self.posterior().drop(columns=["log10_mdisk"])
        with self.assertRaises(AttributeError):
            StandalonePostmergerInference(
                self.prior(),
                posterior,
                self.n_eos,
                str(self.macro_dir),
                str(self.micro_dir),
                False,
            )

    def test_a_prior_missing_a_required_key_is_reported(self):
        prior = self.prior()
        del prior["eta_star"]
        with self.assertRaises(KeyError):
            StandalonePostmergerInference(
                prior,
                self.posterior(),
                self.n_eos,
                str(self.macro_dir),
                str(self.micro_dir),
                False,
            )


class TestPostmergerPrior(PostmergerMixin, unittest.TestCase):
    def test_the_unit_cube_is_rescaled_onto_the_priors(self):
        self.assertEqual(len(self.build().Prior(np.full(5, 0.5))), 5)

    def test_the_cube_edges_map_to_the_prior_edges(self):
        inference = self.build()
        self.assertAlmostEqual(inference.Prior(np.full(5, 0.0))[0], 1.0)
        self.assertAlmostEqual(inference.Prior(np.full(5, 1.0))[0], 1.5)

    def test_the_kepler_run_rescales_seven_values(self):
        self.assertEqual(len(self.build(use_M_max=True).Prior(np.full(7, 0.5))), 7)


class TestBaryonicMass(PostmergerMixin, unittest.TestCase):
    """Integrates the stellar structure equations outward to turn a
    gravitational mass into a baryonic one."""

    def test_a_baryonic_mass_is_returned(self):
        inference = self.build()
        mass = inference.baryonic_mass(1.4, 1)
        self.assertTrue(np.isfinite(mass))
        self.assertGreater(mass, 0.0)

    def test_a_heavier_star_has_a_larger_baryonic_mass(self):
        inference = self.build()
        self.assertGreater(
            inference.baryonic_mass(1.8, 1), inference.baryonic_mass(1.2, 1)
        )

    def test_each_equation_of_state_gives_its_own_answer(self):
        inference = self.build()
        self.assertNotAlmostEqual(
            inference.baryonic_mass(1.4, 1), inference.baryonic_mass(1.4, 3)
        )

    def test_a_failed_integration_is_warned_about_rather_than_hidden(self):
        inference = self.build()
        with patch.object(mmc.scipy.integrate, "simpson", return_value=np.nan):
            with self.assertWarns(UserWarning):
                inference.baryonic_mass(1.4, 1)

    def test_a_missing_table_is_reported(self):
        inference = self.build()
        with self.assertRaises(OSError):
            inference.baryonic_mass(1.4, 99)


class TestPostmergerLogLikelihood(PostmergerMixin, unittest.TestCase):
    """Accepts only those samples where the remnant was heavy enough to
    collapse, using the joint posterior as the prior."""

    def point(
        self,
        chirp_mass=1.2,
        eta_star=-4.0,
        eos=0.0,
        log10_mdisk=-2.0,
        log10_mej_dyn=-2.5,
    ):
        return [chirp_mass, eta_star, eos, log10_mdisk, log10_mej_dyn]

    def test_a_collapsing_remnant_keeps_the_posterior_density(self):
        inference = self.build()
        with patch.object(inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]):
            value = inference.LogLikelihood(self.point())
        self.assertTrue(np.isfinite(value))

    def test_a_surviving_remnant_is_excluded(self):
        # If the threshold mass exceeds the remnant, the remnant would not
        # have collapsed, which contradicts the assumption.
        inference = self.build()
        with patch.object(inference, "baryonic_mass", side_effect=[1.0, 0.9, 5.0]):
            value = inference.LogLikelihood(self.point())
        self.assertLess(value, -1e300)

    def test_the_joint_posterior_is_used_as_the_prior(self):
        inference = self.build()
        with patch.object(
            inference.KDE, "logpdf", return_value=np.array([-3.0])
        ) as logpdf:
            with patch.object(inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]):
                value = inference.LogLikelihood(self.point())
        logpdf.assert_called_once()
        self.assertAlmostEqual(value, -3.0)

    def test_the_threshold_is_the_baryonic_tov_mass_by_default(self):
        inference = self.build()
        with patch.object(
            inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]
        ) as baryonic:
            inference.LogLikelihood(self.point())
        self.assertEqual(baryonic.call_count, 3)

    def test_the_kepler_limit_replaces_the_third_structure_solve(self):
        # The quasi-universal relation is cheaper than integrating the
        # structure equations again.
        inference = self.build(use_M_max=True)
        with patch.object(
            inference, "baryonic_mass", side_effect=[2.0, 1.8]
        ) as baryonic:
            with patch.object(mmc, "baryonic_Kepler_mass", return_value=1.0) as kepler:
                inference.LogLikelihood(self.point() + [1.2, 0.05])
        self.assertEqual(baryonic.call_count, 2)
        kepler.assert_called_once()

    def test_the_ejecta_and_disk_are_removed_from_the_remnant(self):
        # The remnant is what is left of the two stars after the disk and the
        # dynamical ejecta have been shed.
        inference = self.build()
        with patch.object(inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]):
            light_ejecta = inference.LogLikelihood(
                self.point(log10_mdisk=-3.0, log10_mej_dyn=-3.0)
            )
        with patch.object(inference, "baryonic_mass", side_effect=[2.0, 1.8, 4.5]):
            heavy_threshold = inference.LogLikelihood(
                self.point(log10_mdisk=-3.0, log10_mej_dyn=-3.0)
            )
        self.assertTrue(np.isfinite(light_ejecta))
        self.assertLess(heavy_threshold, -1e300)

    def test_the_equation_of_state_index_is_floored_then_shifted(self):
        inference = self.build()
        with patch.object(
            inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]
        ) as baryonic:
            inference.LogLikelihood(self.point(eos=0.7))
        self.assertEqual(baryonic.call_args.args[1], 1)

    def test_the_symmetric_mass_ratio_comes_from_the_logarithmic_parameter(self):
        # Sampling the logarithm keeps the mass ratio away from the equal
        # mass boundary where the conversion is singular.
        inference = self.build()
        with patch.object(inference, "baryonic_mass", side_effect=[2.0, 1.8, 1.0]):
            with patch.object(
                mmc.conversion,
                "symmetric_mass_ratio_to_mass_ratio",
                return_value=0.9,
            ) as conversion:
                inference.LogLikelihood(self.point(eta_star=-4.0))
        self.assertAlmostEqual(conversion.call_args.args[0], 0.25 - np.exp(-4.0))


class TestMaximumMassResampling(unittest.TestCase):
    """The script validates the prior against the chosen threshold before it
    starts sampling, and counts the tables from the directory."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.macro_dir = self.tmp_dir / "macro"
        self.micro_dir = self.tmp_dir / "micro"
        for directory in [self.macro_dir, self.micro_dir]:
            directory.mkdir()
        for index in range(1, 4):
            write_macro_eos(self.macro_dir / f"{index}.dat")
            write_micro_eos(self.micro_dir / f"{index}.dat")
        self.prior_file = self.tmp_dir / "post.prior"
        self.prior_file.write_text(
            "chirp_mass = Uniform(minimum=1.0, maximum=1.5, name='chirp_mass')\n"
            "eta_star = Uniform(minimum=-6, maximum=-2, name='eta_star')\n"
            "log10_mdisk = Uniform(minimum=-3, maximum=-1, name='log10_mdisk')\n"
            "log10_mej_dyn = Uniform(minimum=-3, maximum=-1.5, name='log10_mej_dyn')\n"
        )
        self.posterior_file = self.tmp_dir / "joint.dat"
        pd.DataFrame(
            {
                "chirp_mass": [1.2, 1.21],
                "eta_star": [-4.0, -4.1],
                "EOS": [0.5, 1.5],
                "log10_mdisk": [-2.0, -2.1],
                "log10_mej_dyn": [-2.5, -2.6],
            }
        ).to_csv(self.posterior_file, sep=" ", index=False)
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def args(self, **kwargs):
        defaults = dict(
            outdir=str(self.tmp_dir),
            joint_posterior=str(self.posterior_file),
            prior=str(self.prior_file),
            eos_path_macro=str(self.macro_dir),
            eos_path_micro=str(self.micro_dir),
            use_M_Kepler=False,
            nlive=10,
        )
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_a_kepler_run_without_the_relation_priors_is_refused(self):
        # The prior file has four keys, but the Kepler threshold needs six.
        with self.assertRaises(Exception) as caught:
            mmc.maximum_mass_resampling(self.args(use_M_Kepler=True))
        self.assertIn("ratio_R", str(caught.exception))

    def test_the_sampler_output_directory_is_created(self):
        # The directory is made before the sampler is imported, so it exists
        # even on a machine without the MultiNest library. That import both
        # raises and calls sys.exit, so the guard has to be this wide.
        try:
            mmc.maximum_mass_resampling(self.args())
        except BaseException:
            pass
        self.assertTrue((self.tmp_dir / "pm").is_dir())

    def test_multinest_is_only_imported_when_the_script_runs(self):
        self.assertFalse(hasattr(mmc, "Solver"))

    def test_the_table_count_is_taken_from_the_macroscopic_directory(self):
        # It counts every entry, so a stray file in that directory would be
        # counted as an equation of state.
        self.assertEqual(len(list(self.macro_dir.iterdir())), 3)

    def test_the_entry_point_parses_its_own_arguments_when_none_are_given(self):
        with patch.object(
            mmc, "nmma_base_parsing", return_value=self.args()
        ) as parsing:
            with patch.object(mmc, "maximum_mass_resampling"):
                mmc.main()
        parsing.assert_called_once_with(mmc.maximum_mass_parser)

    def test_the_entry_point_uses_given_arguments_without_reparsing(self):
        args = self.args()
        with patch.object(mmc, "nmma_base_parsing") as parsing:
            with patch.object(mmc, "maximum_mass_resampling") as resampling:
                mmc.main(args)
        parsing.assert_not_called()
        resampling.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
