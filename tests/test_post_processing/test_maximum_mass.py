import pickle
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bilby.core.prior import PriorDict, WeightedCategorical

from nmma.core.parsing import nmma_base_parsing
from nmma.eos.eos_processing import EoSConverter, TabulatedEoSGenerator
from nmma.post_processing import maximum_mass_constraint as mmc
from nmma.post_processing.parser import maximum_mass_parser

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EOS_MACRO_DIR = DATA_DIR / "eos_macro"
EOS_MICRO_DIR = DATA_DIR / "eos_micro"
PRIOR_FILE = DATA_DIR / "maximum_mass_resampling.prior"
JOINT_POSTERIOR_FILE = DATA_DIR / "GW+KN+GRB_posterior.dat"


def read_columns(eos_file: Path, usecols: list[int]) -> np.ndarray:
    "The given columns of an EOS table, one row per column"
    return np.loadtxt(eos_file, usecols=usecols).T


class TestTabulatedMicroEoS:
    n_eos = 10
    eos_index = 3
    gravitational_mass = 1.4
    macro_columns = [0, 1, 3]  # radius, mass, central pressure
    micro_columns = [0, 1, 2]  # density, energy density, pressure

    def setup_method(self) -> None:
        self.micro_eos = mmc.TabulatedMicroEoS(EOS_MICRO_DIR, EOS_MACRO_DIR)
        self.eos_file = f"{self.eos_index + 1}.dat"

    def test_init_reads_all_tables(self) -> None:
        assert self.micro_eos.Neos == self.n_eos
        assert len(self.micro_eos.tables) == self.n_eos
        assert len(self.micro_eos.macro_tables) == self.n_eos

    def test_tables_hold_the_selected_columns(self) -> None:
        expected_macro = read_columns(EOS_MACRO_DIR / self.eos_file, self.macro_columns)
        expected_micro = read_columns(EOS_MICRO_DIR / self.eos_file, self.micro_columns)
        np.testing.assert_array_equal(
            self.micro_eos.macro_tables[self.eos_index], expected_macro
        )
        np.testing.assert_array_equal(
            self.micro_eos.tables[self.eos_index], expected_micro
        )

    def test_call_selects_the_micro_table(self) -> None:
        (table,) = self.micro_eos({"EOS": self.eos_index})
        np.testing.assert_array_equal(table, self.micro_eos.tables[self.eos_index])

    def test_tov_baryonic_mass_is_cached(self) -> None:
        tov_mass = self.micro_eos.macro_tables[self.eos_index][1].max()
        expected = self.micro_eos.baryonic_mass(tov_mass, self.eos_index)

        assert self.micro_eos.tov_baryonic_mass(self.eos_index) == expected
        assert self.micro_eos._tov_baryonic_masses == {self.eos_index: expected}


class TestPostmergerLikelihood:
    sample_index = 0
    # a disk of a solar mass leaves too little baryonic mass for a collapse
    log10_mdisk_too_heavy = 0.0

    def setup_method(self) -> None:
        posterior = pd.read_csv(JOINT_POSTERIOR_FILE, header=0, delimiter=" ")
        eos_converter = EoSConverter(
            generator=TabulatedEoSGenerator(Namespace(eos_data=EOS_MACRO_DIR))
        )
        micro_eos = mmc.TabulatedMicroEoS(EOS_MICRO_DIR, EOS_MACRO_DIR)
        self.likelihood = mmc.PostmergerLikelihood(posterior, eos_converter, micro_eos)
        kde_params = self.likelihood.kde_params
        self.parameters = posterior.iloc[self.sample_index][kde_params]
        self.parameters = self.parameters.to_dict()
        self.kde_point = [self.parameters[key] for key in kde_params]

    def test_collapsing_remnant_gives_the_kde_density(self) -> None:
        expected = self.likelihood.KDE.logpdf(self.kde_point).item()
        assert self.likelihood.log_likelihood(self.parameters) == expected

    def test_light_remnant_is_rejected(self) -> None:
        self.parameters["log10_mdisk"] = self.log10_mdisk_too_heavy
        log_l = self.likelihood.log_likelihood(self.parameters)
        assert log_l == np.nan_to_num(-np.inf)

    def test_pickled_likelihood_agrees(self) -> None:
        unpickled = pickle.loads(pickle.dumps(self.likelihood))
        assert unpickled.log_likelihood(self.parameters) == (
            self.likelihood.log_likelihood(self.parameters)
        )


class TestMain:
    n_eos = 10
    kepler_parameters = ["ratio_R", "delta"]
    prior = str(PRIOR_FILE)
    outdir = Path("outdir_maximum_mass")
    label = "maximum_mass_test"

    def teardown_method(self) -> None:
        shutil.rmtree(self.outdir, ignore_errors=True)

    def test_postmerger_priors_add_a_categorical_eos(self) -> None:
        priors = mmc.setup_postmerger_priors(PriorDict(self.prior), self.n_eos)
        assert isinstance(priors["EOS"], WeightedCategorical)
        assert priors["EOS"].ncategories == self.n_eos
        assert set(priors) == set(mmc.PostmergerLikelihood.kde_params)

    def test_kepler_priors_are_required_with_use_M_max(self) -> None:
        prior = PriorDict(self.prior)
        with pytest.raises(ValueError, match="|".join(self.kepler_parameters)):
            mmc.setup_postmerger_priors(prior, self.n_eos, use_M_max=True)

    def test_main_samples_and_runs_final_diagnostics(self) -> None:
        args = nmma_base_parsing(
            maximum_mass_parser,
            [
                "--outdir", str(self.outdir),
                "--label", self.label,
                "--prior", str(PRIOR_FILE),
                "--joint-posterior", str(JOINT_POSTERIOR_FILE),
                "--eos-path-macro", str(EOS_MACRO_DIR),
                "--eos-path-micro", str(EOS_MICRO_DIR),
                "--sampler", "dynesty",
                "--nlive", str(32),
                "--sampler-kwargs", "{dlogz: 5, naccept: 5}",
                "--cpus", "1",
                "--plot",
            ],
        )  # fmt: skip
        mmc.main(args)
        assert (self.outdir / f"{self.label}_result.json").is_file()
        assert (self.outdir / f"{self.label}_TOV_mass.png").is_file()
