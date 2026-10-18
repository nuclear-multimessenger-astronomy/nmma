"""
gwem-resampling / lc-marginalisation with --bns-fit classifier.

The resampler's BNS branch calls the fitting functions directly (not the conversion chain), so it
needs its own hook. Stubbed classifier (the Zenodo pickles need scikit-learn 1.4.0); expected
values from the Loffredo+24 equations (arXiv:2411.02342 Eq. 9, 11).

Run: pytest nmma/tests/test_post_processing/resampling_pc.py
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bilby.core.prior import Uniform
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from bilby.gw.prior import PriorDict

from nmma.core.conversion import MultimessengerConversion
from nmma.post_processing.parser import lc_marginalisation_parser, resampling_parser
from nmma.post_processing.resampling import EjectaResamplerMixIn

EOS_DIR = Path(__file__).parent.parent / "data" / "eos_macro"
M1, M2 = 1.5, 1.3  # NS-NS for the toy EOS (M_TOV 2.2)
MC = (M1 * M2) ** 0.6 / (M1 + M2) ** 0.2


class StubClassifier:
    def __init__(self, prob):
        self.prob = prob
        self.calls = []

    def probability(self, total_mass, lambda_tilde, chi_eff, mass_ratio):
        self.calls.append((total_mass, lambda_tilde, chi_eff, mass_ratio))
        return np.full(np.shape(total_mass), self.prob, dtype=float)


class Base:  # stands in for pymultinest's Solver, which would start sampling
    def __init__(self, **kwargs):
        pass


class Resampler(EjectaResamplerMixIn, Base):
    pass


def make(drop_spins=False, **kwargs):
    rng = np.random.default_rng(0)
    n = 200
    gw = pd.DataFrame(
        dict(
            EOS=rng.integers(0, 10, n),
            luminosity_distance=np.full(n, 100.0),
            chirp_mass=MC * (1 + 0.01 * rng.standard_normal(n)),
            mass_ratio=np.clip(M2 / M1 + 0.02 * rng.standard_normal(n), 0.5, 1.0),
            chi_1=np.clip(0.1 + 0.03 * rng.standard_normal(n), -0.05, 0.3),
            chi_2=np.clip(0.02 * rng.standard_normal(n), -0.05, 0.05),
        )
    )
    if drop_spins:
        gw = gw.drop(columns=["chi_1", "chi_2"])
    em = pd.DataFrame(
        dict(
            log10_mej_dyn=-2.5 + 0.1 * rng.standard_normal(n),
            log10_mej_wind=-1.5 + 0.1 * rng.standard_normal(n),
        )
    )
    gw_prior = PriorDict(
        dict(
            chirp_mass=Uniform(1.0, 2.0),
            mass_ratio=Uniform(0.5, 1.0),
            chi_1=Uniform(-0.05, 0.3),
            chi_2=Uniform(-0.05, 0.05),
        )
    )
    em_prior = PriorDict(dict(alpha=Uniform(-0.01, 0.01), zeta=Uniform(0.1, 0.9)))
    return Resampler(gw, em, gw_prior, em_prior, 10, str(EOS_DIR), False, **kwargs)


X = (MC, M2 / M1, 0, 0.0, 0.3)  # chirp mass, q, EOS index 0 (file 1), alpha, zeta
CHI = (0.1, 0.01)
X_SPIN = X + CHI  # + chi_1, chi_2 (sampled with --bns-fit classifier)


def expected_pc_loglike(r, eos_file=1, alpha=0.0, zeta=0.3):
    masses, lambdas = r.EOS_masses_dict[eos_file], r.EOS_lambda_dict[eos_file]
    l1, l2 = np.interp((M1, M2), masses, lambdas)
    q = M2 / M1
    lt = lambda_1_lambda_2_to_lambda_tilde(l1, l2, M1, M2)
    mdyn = 1.25e-4 * lt * (1 / q - 0.982) * np.exp(-2.44 / q) + alpha
    disk = min(-1.0, 7.70 - 13.4 * q + 8.16e-3 * lt * q**2)
    logprior = (
        r.mcKDE.logpdf(MC)
        + r.invqKDE.logpdf(M1 / M2)
        + np.log(np.sum(r.EOSsamples == eos_file))
        + r.chi_1KDE.logpdf(CHI[0])
        + r.chi_2KDE.logpdf(CHI[1])
    )
    return np.nan_to_num(
        (r.EMKDE.logpdf((mdyn, zeta * 10**disk)) + logprior).item()
    )


def test_default_is_unchanged_and_classifier_free():
    r = make()
    assert r.BNSEjecta.bns_fit == "default"
    assert np.isfinite(r.LogLikelihood(X))


def test_default_bns_keeps_five_parameters_and_no_spins():
    r = make()
    assert list(r.priors.keys()) == ["chirp_mass", "mass_ratio", "EOS", "alpha", "zeta"]


def test_classifier_samples_the_spins():
    r = make(bns_fit="classifier", pc_classifier=StubClassifier(0.0))
    assert list(r.priors.keys()) == [
        "chirp_mass", "mass_ratio", "EOS", "alpha", "zeta", "chi_1", "chi_2",
    ]


def test_classifier_needs_spin_samples_in_the_gw_posterior():
    with pytest.raises(ValueError, match="chi_1"):
        make(drop_spins=True, bns_fit="classifier", pc_classifier=StubClassifier(0.0))


def test_classifier_with_no_collapse_equals_the_default_up_to_the_spin_prior():
    # same ejecta (default fits); the extra chi_1, chi_2 KDE terms are the only difference
    default = make().LogLikelihood(X)
    r = make(bns_fit="classifier", pc_classifier=StubClassifier(0.0))
    spin_prior = r.chi_1KDE.logpdf(CHI[0]) + r.chi_2KDE.logpdf(CHI[1])
    assert r.LogLikelihood(X_SPIN) == pytest.approx(default + spin_prior.item(), rel=1e-12)


def test_collapse_uses_the_loffredo_fits():
    stub = StubClassifier(1.0)
    r = make(bns_fit="classifier", pc_classifier=stub)
    assert r.LogLikelihood(X_SPIN) == pytest.approx(expected_pc_loglike(r), rel=1e-9)
    assert r.LogLikelihood(X_SPIN) != pytest.approx(make().LogLikelihood(X), abs=1e-3)


def test_sampled_spins_give_the_classifier_its_chi_eff():
    stub = StubClassifier(0.0)
    r = make(bns_fit="classifier", pc_classifier=stub)
    r.LogLikelihood(X_SPIN)
    chi_eff = float(np.ravel(stub.calls[-1][2])[0])
    assert chi_eff == pytest.approx((M1 * CHI[0] + M2 * CHI[1]) / (M1 + M2), rel=1e-9)


def test_threshold_is_forwarded():
    default = make().LogLikelihood(X)
    r = make(bns_fit="classifier", pc_classifier=StubClassifier(0.6), pc_threshold=0.7)
    spin_prior = (r.chi_1KDE.logpdf(CHI[0]) + r.chi_2KDE.logpdf(CHI[1])).item()
    assert r.LogLikelihood(X_SPIN) == pytest.approx(default + spin_prior, rel=1e-12)


def test_parsers_have_the_options():
    for parser_func in (resampling_parser, lc_marginalisation_parser):
        parser = parser_func(ArgumentParser())
        opts = {a.dest: a.default for a in parser._actions}
        assert opts["bns_fit"] == "default"
        assert opts["pc_classifier_dir"] is None and opts["pc_threshold"] == 0.5


def test_basic_cbc_builds_the_requested_ejecta_fitter(tmp_path):
    chain = MultimessengerConversion.basic_cbc(
        lambda p: p, lambda p: p, ejecta_kwargs=dict(
            bns_fit="classifier", pc_classifier=StubClassifier(1.0), pc_threshold=0.7
        )
    )
    fitter = chain._conversions[2]
    assert fitter.bns_fit == "classifier" and fitter.pc_threshold == 0.7
    default = MultimessengerConversion.basic_cbc(lambda p: p, lambda p: p)
    assert default._conversions[2].bns_fit == "default"
