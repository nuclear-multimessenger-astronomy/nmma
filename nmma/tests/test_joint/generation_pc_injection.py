"""
nmma-generation injection with the classifier BNS fit (--injection-bns-fit classifier).

The injected system is chosen by the injection-* options only: like --force-bns/--force-nsbh
(gotcha 14), the analysis option --bns-fit must NOT change the data. The classifier is stubbed
(the Zenodo pickles need scikit-learn 1.4.0); expected values come from the Loffredo+24
equations (arXiv:2411.02342 Eq. 9, 11) as quoted in arXiv:2607.28438.

Run: pytest nmma/tests/test_joint/generation_pc_injection.py
"""

import numpy as np
import pytest
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde
from generation_eos_injection import converted_injection

from nmma.core.prompt_collapse import PromptCollapseClassifier

M1, M2 = 1.5, 1.3  # NS-NS on the toy EOS (M_TOV 2.2)


class StubClassifier:
    def __init__(self, prob):
        self.prob = prob

    def probability(self, total_mass, lambda_tilde, chi_eff, mass_ratio):
        return np.full(np.shape(total_mass), self.prob, dtype=float)


@pytest.fixture
def stub(monkeypatch):
    """Make --injection-pc-classifier-dir load a stub with the requested P(PC)."""

    def install(prob):
        monkeypatch.setattr(
            PromptCollapseClassifier,
            "from_directory",
            classmethod(lambda cls, directory, classifier="A": StubClassifier(prob)),
        )

    return install


def pc_reference(c, zeta=0.3):
    q = c["mass_2_source"] / c["mass_1_source"]
    lt = lambda_1_lambda_2_to_lambda_tilde(
        c["lambda_1"], c["lambda_2"], c["mass_1_source"], c["mass_2_source"]
    )
    mdyn = 1.25e-4 * lt * (1 / q - 0.982) * np.exp(-2.44 / q)
    disk = min(-1.0, 7.70 - 13.4 * q + 8.16e-3 * lt * q**2)
    return np.log10(mdyn), np.log10(zeta) + disk


INJ = dict(injection_bns_fit="classifier", injection_pc_classifier_dir="stub")


def test_injection_uses_the_collapse_fits_when_the_classifier_says_collapse(tmp_path, stub):
    stub(1.0)
    c = converted_injection(tmp_path, M1, M2, **INJ)
    ref_dyn, ref_wind = pc_reference(c)
    assert c["log10_mej_dyn"] == pytest.approx(ref_dyn, rel=1e-9)
    assert c["log10_mej_wind"] == pytest.approx(ref_wind, rel=1e-9)
    default = converted_injection(tmp_path / "default", M1, M2)
    assert c["log10_mej_dyn"] != pytest.approx(default["log10_mej_dyn"], abs=0.05)


def test_injection_is_the_default_fit_when_no_collapse(tmp_path, stub):
    stub(0.0)
    c = converted_injection(tmp_path / "new", M1, M2, **INJ)
    default = converted_injection(tmp_path / "default", M1, M2)
    for key in ("log10_mej_dyn", "log10_mej_wind"):
        assert c[key] == pytest.approx(default[key], rel=1e-12)


def test_injection_threshold_option(tmp_path, stub):
    stub(0.6)
    default = converted_injection(tmp_path / "default", M1, M2)
    above = converted_injection(tmp_path / "a", M1, M2, injection_pc_threshold=0.5, **INJ)
    below = converted_injection(tmp_path / "b", M1, M2, injection_pc_threshold=0.7, **INJ)
    assert above["log10_mej_wind"] == pytest.approx(pc_reference(above)[1], rel=1e-9)
    assert below["log10_mej_wind"] == pytest.approx(default["log10_mej_wind"], rel=1e-12)


def test_analysis_bns_fit_does_not_change_the_injection(tmp_path, stub):
    stub(1.0)
    default = converted_injection(tmp_path / "default", M1, M2)
    analysis_flag = converted_injection(
        tmp_path / "flag", M1, M2, bns_fit="classifier", pc_classifier_dir="stub"
    )
    for key in ("log10_mej_dyn", "log10_mej_wind", "lambda_1", "lambda_2"):
        assert analysis_flag[key] == pytest.approx(default[key], rel=1e-12), key


def test_nsbh_injection_is_unaffected_and_never_asks_the_classifier(tmp_path, monkeypatch):
    class Boom:
        def probability(self, *a, **k):
            raise AssertionError("classifier must not be used for an NSBH")

    monkeypatch.setattr(
        PromptCollapseClassifier, "from_directory",
        classmethod(lambda cls, directory, classifier="A": Boom()),
    )
    c = converted_injection(tmp_path / "new", 2.8, 1.3, **INJ)  # BH above M_TOV
    default = converted_injection(tmp_path / "default", 2.8, 1.3)
    for key in ("log10_mej_dyn", "log10_mej_wind"):
        assert c[key] == pytest.approx(default[key], rel=1e-12)


def test_configuration_errors_are_not_swallowed(tmp_path):
    # generation used to continue silently without ejecta when the conversion failed;
    # a requested classifier fit that cannot be set up must stop the run instead
    with pytest.raises(ValueError, match="pc-classifier-dir"):
        converted_injection(tmp_path, M1, M2, injection_bns_fit="classifier")
