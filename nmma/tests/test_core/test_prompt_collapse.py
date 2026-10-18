"""
Prompt-collapse (PC) treatment in the BNS ejecta fit: a remnant classifier
(Puecher & Dietrich 2024, arXiv:2408.10678) gives P(PC); where P > threshold the
Loffredo+24 (arXiv:2411.02342) fits replace the default dynamical/disk fits, as in
arXiv:2607.28438. Off by default.

Expected values are written from the paper equations, independent of NMMA:
  m_dyn    = 1.25e-4 * Lt * (1/q - 0.982) * exp(-2.44/q)             q = m2/m1
  log_disk = min(-1, 7.70 - 13.4 q + 8.16e-3 Lt q^2)
  wind     = log10(zeta) + log_disk,   dyn = log10(m_dyn + alpha)
The classifier is replaced by stubs: the real pickles need scikit-learn 1.4.0.
"""

import numpy as np
import pytest
from bilby.gw.conversion import lambda_1_lambda_2_to_lambda_tilde

from nmma.core.conversion import BNSEjectaFitting, KilonovaEjectaFitting
from nmma.core.prompt_collapse import PromptCollapseClassifier

# 2.2 + 1.35 on EOS #4611 (AP3-like): radii [km], Lambdas, M_TOV, R_1.6
SYSTEM = dict(
    mass_1_source=2.2,
    mass_2_source=1.35,
    radius_1=11.69,
    radius_2=12.11,
    lambda_1=12.8,
    lambda_2=499.0,
    TOV_mass=2.304,
    R_16=12.12,
    alpha=0.0,
    ratio_zeta=0.3,
)


def system(**updates):
    p = {k: np.atleast_1d(np.array(v, dtype=float)) for k, v in SYSTEM.items()}
    p.update({k: np.atleast_1d(np.array(v, dtype=float)) for k, v in updates.items()})
    return p


def ref_pc(m1, m2, l1, l2, alpha=0.0, zeta=0.3):
    q = m2 / m1
    lt = lambda_1_lambda_2_to_lambda_tilde(l1, l2, m1, m2)
    mdyn = 1.25e-4 * lt * (1 / q - 0.982) * np.exp(-2.44 / q)
    disk = np.minimum(-1.0, 7.70 - 13.4 * q + 8.16e-3 * lt * q**2)
    return tuple(np.atleast_1d(x) for x in (np.log10(mdyn + alpha), np.log10(zeta) + disk, disk))


class StubClassifier:
    """Fixed P(PC) (or a function of Mtot); records the inputs it was called with."""

    def __init__(self, prob):
        self.prob = prob
        self.calls = []

    def probability(self, total_mass, lambda_tilde, chi_eff, mass_ratio):
        self.calls.append((total_mass, lambda_tilde, chi_eff, mass_ratio))
        total_mass = np.asarray(total_mass, dtype=float)
        if callable(self.prob):
            return self.prob(total_mass)
        return np.full_like(total_mass, self.prob)


def dyn_wind(fit, params):
    dyn, wind, _, _ = fit.bns_ejecta_conversion(params)
    return np.atleast_1d(dyn), np.atleast_1d(wind)


# ─── default behaviour is unchanged ──────────────────────────────────────────
def test_off_by_default_matches_the_plain_fit():
    p = system()
    plain = BNSEjectaFitting().bns_ejecta_conversion(p)
    with_none = BNSEjectaFitting(bns_fit="default").bns_ejecta_conversion(p)
    for a, b in zip(plain, with_none):
        assert np.array_equal(a, b)
    # value checked against the earlier hand scan: 2.2 + 1.35 on #4611
    assert plain[0][0] == pytest.approx(-1.849, abs=2e-3)
    assert plain[1][0] == pytest.approx(-1.404, abs=2e-3)


def test_default_is_the_plain_fit_and_has_no_classifier():
    fit = BNSEjectaFitting()
    assert fit.bns_fit == "default" and fit.pc_classifier is None


def test_classifier_with_zero_probability_changes_nothing():
    p = system()
    plain = BNSEjectaFitting().bns_ejecta_conversion(p)
    out = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(0.0)).bns_ejecta_conversion(p)
    for a, b in zip(plain, out):
        assert np.array_equal(a, b)


# ─── PC branch ───────────────────────────────────────────────────────────────
def test_pc_branch_matches_paper_equations():
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0))
    dyn, wind = dyn_wind(fit, system())
    ref_dyn, ref_wind, _ = ref_pc(2.2, 1.35, 12.8, 499.0)
    assert dyn[0] == pytest.approx(ref_dyn[0], rel=1e-9)
    assert wind[0] == pytest.approx(ref_wind[0], rel=1e-9)
    assert dyn[0] == pytest.approx(-3.76, abs=0.02)  # value from the dev study
    assert wind[0] == pytest.approx(-1.52, abs=0.02)


def test_alpha_and_zeta_enter_the_pc_branch():
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0))
    dyn, wind = dyn_wind(fit, system(alpha=1e-3, ratio_zeta=0.5))
    ref_dyn, ref_wind, _ = ref_pc(2.2, 1.35, 12.8, 499.0, alpha=1e-3, zeta=0.5)
    assert dyn[0] == pytest.approx(ref_dyn[0], rel=1e-9)
    assert wind[0] == pytest.approx(ref_wind[0], rel=1e-9)


def test_grb_energy_uses_the_pc_disk_mass():
    p = system()
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0))
    out = fit.bns_parameter_conversion(p)  # dyn, wind, total, E0
    _, _, ref_disk = ref_pc(2.2, 1.35, 12.8, 499.0)
    expected_E0 = BNSEjectaFitting().grb_energy_conversion(p, ref_disk)
    assert out[3][0] == pytest.approx(float(np.squeeze(expected_E0)), rel=1e-9)
    plain = BNSEjectaFitting().bns_parameter_conversion(p)
    assert out[3][0] != pytest.approx(plain[3][0], abs=0.1)  # disk really changed


def test_total_ejecta_is_consistent_with_the_pc_components():
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0))
    dyn, wind, total, _ = fit.bns_ejecta_conversion(system())
    assert 10 ** total[0] == pytest.approx(10 ** dyn[0] + 10 ** wind[0], rel=1e-9)


# ─── threshold ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "prob,threshold,is_pc",
    [(0.6, 0.5, True), (0.4, 0.5, False), (0.6, 0.7, False), (0.8, 0.7, True),
     (0.5, 0.5, False), (1.0, 0.99, True), (0.0, 0.0, False)],
)
def test_threshold_is_strictly_greater_than(prob, threshold, is_pc):
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(prob), pc_threshold=threshold)
    dyn, wind = dyn_wind(fit, system())
    ref_dyn, ref_wind, _ = ref_pc(2.2, 1.35, 12.8, 499.0)
    plain_dyn, plain_wind = dyn_wind(BNSEjectaFitting(), system())
    expected = (ref_dyn[0], ref_wind[0]) if is_pc else (plain_dyn[0], plain_wind[0])
    assert (dyn[0], wind[0]) == pytest.approx(expected, rel=1e-9)


def test_default_threshold_is_one_half():
    assert BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(0.5)).pc_threshold == 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.1, np.nan])
def test_bad_threshold_raises(bad):
    with pytest.raises(ValueError):
        BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(0.5), pc_threshold=bad)


def test_nan_probability_is_not_prompt_collapse():
    plain_dyn, plain_wind = dyn_wind(BNSEjectaFitting(), system())
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(np.nan))
    dyn, wind = dyn_wind(fit, system())
    assert (dyn[0], wind[0]) == pytest.approx((plain_dyn[0], plain_wind[0]))


# ─── arrays: mixed PC / non-PC rows ──────────────────────────────────────────
def test_mixed_rows_follow_the_classifier_row_by_row():
    m1 = np.array([1.4, 2.2, 1.6])
    m2 = np.array([1.4, 1.35, 1.3])
    l1 = np.array([400.0, 12.8, 150.0])
    l2 = np.array([400.0, 499.0, 600.0])
    r1 = np.array([12.1, 11.69, 12.1])
    r2 = np.array([12.1, 12.11, 12.1])
    p = system(mass_1_source=m1, mass_2_source=m2, lambda_1=l1, lambda_2=l2,
               radius_1=r1, radius_2=r2,
               TOV_mass=np.full(3, 2.304), R_16=np.full(3, 12.12),
               alpha=np.zeros(3), ratio_zeta=np.full(3, 0.3))
    fit = BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(lambda mt: (mt > 3.0) * 1.0))
    dyn, wind = dyn_wind(fit, p)
    plain_dyn, plain_wind = dyn_wind(BNSEjectaFitting(), p)
    pc_dyn, pc_wind, _ = ref_pc(m1, m2, l1, l2)
    is_pc = (m1 + m2) > 3.0  # [False, True, False]
    np.testing.assert_allclose(dyn, np.where(is_pc, pc_dyn, plain_dyn), rtol=1e-9)
    np.testing.assert_allclose(wind, np.where(is_pc, pc_wind, plain_wind), rtol=1e-9)
    assert is_pc.tolist() == [False, True, False]


def test_not_a_ns_stays_minus_inf_even_if_collapsing():
    p = system(radius_1=0.0)
    dyn, wind = dyn_wind(BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0)), p)
    assert dyn[0] == -np.inf and wind[0] == -np.inf


# ─── what the classifier is fed ──────────────────────────────────────────────
def test_classifier_inputs_are_mtot_lambdatilde_chieff_and_q_ge_1():
    stub = StubClassifier(0.0)
    p = system(chi_1=0.1, chi_2=-0.04)
    BNSEjectaFitting(bns_fit="classifier", pc_classifier=stub).bns_ejecta_conversion(p)
    mt, lt, chi_eff, q = (np.asarray(x, dtype=float) for x in stub.calls[-1])
    assert mt[0] == pytest.approx(3.55)
    assert lt[0] == pytest.approx(
        lambda_1_lambda_2_to_lambda_tilde(12.8, 499.0, 2.2, 1.35), rel=1e-9
    )
    assert chi_eff[0] == pytest.approx((2.2 * 0.1 + 1.35 * -0.04) / 3.55, rel=1e-9)
    assert q[0] == pytest.approx(2.2 / 1.35, rel=1e-9)  # classifier uses m1/m2 >= 1


def test_missing_spins_default_to_zero():
    stub = StubClassifier(0.0)
    BNSEjectaFitting(bns_fit="classifier", pc_classifier=stub).bns_ejecta_conversion(system())
    assert float(np.asarray(stub.calls[-1][2])[0]) == 0.0


def test_missing_lambdas_raise_a_clear_error():
    p = system()
    del p["lambda_1"]
    with pytest.raises(KeyError, match="lambda_1"):
        BNSEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0)).bns_ejecta_conversion(p)


# ─── KilonovaEjectaFitting routing ───────────────────────────────────────────
def test_kilonova_fitting_passes_the_classifier_to_the_bns_branch_only():
    fit = KilonovaEjectaFitting(bns_fit="classifier", pc_classifier=StubClassifier(1.0))
    bns = system()
    out = fit.ejecta_parameter_conversion(bns)
    ref_dyn, ref_wind, _ = ref_pc(2.2, 1.35, 12.8, 499.0)
    assert out[0][0] == pytest.approx(ref_dyn[0], rel=1e-9)
    assert out[1][0] == pytest.approx(ref_wind[0], rel=1e-9)

    nsbh = system(mass_1_source=2.7, mass_2_source=1.25, radius_1=0.0, lambda_1=0.0,
                  radius_2=12.09, lambda_2=700.0, chi_1=0.3)
    stub = StubClassifier(1.0)
    out_nsbh = KilonovaEjectaFitting(bns_fit="classifier", pc_classifier=stub).ejecta_parameter_conversion(nsbh)
    plain_nsbh = KilonovaEjectaFitting().ejecta_parameter_conversion(nsbh)
    np.testing.assert_allclose(out_nsbh, plain_nsbh)
    assert stub.calls == []  # the classifier is never asked about an NSBH


# ─── PromptCollapseClassifier wrapper (fake sklearn objects) ─────────────────
class FakeScaler:
    feature_names_in_ = np.array(["Mtot", "LambdaTilde", "ChiEff", "Mratio_fixed"])

    def transform(self, x):
        self.seen = x.copy()
        return x.to_numpy()


class FakeModel:
    """predict_proba columns ordered as classes_ = [0, 1] (1 = prompt collapse)."""

    classes_ = np.array([0, 1])

    def predict_proba(self, x):
        p = np.clip((x[:, 0] - 3.0), 0, 1)  # depends on Mtot only
        return np.column_stack([1 - p, p])


def test_wrapper_returns_the_collapse_column_with_feature_order():
    scaler, model = FakeScaler(), FakeModel()
    clf = PromptCollapseClassifier(model, scaler)
    p = clf.probability(
        np.array([2.8, 3.5, 3.2]), np.array([400.0] * 3), np.zeros(3), np.array([1.0, 1.6, 1.2])
    )
    np.testing.assert_allclose(p, [0.0, 0.5, 0.2], atol=1e-12)
    assert list(scaler.seen.columns) == list(FakeScaler.feature_names_in_)
    assert scaler.seen["Mratio_fixed"].tolist() == [1.0, 1.6, 1.2]


def test_wrapper_picks_the_column_of_label_1_whatever_the_class_order():
    class Reversed(FakeModel):
        classes_ = np.array([1, 0])

        def predict_proba(self, x):
            return FakeModel.predict_proba(self, x)[:, ::-1]

    clf = PromptCollapseClassifier(Reversed(), FakeScaler())
    p = clf.probability(np.array([3.5]), np.array([400.0]), np.zeros(1), np.array([1.2]))
    assert p[0] == pytest.approx(0.5)


def test_wrapper_flags_extrapolation_outside_the_training_range():
    clf = PromptCollapseClassifier(FakeModel(), FakeScaler())
    inside = clf.outside_training_range(np.array([3.0]), np.array([0.0]), np.array([1.5]))
    out_mtot = clf.outside_training_range(np.array([3.55]), np.array([0.0]), np.array([1.63]))
    out_q = clf.outside_training_range(np.array([3.0]), np.array([0.0]), np.array([2.5]))
    assert not inside[0] and out_mtot[0] and out_q[0]


def test_from_directory_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="classifierA_model.pkl"):
        PromptCollapseClassifier.from_directory(tmp_path)


# ─── choosing the "old" (default) or the "new" (classifier) approach ────────
def test_invalid_bns_fit_raises():
    with pytest.raises(ValueError, match="bns_fit"):
        BNSEjectaFitting(bns_fit="loffredo")


def test_classifier_fit_needs_a_classifier():
    with pytest.raises(ValueError, match="pc_classifier"):
        BNSEjectaFitting(bns_fit="classifier")


def test_default_fit_refuses_a_classifier_instead_of_ignoring_it():
    with pytest.raises(ValueError, match="ignore"):
        BNSEjectaFitting(bns_fit="default", pc_classifier=StubClassifier(1.0))


def test_classifier_directory_is_loaded_lazily(tmp_path):
    with pytest.raises(FileNotFoundError, match="classifierA_model.pkl"):
        BNSEjectaFitting(bns_fit="classifier", pc_classifier=tmp_path)


def test_old_and_new_approach_differ_only_for_collapsing_systems():
    p = system()
    old = BNSEjectaFitting(bns_fit="default").bns_ejecta_conversion(p)
    new_nopc = BNSEjectaFitting(
        bns_fit="classifier", pc_classifier=StubClassifier(0.0)
    ).bns_ejecta_conversion(p)
    new_pc = BNSEjectaFitting(
        bns_fit="classifier", pc_classifier=StubClassifier(1.0)
    ).bns_ejecta_conversion(p)
    assert all(np.array_equal(a, b) for a, b in zip(old, new_nopc))
    assert new_pc[0][0] != old[0][0] and new_pc[1][0] != old[1][0]


# ─── user-facing options (CLI -> conversion chain) ───────────────────────────
from argparse import ArgumentParser, Namespace  # noqa: E402

from nmma.core.conversion import (  # noqa: E402
    MultimessengerConversion,
    ejecta_instruction_from_args,
)
from nmma.joint.joint_parsing import joint_likelihood_parsing  # noqa: E402


def parse(*argv):
    return joint_likelihood_parsing(ArgumentParser()).parse_args(list(argv))


def test_cli_defaults_keep_the_old_approach():
    args = parse()
    assert args.bns_fit == "default"
    assert args.pc_classifier_dir is None
    assert args.pc_threshold == 0.5


def test_cli_options_parse():
    args = parse(
        "--bns-fit", "classifier", "--pc-classifier-dir", "/x", "--pc-threshold", "0.7"
    )
    assert (args.bns_fit, args.pc_classifier_dir, args.pc_threshold) == (
        "classifier", "/x", 0.7,
    )


def test_cli_rejects_unknown_bns_fit():
    with pytest.raises(SystemExit):
        parse("--bns-fit", "foo")


def test_instruction_is_plain_true_for_the_old_approach():
    assert ejecta_instruction_from_args(Namespace()) is True  # old Namespaces
    assert ejecta_instruction_from_args(parse()) is True


def test_instruction_carries_the_classifier_options():
    args = parse(
        "--bns-fit", "classifier", "--pc-classifier-dir", "/x", "--pc-threshold", "0.7"
    )
    assert ejecta_instruction_from_args(args) == dict(
        bns_fit="classifier", pc_classifier="/x", pc_threshold=0.7
    )


def test_instruction_needs_a_directory_for_the_classifier_approach():
    with pytest.raises(ValueError, match="pc-classifier-dir"):
        ejecta_instruction_from_args(parse("--bns-fit", "classifier"))


def test_conversion_chain_builds_the_requested_fitter(tmp_path):
    chain = MultimessengerConversion.from_dict({"ejecta": True})
    fitters = [c for c in chain._conversions if isinstance(c, KilonovaEjectaFitting)]
    assert fitters[0].bns_fit == "default"
    with pytest.raises(FileNotFoundError, match="classifierA_model.pkl"):
        MultimessengerConversion.from_dict(
            {"ejecta": dict(bns_fit="classifier", pc_classifier=str(tmp_path))}
        )


# ─── sklearn-free classifier files (classifier<X>_numpy.npz) ─────────────────
from nmma.core.prompt_collapse import NumpyGradientBoostingClassifier  # noqa: E402


def stump(feature, threshold, left_value, right_value):
    """One decision stump as (left, right, feature, threshold, value) node arrays."""
    return (
        np.array([1, -1, -1]), np.array([2, -1, -1]), np.array([feature, -2, -2]),
        np.array([threshold, -2.0, -2.0]), np.array([0.0, left_value, right_value]),
    )


def write_npz(path, trees, tree_class, init_raw, classes, lr=0.1, mean=None, scale=None):
    start = np.cumsum([0] + [len(t[0]) for t in trees])
    np.savez(
        path,
        classes=np.array(classes), learning_rate=lr, init_raw=np.array(init_raw),
        tree_class=np.array(tree_class), tree_start=start,
        left=np.concatenate([t[0] for t in trees]),
        right=np.concatenate([t[1] for t in trees]),
        feature=np.concatenate([t[2] for t in trees]),
        threshold=np.concatenate([t[3] for t in trees]),
        value=np.concatenate([t[4] for t in trees]),
        mean=np.zeros(4) if mean is None else np.array(mean),
        scale=np.ones(4) if scale is None else np.array(scale),
    )


def expit(x):
    return 1 / (1 + np.exp(-x))


def test_numpy_binary_model_matches_hand_computation(tmp_path):
    # raw = init + lr * (tree0 + tree1); stump 0 splits Mtot (scaled) at 0, stump 1 LambdaTilde at 1
    write_npz(tmp_path / "m.npz", [stump(0, 0.0, -1.0, 2.0), stump(1, 1.0, 0.5, -3.0)],
              [0, 0], [-0.4], [0, 1], lr=0.1, mean=[3.0, 100.0, 0, 1], scale=[0.5, 50.0, 1, 1])
    clf = NumpyGradientBoostingClassifier.from_file(tmp_path / "m.npz")
    x = np.array([[2.0, 100.0, 0.0, 1.0],   # scaled [-2, 0]: -1.0 and 0.5
                  [3.5, 200.0, 0.0, 1.0]])  # scaled [ 1, 2]:  2.0 and -3.0
    raw = np.array([-0.4 + 0.1 * (-1.0 + 0.5), -0.4 + 0.1 * (2.0 - 3.0)])
    proba = clf.predict_proba(clf.transform(x))
    np.testing.assert_allclose(proba[:, 1], expit(raw), rtol=1e-12)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(clf.classes_) == [0, 1]


def test_numpy_threshold_goes_left_on_equal(tmp_path):
    write_npz(tmp_path / "m.npz", [stump(0, 1.0, -5.0, 5.0)], [0], [0.0], [0, 1], lr=1.0)
    clf = NumpyGradientBoostingClassifier.from_file(tmp_path / "m.npz")
    p = clf.predict_proba(np.array([[1.0, 0, 0, 0], [1.0000001, 0, 0, 0]]))[:, 1]
    np.testing.assert_allclose(p, [expit(-5.0), expit(5.0)], rtol=1e-6)  # float32 inputs


def test_numpy_multiclass_model_is_a_softmax(tmp_path):
    trees = [stump(0, 0.0, 1.0, -1.0), stump(0, 0.0, 0.0, 2.0), stump(1, 0.0, 0.3, 0.3)]
    write_npz(tmp_path / "m.npz", trees, [0, 1, 2], [-0.1, -0.2, -0.3], [1, 2, 4], lr=0.5)
    clf = NumpyGradientBoostingClassifier.from_file(tmp_path / "m.npz")
    x = np.array([[-1.0, 1.0, 0, 0]])  # left, left, right
    raw = np.array([-0.1 + 0.5 * 1.0, -0.2 + 0.5 * 0.0, -0.3 + 0.5 * 0.3])
    expected = np.exp(raw) / np.exp(raw).sum()
    np.testing.assert_allclose(clf.predict_proba(x)[0], expected, rtol=1e-12)
    assert list(clf.classes_) == [1, 2, 4]


def test_wrapper_from_directory_prefers_the_numpy_file_and_needs_no_sklearn(tmp_path):
    write_npz(tmp_path / "classifierA_numpy.npz", [stump(0, 3.0, -4.0, 4.0)], [0], [0.0],
              [0, 1], lr=1.0)
    clf = PromptCollapseClassifier.from_directory(tmp_path)  # no pickles in the directory
    p = clf.probability(np.array([2.5, 3.5]), np.array([100.0] * 2), np.zeros(2),
                        np.array([1.2, 1.2]))
    np.testing.assert_allclose(p, expit(np.array([-4.0, 4.0])), rtol=1e-12)


def test_numpy_file_rejects_a_missing_key(tmp_path):
    np.savez(tmp_path / "bad.npz", classes=np.array([0, 1]))
    with pytest.raises(ValueError, match="left"):
        NumpyGradientBoostingClassifier.from_file(tmp_path / "bad.npz")


# ─── shapes: scalars in -> scalars out (nmma-generation injects scalar parameters) ─────────
class ConstModel:
    """predict_proba with a fixed collapse probability, through the real wrapper."""

    classes_ = np.array([0, 1])

    def __init__(self, prob):
        self.prob = prob

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.prob), np.full(len(x), self.prob)])


def real_wrapper(prob):
    return PromptCollapseClassifier(ConstModel(prob), FakeScaler())


def test_wrapper_keeps_the_shape_of_scalar_inputs():
    p = real_wrapper(0.7).probability(3.5, 400.0, 0.0, 1.2)
    assert np.shape(p) == () and float(p) == pytest.approx(0.7)
    assert real_wrapper(0.7).probability(np.array([3.5]), 400.0, 0.0, 1.2).shape == (1,)
    assert real_wrapper(0.7).probability(np.array([3.5, 3.0]), 400.0, 0.0, 1.2).shape == (2,)


@pytest.mark.parametrize("prob", [0.0, 1.0])
def test_scalar_parameters_give_the_same_output_shapes_as_the_default_fit(prob):
    p = {k: float(v[0]) for k, v in system().items()}
    old = BNSEjectaFitting().bns_parameter_conversion(p)
    new = BNSEjectaFitting(
        bns_fit="classifier", pc_classifier=real_wrapper(prob)
    ).bns_parameter_conversion(p)
    assert np.shape(new) == np.shape(old)
    out = KilonovaEjectaFitting(bns_fit="classifier", pc_classifier=real_wrapper(prob))(dict(p))
    for key in ("log10_mej_dyn", "log10_mej_wind", "log10_mej", "log10_E0"):
        assert np.shape(out[key]) == np.shape(KilonovaEjectaFitting()(dict(p))[key]), key
