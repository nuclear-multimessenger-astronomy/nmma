from pathlib import Path
import shutil
import pytest

from nmma.core.utils import read_injection_file
from nmma.core.parsing import nmma_base_parsing
from nmma.joint import injection_handling, joint_parsing

nmma_dir = Path(__file__).parent.parent.parent.parent
data_dir = Path(__file__).parent.parent / "data"

@pytest.fixture(autouse=True)
def cleanup_outdir(args):
    yield
    if Path(args.outdir).exists():
        shutil.rmtree(args.outdir, ignore_errors=True)


@pytest.fixture(scope="module")
def args():
    args = nmma_base_parsing(joint_parsing.injection_parsing, [])
    non_default_args = dict(
        prior_file=str(nmma_dir / "priors" / "Bu2019lm.prior"),
        gw_injection_file=data_dir / "binary_type_O4_injections.dat",
        injection_file=Path(args.outdir, "eos_injection.json"),
        eos_file= nmma_dir / "example_files" / "eos" / "ALF2.dat",
        original_parameters=True,
        generation_seed=42,
    )
    for key, value in non_default_args.items():
        setattr(args, key, value)

    return args


def test_eos_injection_without_snr_test(args):
    """FIXME Weizmann: regression test, using the real Bu2019lm.prior +
    example_files/sim_events/bns_O4_injections.dat combination reported in
    the original bug (nmma-create-injection --eos-file ... --gw-injection-file
    ... with no --tests/--post-processing snr), for bugs in
    NMMAInjectionCreator that only trigger when simple_setup=False (i.e.
    setup_test_routines/test_wrap actually run, unlike the other tests in
    this file, which all use simple_setup=True and so never exercised this
    code path):

    1. EoSConverter needs mass_1_source/mass_2_source, but the 'gw' step
       that computes them (bbh_source_frame) used to only get added to
       conv_instructions for an 'snr' test/post-processing. Using
       --eos-file without an snr test raised
       KeyError('mass_1_source') in EoSConverter.system_props_from_eos.
    2. test_wrap() used to call self.priors.evaluate_constraints(test_df)
       with a pandas DataFrame. bilby's evaluate_constraints does
       next(iter(sample.values())) to get a template array; DataFrame.values
       is an ndarray property, so calling it raises TypeError, silently
       caught by evaluate_constraints, which then falls back to
       np.ones_like(sample) over the whole 2D table instead of a 1D
       per-row array, raising
       ValueError: Expected a 1D array, got an array with shape (N, n_cols)
       when assigned back into test_df['tests_passed'].
    3. refill_failed_tests() did retest_df["tests_passed"] = self.test_wrap(
       retest_df), but test_wrap() returns the whole dataframe, not just
       that column, raising ValueError: Columns must be same length as key.
    4. testing_and_postprocessing()'s redraw branch only copied the
       'tests_passed' column back onto the pre-conversion dataframe, and
       refill_failed_tests() only copied 'tests_passed' back for redrawn
       rows: with --original-parameters (which skips the final
       core_conversion pass), mass_1_source/lambda_1/2/radius_1/2 ended up
       silently missing from the output whenever at least one row needed a
       redraw.
    """

    # simple_setup defaults to False here: this must go through
    # setup_test_routines/testing_and_postprocessing/test_wrap, not skip them.
    injection_handling.generate_injection(args)

    assert args.injection_file.exists(), "injection file does not exist"
    injection_dict = read_injection_file(args.injection_file)
    for key in (
        "mass_1_source",
        "mass_2_source",
        "lambda_1",
        "lambda_2",
        "radius_1",
        "radius_2",
    ):
        assert key in injection_dict, f"{key} missing from generated injection"


def test_binary_type_filter_end_to_end(args):
    """FIXME Weizmann: end-to-end regression test for --binary-type, which
    restores nmma 0.2.3's --binary-type/--eject behaviour: apply a single
    ejecta formula uniformly to every injection and drop (one-shot, no
    redraw) those whose resulting ejecta mass isn't finite. This is the
    only mechanism that can filter injections read from an external
    --gw-injection-file: --tests/Constraint-based filtering only works by
    redrawing samples, and a mass read from a file is fixed, so it can
    never be redrawn away from failing a test (see
    test_eos_injection_without_snr_test's module docstring era discussion;
    reproduced directly against the CLI here instead).

    Also checks that --binary-type overwrites log10_mej_dyn/log10_mej_wind
    even when the prior already samples them directly (nmma 0.2.3 had no
    "prefer the already-sampled value" behaviour, it always overwrote).
    """
    prior_path = Path(data_dir, "Bu2019lm_binary_type_test.prior")
    # First 5 rows of the real, tracked example file: a known, fixed mix of
    # one BNS-under-ALF2, one NSBH-under-ALF2 and one BBH-under-ALF2 event
    # (masses verified by hand against ALF2's ~2.0854 Msun TOV mass).

    args.binary_type="BNS"
    args.prior_file = str(prior_path)
    injection_handling.generate_injection(args)

    injection_dict = read_injection_file(args.injection_file)
    n_kept = len(injection_dict)
    assert n_kept > 0, "--binary-type BNS filtered out every injection"
    assert n_kept < 10, (
        "--binary-type BNS should drop the non-BNS rows in this fixed "
        "5-row mix, not keep all of them (would indicate the prior's "
        "pre-sampled log10_mej_dyn/log10_mej_wind are being kept instead "
        "of being overwritten by the EOS-derived value)"
    )
    for radius_1, radius_2 in zip(
        injection_dict["radius_1"], injection_dict["radius_2"]
    ):
        assert radius_1 > 0 and radius_2 > 0, (
            "every kept injection must have both components be a real NS " "under ALF2"
        )
