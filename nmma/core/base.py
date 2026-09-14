import inspect
import os
import h5py
from ast import literal_eval
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import product

from bilby import run_sampler
from bilby.core.likelihood import Likelihood
from bilby.core.prior import (
    Prior,
    Constraint,
    Interped,
    ConditionalPriorDict,
    PriorDict,
    MultivariateGaussianDist,
    MultivariateGaussian,
)
from bilby.core.result import FileMovedError
from .utils import input_obj_to_str, read_bestfit_from_posterior
from .constants import set_cosmology
from .conversion import cosmology_to_distance
from .parsing import single_messenger_analysis_parsing, nmma_base_parsing


def initialisation_args_from_signature_and_namespace(_callable, namespace, prefixes=[]):
    """Build kwargs for ``_callable`` from ``namespace``, matching each of
    its signature parameters (with or without a default) against a
    same-named (optionally prefixed) attribute on ``namespace``.

    Parameters
    ----------
    _callable: callable
        Whose signature parameters to fill in.
    namespace: argparse.Namespace
        Source of values, e.g. parsed CLI args.
    prefixes: list of str, optional
        Attribute-name prefixes to also try (e.g. ``em_`` to match
        ``namespace.em_tmin`` for a ``tmin`` parameter), tried in order,
        first match wins. The bare (unprefixed) name is always tried too.

    Returns
    -------
    dict
        kwargs for ``_callable``: signature defaults, overridden by
        matching namespace attributes that aren't None.
    """
    # FIX ME: mutable default argument -- `prefixes` is created once at
    # function-definition time and shared across every call that omits
    # it, so this .append('') grows the SAME list on every such call
    # (confirmed: ['', '', ''] after 3 calls). Currently harmless output
    # -wise (the loop below always matches and breaks on the first ''
    # entry), but it's unbounded accumulating state for the life of the
    # process. Should default to `prefixes=None` and do
    # `prefixes = list(prefixes) if prefixes else ['']` instead.
    prefixes.append("")
    signature = inspect.signature(_callable)
    # step 1: get all default kwargs from the signature
    default_kwargs = {
        key: val.default
        for key, val in signature.parameters.items()
        if val.default is not inspect.Parameter.empty
    }

    # step 2: get all available kwargs from the namespace
    for key in signature.parameters.keys():
        # this checks if further parameters from args-Namespace are only used as shorthands in the class definition (e.g. tmin, tmax)
        for prefix in prefixes:
            if hasattr(namespace, prefix + key):
                kwarg = getattr(namespace, prefix + key)
                if kwarg is not None:
                    default_kwargs[key] = kwarg
                break
    return default_kwargs


class NMMALikelihoodMixin:
    """Shared likelihood contract: constraint handling and the
    likelihood/log-likelihood interface. Mixed into ``NMMALikelihood``
    and, separately, the EOS ``JointEoSConstraint`` hierarchy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def priors(self):
        """The sampling priors. Setting this also extracts
        ``constraints`` from it and runs
        ``check_parameter_equivalencies`` on the remaining
        (non-constraint) keys."""
        return self._priors

    @priors.setter
    def priors(self, value):
        self.constraints = value
        sampling_keys = [k for k in value.keys() if k not in self.constraints]
        self.check_parameter_equivalencies(sampling_keys)
        self._priors = value

    @property
    def constraints(self):
        """Dict of ``bilby.core.prior.Constraint`` used by
        ``evaluate_constraints``."""
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        """Normalize ``value`` -- a PriorDict (filtered to its
        Constraint entries), a single Constraint, or a dict of
        Constraints -- into ``self._constraints``."""
        if isinstance(value, PriorDict):
            constr = {k: v for k, v in value.items() if isinstance(v, Constraint)}
        elif isinstance(value, Constraint):
            constr = {value.name: value}
        elif isinstance(value, dict):
            constr = value
            assert all(
                isinstance(v, Constraint) for v in value.values()
            ), "All entries in constraints dict must be of type Constraint"
        # FIX ME: no else/final branch -- assigning `.constraints` to
        # anything other than a PriorDict, Constraint, or dict leaves
        # `constr` unset, so this raises a confusing UnboundLocalError
        # instead of a clear TypeError. Confirmed: `d.constraints = 5`
        # -> "UnboundLocalError: cannot access local variable 'constr'".
        self._constraints = constr

    def evaluate_constraints(self, out_sample):
        """Product of each constraint's ``.prob(value)`` for
        ``out_sample`` -- 0 if any constraint is violated, else the
        joint probability."""
        return np.prod([con.prob(out_sample[k]) for k, con in self.constraints.items()])

    def identity_conversion(self, parameters):
        """No-op parameter conversion; returns ``parameters`` unchanged."""
        return parameters

    def __call__(self, parameters):
        """``exp(self.log_likelihood(parameters))`` -- the plain (not
        log) likelihood."""
        return np.exp(self.log_likelihood(parameters))

    def log_likelihood(self, parameters):
        """Convert ``parameters``, then ``sub_log_likelihood`` if
        constraints and sanity checks pass, else -inf."""
        parameters = self.parameter_conversion(parameters)
        if self.evaluate_constraints(parameters) and self.sanity_checks():
            return self.sub_log_likelihood(parameters)
        else:
            return np.nan_to_num(-np.inf)

    def sanity_checks(self):
        """Always True; subclasses override for messenger-specific
        checks."""
        return True

    def final_diagnostics(self, bestfit_params, args, result=None):
        """Delegate to ``self.sub_model.final_diagnostics``, if it has
        one (e.g. an M-R plot for EOS, a light-curve plot for EM);
        returns None if the sub-model doesn't define it.

        Parameters
        ----------
        bestfit_params: dict
            Best-fit parameters to plot.
        args: argparse.Namespace
            Forwarded to the sub-model's ``final_diagnostics``.
        result: bilby.core.result.Result, optional
            Forwarded to the sub-model's ``final_diagnostics``.

        Returns
        -------
        fig: matplotlib.figure.Figure or None
        """
        try:
            return self.sub_model.final_diagnostics(bestfit_params, args, result)
        except AttributeError:
            pass

    def post_process_bestfit(self, args, result=None):
        """Read the best-fit sample from the saved result on disk
        (via ``read_bestfit_from_posterior(args)``, not from
        ``result``), convert it, and hand it to ``final_diagnostics``."""
        bestfit_params = read_bestfit_from_posterior(args)
        bestfit_params = self.parameter_conversion(bestfit_params)
        return self.final_diagnostics(bestfit_params, args, result)

    def check_parameter_equivalencies(self, parameter_names):
        """Check for equivalent parameters and terminate if found"""
        # FIXME: to be extended
        single_equivalency_groups = [
            ["inclination_EM", "KNtheta", "theta_jn", "cos_theta_jn", "thetaObs"],
        ]
        for group in single_equivalency_groups:
            intersection = set(parameter_names).intersection(set(group))
            if len(intersection) > 1:
                raise ValueError(
                    f"Multiple equivalent parameters found: {intersection}. Please only provide one of these."
                )

        double_equivalency_groups = [
            [
                "redshift",
                "luminosity_distance",
                "Hubble_constant",
            ],  # FIXME: this would be ok if Omega_matter is investigated
            [
                "mass_1",
                "mass_1_source",
                "chirp_mass",
                "mass_ratio",
                "eta",
                "mass_2",
                "mass_2_source",
            ],
        ]
        for group in double_equivalency_groups:
            intersection = set(parameter_names).intersection(set(group))
            if len(intersection) > 2:
                raise ValueError(
                    f"Mutually dependent parameters found: {intersection}. Please only provide up to two of these."
                )


class NMMALikelihood(NMMALikelihoodMixin, Likelihood):
    """The base likelihood object for modular multi-messenger analysis

    Parameters
    ----------
    sub_model: bilby.core.likelihood.Likelihood
        The submodel to be used in each messenger. Must have a log_likelihood method.
    priors: dict
        The analysis priors, required for marginalization in some submodels
    **kwargs
        Currently unused -- accepted for subclass constructor
        compatibility, but not forwarded anywhere.

    """

    def __init__(self, sub_model, priors, **kwargs):
        super().__init__()

        self.sub_model = sub_model
        try:
            self._noise_logl = self.sub_model.noise_log_likelihood()
        except AttributeError:
            self._noise_logl = 0.0
        self.conv_functions = []
        self.priors = priors
        self.setup_submodel_conversion()

    def __repr__(self):
        """``"{class_name} with {sub_model!r}"``."""
        return self.__class__.__name__ + " with " + self.sub_model.__repr__()

    def setup_parameter_conversion(self):
        """Register ``cosmology_to_distance`` into ``conv_functions`` if
        sampling over ``Hubble_constant``. Called from
        ``check_priors_and_likelihood_for_nmma``, after all messenger
        likelihoods are set up."""
        # FUTURE: add more standard conversions here
        if "Hubble_constant" in self.priors:
            self.conv_functions.append(cosmology_to_distance)

    def setup_submodel_conversion(self):
        """No-op by default; subclasses override to register their
        sub_model's own parameter conversion into ``conv_functions``."""
        pass

    def parameter_conversion(self, parameters):
        """Apply ``conv_functions`` in reverse order -- conversions
        added last (e.g. by ``setup_submodel_conversion``, called in
        ``__init__``) run first, ahead of ones added later (e.g. by
        ``setup_parameter_conversion``)."""
        # reverse because "main conversion" are added last
        for conv in reversed(self.conv_functions):
            parameters = conv(parameters)
        return parameters

    def posterior_conversion(self, parameters):
        """Alias for ``parameter_conversion``."""
        return self.parameter_conversion(parameters)

    def sub_log_likelihood(self, parameters):
        """``self.sub_model.log_likelihood(parameters)``, clamped to
        -inf if non-finite."""
        logL_model = self.sub_model.log_likelihood(parameters)
        if not np.isfinite(logL_model):
            return np.nan_to_num(-np.inf)
        return logL_model

    def noise_log_likelihood(self):
        """Cached noise log-likelihood from ``self.sub_model``, or 0. if
        it doesn't define one."""
        return self._noise_logl


class NMMADummyPrior(Prior):
    """A dummy prior that can be read from a prior-file into a prior dict, but is set to be replaced later

    Parameters
    ----------
    setup_props: dict
        Arbitrary properties needed to build the real prior later, via
        ``adjust_priors_for_nmma`` (dispatched on the dict key this is
        stored under, e.g. containing "h5" or "hubble").
    """

    def __init__(self, setup_props):
        super().__init__(name="NMMADummyPrior")
        self.setup_props = setup_props

    @classmethod
    def from_repr(cls, repr_str):
        """Reconstruct from a saved prior file, e.g.
        ``key = nmma.core.base.NMMADummyPrior(setup_props={...})``.

        Parameters
        ----------
        repr_str: str
            The constructor-args portion of the repr.

        Returns
        -------
        NMMADummyPrior
        """
        # FIX ME: bilby's PriorDict.from_dictionary passes the args
        # portion WITH the "setup_props=" prefix still attached (e.g.
        # "setup_props={'Hubble_weight': 'x.dat'}"), but literal_eval
        # can't parse a key=value string -- only bare literals. Confirmed
        # this crashes with a SyntaxError on exactly the round-trip this
        # class exists for (write to a prior file, read it back). Needs
        # either stripping the "setup_props=" prefix first, or using
        # bilby's own Prior._from_repr/_split_repr kwarg parsing instead.
        setup_props = literal_eval(repr_str)
        return cls(setup_props)


def adjust_priors_for_nmma(priors, logger=None):
    """Replace any ``NMMADummyPrior`` placeholders in ``priors`` with the
    real prior their key implies (a multivariate Gaussian from an HDF5
    file for a key containing "h5", or a Hubble-weighted Interped prior
    for a key containing "hubble").

    Parameters
    ----------
    priors: dict | str
        The analysis priors, or a path to a prior file.
    logger: logging.Logger, optional
        The logger to use for logging messages.

    Returns
    -------
    dict
        The adjusted priors dictionary.
    """
    if isinstance(priors, str):
        priors = PriorDict(priors)
    for key, prior in priors.copy().items():
        if not isinstance(prior, NMMADummyPrior):
            continue
        elif "h5" in key:
            priors.pop(key)  # Remove the dummy prior
            if logger:
                logger.info(
                    f"Replacing dummy prior for {key} with multivariate Gaussian prior from HDF5 file"
                )
            priors = h5_to_multivar_prior(prior.setup_props, priors)
        elif "hubble" in key.lower():
            priors.pop(key)  # Remove the dummy prior
            if logger:
                logger.info(
                    f"Replacing dummy prior for {key} with Interped prior from Hubble weighting file"
                )
            priors = adjust_hubble_prior(priors, prior.setup_props, logger)
        # to be extended
    return priors


def adjust_hubble_prior(priors, args, logger=None):
    """Set the cosmology (if sampling over the Hubble constant) and, if
    a precomputed weighting file is given, replace
    ``priors["Hubble_constant"]`` with an Interped prior built from it.

    Parameters
    ----------
    priors: dict
        The analysis priors, updated in place (and returned).
    args: argparse.Namespace | dict
        Must provide "Hubble_weight" (path to a whitespace-delimited
        file, with or without a "Hubble prior_weight" header) and,
        optionally, "Hubble"/"cosmology".
    logger: logging.Logger, optional

    Returns
    -------
    dict
        ``priors``, with "Hubble_constant" replaced if a weighting file
        was given.
    """
    if getattr(args, "Hubble", False) or "Hubble_constant" in priors:
        set_cosmology(getattr(args, "cosmology", None))

    hubble_weight = input_obj_to_str(args, "Hubble_weight")
    if hubble_weight:
        if logger:
            logger.info("Sampling over Hubble constant with pre-calculated prior")
            logger.info("Overwriting any Hubble prior in the prior file")
        try:
            Hubble_prior_data = pd.read_csv(hubble_weight, delimiter=" ", header=0)
            xx = Hubble_prior_data.Hubble.to_numpy()
            yy = Hubble_prior_data.prior_weight.to_numpy()
        except:  # noqa: E722
            xx, yy = np.loadtxt(hubble_weight).T

        Hmin = xx[0]
        Hmax = xx[-1]

        priors["Hubble_constant"] = Interped(
            xx, yy, minimum=Hmin, maximum=Hmax, name="Hubble_constant"
        )
    return priors


def h5_to_multivar_prior(h5_file_path, priors={}):
    """Build a MultivariateGaussian prior over each dataset in an HDF5
    file (mean/covariance estimated from the stored samples), merged
    into ``priors`` (upgraded to a ConditionalPriorDict if it wasn't one
    already).

    Parameters
    ----------
    h5_file_path: str
        Path to an HDF5 file whose top-level datasets are the
        parameters' posterior samples.
    priors: dict, optional
        Priors to merge the new ones into.

    Returns
    -------
    ConditionalPriorDict
    """
    # FIX ME: mutable default argument -- `priors={}` is shared across
    # every call that omits it, and this function mutates it in place
    # (.update below). Confirmed: two calls without passing priors=
    # leak keys from the first call into the second's result. Currently
    # dormant since the only call site (adjust_priors_for_nmma) always
    # passes priors explicitly, but a landmine for any other caller.
    h5_file_path = input_obj_to_str(h5_file_path, "h5 file path")
    with h5py.File(h5_file_path, "r") as f:
        # Load the data from the HDF5 file
        keys = list(f.keys())
        data_array = np.column_stack([f[key][:] for key in keys])
    mean = np.mean(data_array, axis=0)
    # FIX ME: np.cov returns a 0-d scalar (not a (1,1) array) when
    # data_array has only one column (one HDF5 key) -- MultivariateGaussianDist
    # rejects that shape ("List of covariances the wrong shape"), so a
    # single-parameter HDF5 prior file crashes this function entirely.
    cov = np.cov(data_array, rowvar=False)

    eos_dist = MultivariateGaussianDist(keys, mus=[mean], covs=[cov])
    priors.update({key: MultivariateGaussian(eos_dist, key) for key in keys})

    # We need "at least" a conditional Prior Dict, but should not "downgrade" CBC-dicts
    if isinstance(priors, ConditionalPriorDict):
        return priors
    return ConditionalPriorDict(priors)


def check_priors_and_likelihood_for_nmma(priors, likelihood):
    """Final pre-sampling touch-up: move any stray ``Constraint`` priors
    into ``likelihood.constraints``, guard against a ``priors``
    conversion function that produces duplicate-named parameters (see
    FIX ME below), and call ``likelihood.setup_parameter_conversion()``.

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
    likelihood: NMMALikelihood

    Returns
    -------
    priors, likelihood
    """
    # remove constraints from priors and add to likelihood (should have happened already, but just in case)
    constraints = {
        k: priors.pop(k)
        for k in priors.copy().keys()
        if isinstance(priors[k], Constraint)
    }
    likelihood.constraints.update(constraints)

    test_draw = priors.sample(1)
    test_conversion = priors.conversion_function(test_draw)
    if len(set(test_conversion.keys())) != len(test_conversion.keys()):
        # FIX ME: intent looks like it's meant to preserve the original
        # (duplicate-key-producing) conversion_function by moving it into
        # likelihood.conv_functions, while swapping priors.conversion_function
        # to the safe default. But likelihood.priors IS priors (same
        # object), and this reads likelihood.priors.conversion_function
        # AFTER already overwriting it on the line above -- so it appends
        # default_conversion_function again, not the original custom one.
        # Confirmed: the custom conversion_function is silently discarded
        # entirely, and default_conversion_function ends up both as
        # priors.conversion_function and in conv_functions (likely runs
        # twice). Needs the old function captured before the overwrite.
        priors.conversion_function = priors.default_conversion_function
        likelihood.conv_functions.append(likelihood.priors.conversion_function)

    # add final conversions
    likelihood.setup_parameter_conversion()
    return priors, likelihood


def bilby_sampling(likelihood, priors, args, injection_parameters=None, rank=0):
    """Run bilby's run_sampler and post-process the result: convert the
    posterior, save it, plot a corner plot, and (if requested) produce
    best-fit diagnostics.

    Parameters
    ----------
    likelihood: NMMALikelihood
    priors: bilby.core.prior.PriorDict
    args: argparse.Namespace | dict
        If a dict, merged onto ``nmma_base_parsing``'s defaults first.
        Must provide ``sampler``, ``nlive`` (or ``reactive_sampling=True``
        with ``sampler="ultranest"``), ``outdir``, ``label``,
        ``sampling_seed``, ``soft_init``, ``cpus``, ``skip_sampling``,
        and, for post-processing, ``bestfit``/``plot``.
    injection_parameters: dict, optional
        True parameter values, restricted to varying columns before
        being passed to ``plot_corner``.
    rank: int, default 0
        MPI rank; only rank 0 does any post-processing (others return
        None immediately after sampling).

    Returns
    -------
    bilby.core.result.Result or None
        None on non-zero rank.
    """
    if isinstance(args, dict):
        def_args = nmma_base_parsing(single_messenger_analysis_parsing)
        def_args.__dict__.update(args)
        args = def_args
    # fetch the additional sampler kwargs
    sampler_kwargs = getattr(args, "sampler_kwargs", {})
    print("Running with the following additional sampler_kwargs:")
    print(sampler_kwargs)

    # check if it is running with reactive sampler
    nlive = None if getattr(args, "reactive_sampling", False) else args.nlive
    if nlive is None and args.sampler != "ultranest":
        raise ValueError(
            "reactive sampling is only available for ultranest, "
            "please set nlive or use ultranest sampler"
        )

    if args.skip_sampling:
        print("Sampling for 1 iteration and plotting checkpointed results.")
        if args.sampler == "pymultinest":
            sampler_kwargs["max_iter"] = 1
        elif args.sampler == "ultranest":
            sampler_kwargs["niter"] = 1
        elif args.sampler == "dynesty":
            sampler_kwargs["maxiter"] = 1

    result = run_sampler(
        likelihood,
        priors,
        sampler=args.sampler,
        outdir=args.outdir,
        label=args.label,
        nlive=nlive,
        seed=args.sampling_seed,
        soft_init=args.soft_init,
        queue_size=args.cpus,
        check_point_delta_t=3600,
        save=False,
        **sampler_kwargs,
    )

    if rank != 0:
        return

    try:
        result.posterior = likelihood.posterior_conversion(result.posterior)
        result.save_to_file()
        result.save_posterior_samples()
    except FileMovedError:
        # We assume the result was only moved here, no need to save
        result.outdir = args.outdir
        result.label = args.label
        result.parameter_labels_with_unit = None
        result_prior = result.priors.copy()
        for k, v in result_prior.items():
            if k in priors:
                v.latex_label = priors[k].latex_label
                result_prior[k] = v
        result.priors = result_prior
        # result.save_posterior_samples()

    if injection_parameters:
        var_columns = {
            col for col in result.posterior if len(result.posterior[col].unique()) > 1
        }
        injection_parameters = {
            k: v for k, v in injection_parameters.items() if k in var_columns
        }
    try:
        result.plot_corner(injection_parameters, priors)
    except RuntimeError:
        result.parameter_labels_with_unit = None
        for k, v in priors.copy().items():
            v.latex_label = None
            priors[k] = v
        result.priors = priors
        result.plot_corner(injection_parameters, priors)

    if args.bestfit or args.plot:
        result.posterior = likelihood.posterior_conversion(result.posterior)
        likelihood.post_process_bestfit(args, result)
    return result


def multi_analysis_loop(args, analysis_setup):
    """Run one or more analyses, expanding ``--multi``/``--matrix``
    sweep specs into separate runs first.

    Parameters
    ----------
    args: argparse.Namespace
        May provide ``multi`` (sweep one parameter across a list of
        values, or run several named variants each with their own
        parameter changes -- see the FIX ME below for the
        single-named-run edge case) or ``matrix`` (cartesian product of
        parameter lists, each combination its own run).
    analysis_setup: callable
        Builds one run's priors, likelihood, and injection parameters
        from its (possibly modified) args.

    Returns
    -------
    bilby.core.result.Result
        From the last sub-run.
    """
    USE_MPI = False
    rank = 0
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        if MPI.COMM_WORLD.Get_size() > 1:
            USE_MPI = True
    except:  # noqa: E722
        pass

    if rank != 0 and not getattr(args, "verbose", False):
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

    if getattr(args, "multi", None):
        sub_runs = []
        # FIX ME: `len(args.multi) == 1` is meant to distinguish "sweep
        # one parameter across a list of values" (this branch) from
        # "named sub-runs, each with their own changes" (the `else`
        # below) -- but it can't actually tell these apart, since a
        # single NAMED run (e.g. --multi '{"myrun": {"mass_1": 5.0}}',
        # a reasonable "run one variant" use case) also has length 1.
        # Confirmed: that case gets misrouted here, where `vals` is
        # actually the changes dict {"mass_1": 5.0} -- enumerate(vals)
        # iterates its KEYS, so the intended change is silently dropped
        # and a bogus attribute (run_args.myrun = "mass_1") gets set
        # instead, with no error. Needs to check whether the single value
        # is a dict (named-run form) vs a list (sweep form), not just
        # count keys.
        if len(args.multi) == 1:
            arg, vals = list(args.multi.items())[0]
            for i, val in enumerate(vals):
                run_args = deepcopy(args)
                setattr(run_args, arg, val)
                setattr(run_args, "label", f"{args.label}_{i}")
                sub_runs.append(run_args)
        else:
            for run_name, changes in args.multi.items():
                run_args = deepcopy(args)
                setattr(run_args, "label", f"{args.label}_{run_name}")
                for key, value in changes.items():
                    if key not in args:
                        raise KeyError(f"{key} not a known argument... please remove")
                    setattr(run_args, key, value)
                sub_runs.append(run_args)
    elif getattr(args, "matrix", None):
        sub_runs = []
        keys = args.matrix.keys()
        vals = args.matrix.values()
        for arg_variation in product(*vals):
            run_args = deepcopy(args)
            run_name = args.label
            for i, var in enumerate(arg_variation):
                rep = f"_{var}"
                if len(rep) > 20:
                    # FIX ME: keys/vals are dict_keys/dict_values views
                    # (from args.matrix.keys()/.values() above), which
                    # aren't subscriptable -- confirmed this raises
                    # "TypeError: 'dict_keys' object is not subscriptable"
                    # whenever a variation value's label exceeds 20 chars.
                    # This fallback (meant to shorten an over-long label)
                    # is dead: it always crashes instead of running.
                    # Needs `keys = list(args.matrix.keys())` etc. above.
                    key = keys[i]
                    var_idx = vals[i].index(var)
                    rep = f"_{key}_{var_idx}"
                run_name += rep
            setattr(run_args, "label", run_name)
            for key, val in zip(keys, arg_variation):
                if key not in args:
                    raise KeyError(f"{key} not a known argument... please remove")
                setattr(run_args, key, val)
            sub_runs.append(run_args)

    else:
        sub_runs = [args]
    for run_args in sub_runs:
        priors, likelihood, injection_parameters = analysis_setup(run_args)
        priors, likelihood = check_priors_and_likelihood_for_nmma(priors, likelihood)
        if USE_MPI and run_args.sampler == "dynesty":
            from .mpi_setup import pbilby_sampling

            run_function = pbilby_sampling
        else:
            run_function = bilby_sampling
        out = run_function(likelihood, priors, run_args, injection_parameters, rank)
    return out
