import json
import shutil
from argparse import Namespace
from ast import literal_eval
from pathlib import Path

import matplotlib
import numpy as np
from bilby.core.prior import PriorDict, WeightedCategorical
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp
from scipy.stats import norm

from ..core.base import NMMALikelihood
from ..core.plotting_utils import fading_cmap, fig_setup, setup_multi_axes
from ..core.utils import nan_level
from .eos_processing import EoSConverter, TabulatedEoSGenerator

nmma_colors = fig_setup()


def setup_tabulated_eos_priors(args, priors, logger=None, weights=None):
    """Add an "EOS" prior to ``priors``, in place, for sampling over a set
    of tabulated EOS files.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI args. ``Neos`` sets how many EOS to sample over
        directly; otherwise it's taken from the length of ``eos_weight``
        (if given) or the file count in ``eos_data`` (``Neos=0`` counts
        as unset). ``eos_weight`` also sets the prior's per-EOS weights,
        otherwise they're uniform.
    priors: bilby.core.prior.PriorDict
        Prior dict to add the "EOS" entry to.
    logger: logging.Logger, optional
        If given, logs that tabulated-EOS sampling is being used.
    weights: array_like, optional
        Per-EOS weights, if not given inferred from args.

    Returns
    -------
    priors: bilby.core.prior.PriorDict
        The same dict, with "EOS" added.
    """
    if logger:
        logger.info("Sampling over precomputed EOSs")
    if weights is None and getattr(args, "eos_weight", None):
        weights = np.loadtxt(args.eos_weight)
    if getattr(args, "Neos", False):
        n_eos = args.Neos
    elif weights is not None:
        n_eos = len(weights)
    else:
        n_eos = len(list(Path(args.eos_data).iterdir()))
    priors["EOS"] = WeightedCategorical(n_eos, weights, name="EOS")
    return priors


def setup_eos_kwargs(data_dump, args, logger):
    """Build the kwargs for ``EquationofStateLikelihood`` when the EOS is
    generated on the fly (the "emulated" case).

    Parameters
    ----------
    data_dump: dict
        Generation-time data dump; must contain ``"eos_constraint_dict"``.
    args: argparse.Namespace
        Parsed CLI args, passed to ``EoSConverter(args)``.
    logger: logging.Logger
        Unused; accepted for interface consistency with the other
        messengers' ``setup_*_kwargs``.

    Returns
    -------
    dict
        ``{"constraint_dict": ..., "eos_converter": EoSConverter(...)}``.
    """
    # default_eos_kwargs = initialisation_args_from_signature_and_namespace(EquationofStateLikelihood, args)
    # eos_kwargs = default_eos_kwargs | dict(
    eos_kwargs = dict(
        constraint_dict=data_dump["eos_constraint_dict"],
        eos_converter=EoSConverter(args),
        # crust_path=args.eos_crust_file
    )
    return eos_kwargs


def tabulated_eos_setup(args):
    """Build priors and an EquationofStateLikelihood for a tabulated-EOS
    run.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI args (``eos_data``, ``eos_weight``, ``Neos``, plus any constraint
        args for ``compose_eos_constraints``). ``Neos`` is resolved and written
        back onto ``args`` in place.

    Returns
    -------
    priors: bilby.core.prior.PriorDict
    eos_likelihood: EquationofStateLikelihood
    injection_parameters: None
        Always None; EOS has no injection parameters of its own.
    """
    priors = PriorDict()
    priors = setup_tabulated_eos_priors(args, priors)
    args.Neos = priors["EOS"].ncategories
    eos_converter = EoSConverter(args, macro_only=True)
    eos_likelihood_kwargs = {
        "constraint_dict": compose_eos_constraints(args),
        "eos_converter": eos_converter,
    }
    eos_likelihood = EquationofStateLikelihood(priors, **eos_likelihood_kwargs)
    return priors, eos_likelihood, None


class EquationofStateLikelihood(NMMALikelihood):
    """Likelihood for sampling over an EOS against a set of astrophysical
    constraints, via a JointEoSConstraint sub-model.

    Parameters
    ----------
    priors: bilby.core.prior.PriorDict
        Sampling priors.
    constraint_dict: dict
        Constraints to apply, as built by ``compose_eos_constraints``.
    eos_converter: EoSConverter
        Converts sampled EOS parameters to macroscopic mass/radius/tidal
        properties.
    """

    def __init__(self, priors, constraint_dict, eos_converter, **kwargs):
        constraint = JointEoSConstraint(constraint_dict, eos_converter=eos_converter)
        # TODO: to be extended for more complex likelihood expressions
        super().__init__(constraint, priors, **kwargs)

    def setup_submodel_conversion(self):
        """Register the EOS parameter conversion so it runs on every
        ``self.parameter_conversion`` call."""
        self.conv_functions.append(self.sub_model.parameter_conversion)

    def final_diagnostics(self, bestfit_params, args, result=None, fig=None):
        """Draw and save the best-fit EOS's mass-radius diagnostic plot, with
        constraint overlays and, if ``result`` is given, posterior credible
        bands and the injected EOS's curve.

        Parameters
        ----------
        bestfit_params: dict
            Best-fit sample to convert and plot.
        args: argparse.Namespace
            Must have ``label`` and ``outdir`` (used for the saved filename).
            ``fig`` — an existing Figure to overlay onto (skipping
            constraints already labelled on it) — is optional, defaulting
            to None (start fresh).
        result: bilby.core.result.Result, optional
            If given, also draws 50%/90% credible bands from
            ``result.posterior`` and, if set, ``result.injection_parameters``.
        fig: unused
            Ignored — pass an existing figure via ``args.fig`` instead.

        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        matplotlib.rcParams.update({"font.size": 16, "font.family": "serif"})
        # matplotlib.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        bestfit_params = self.parameter_conversion(bestfit_params)
        macro_parameters = self.sub_model.eos_converter.macro_parameters
        radii, masses = macro_parameters["radii"], macro_parameters["masses"]
        fig = getattr(args, "fig", None)
        if fig is None:
            x_lim = (min(np.min(radii) - 0.3, 9.3), max(np.max(radii) + 0.3, 14.5))
            y_lim = (masses[0], masses[-1] + 0.1)
            fig, ax = setup_multi_axes(1, figsize=(8, 8))
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_xlabel(r"Radius\,[km]")
            ax.set_ylabel(r"Mass\,[M$_\odot$]")

            for constraint in self.sub_model.constraints:
                ax = constraint.plot(ax=ax)
        else:
            ax = fig.axes[0]
            xlow, xhigh = ax.get_xlim()
            ylow, yhigh = ax.get_ylim()
            ax.set_xlim(min(np.min(radii) - 0.3, xlow), max(np.max(radii) + 0.3, xhigh))
            ax.set_ylim(min(masses[0], ylow), max(masses[-1] + 0.1, yhigh))

            const_labels = [t.get_text() for t in ax.texts]
            for constraint in self.sub_model.constraints:
                if constraint.name in const_labels:
                    continue
                color = ax._get_lines.get_next_color()
                ax = constraint.plot(ax=ax, color=color)
            fig.legends.clear()  ## remove old legend to avoid duplicates

        line_kwargs = dict(label=args.label, linewidth=5, zorder=10)
        line = ax.plot(radii, masses, color=next(nmma_colors), **line_kwargs)

        if result is not None:
            cmap = fading_cmap(line[0].get_color())
            posterior = self.parameter_conversion(result.posterior)
            generator = self.sub_model.eos_converter.generator
            post_curves, post_m_tov = generator.curves(posterior)
            mass_range = np.linspace(1.0, np.max(post_m_tov), 151)
            # radii along the mass range, NaN outside each curve
            show_radii = generator.mass_grid.interpolate(
                mass_range, post_m_tov, post_curves
            )[0].T

            for i, level in enumerate([0.5, 0.9]):
                bounds = np.array([nan_level(radii, level) for radii in show_radii])
                # color=cmap(1-0.4*level)
                color = cmap(0.75)
                if i == 0:
                    kwargs = dict(color=color, zorder=1)
                else:
                    kwargs = dict(
                        facecolor="none", edgecolor=color, zorder=0, hatch="//"
                    )
                ax.fill_betweenx(mass_range, bounds[:, 0], bounds[:, 1], **kwargs)

            if result.injection_parameters is not None:
                inj_radii, _, inj_m_tov = generator(result.injection_parameters)
                ax.plot(
                    inj_radii[0],
                    generator.mass_grid.masses(inj_m_tov[0]),
                    label="Injection",
                    color="black",
                    linestyle="dashed",
                    linewidth=3,
                    zorder=10,
                )

        fig.legend(
            loc="upper center", ncols=2, bbox_to_anchor=(0.5, 0.00), handlelength=2
        )
        fig.savefig(
            Path(args.outdir, f"{args.label}_mr_curve.png"), bbox_inches="tight"
        )
        return fig


def compose_eos_constraints(
    args, constraint_kinds=["lower_mtov", "upper_mtov", "mass_radius"]
):
    """Build the full EOS constraint dict from CLI args, merged on top of
    any existing ``args.eos_constraint_json`` file, and persist the result
    back to that file.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed CLI args; see ``read_constraint_from_args`` for how each
        kind is read. ``eos_constraint_json`` is optional — without a
        readable/writable path, constraints are just read from args with
        no persistence.
    constraint_kinds: list of str, default ["lower_mtov", "upper_mtov", "mass_radius"]
        Which constraint kinds to read.

    Returns
    -------
    dict
        ``{constraint_kind: {label: {property: value, ...}, ...}, ...}``,
        merged per-label with any constraints already on file.
    """
    try:
        with open(args.eos_constraint_json, "r") as f:
            constraint_dict = json.load(f)
    except:
        constraint_dict = {}

    for constraint_kind in constraint_kinds:
        sub_dict = read_constraint_from_args(args, constraint_kind)
        if sub_dict is None:
            continue
        try:
            constraint_dict[constraint_kind].update(sub_dict)
        except KeyError:
            constraint_dict[constraint_kind] = sub_dict
        except AttributeError:
            constraint_dict[constraint_kind] = sub_dict

    try:
        with open(args.eos_constraint_json, "w") as f:
            json.dump(constraint_dict, f, indent=4)
    except:
        pass
    return constraint_dict


def read_constraint_from_args(args, constraint_kind):
    """Assemble one kind of EOS constraint into the dict format expected by
    ``JointEoSConstraint.initialise_from_dict``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments, as produced during the generation
        step (e.g. by ``nmma-generation``).
    constraint_kind : str
        Which kind of constraint to read, e.g. ``"lower_mtov"``,
        ``"upper_mtov"`` or ``"mass_radius"``.

    Returns
    -------
    dict or None
        ``{label: {property: value, ...}, ...}`` for every named
        constraint of this kind, or ``None`` if none were specified.
    """

    ##preferred: Have the dict with the subconstraints already set up
    prep_dict = getattr(args, constraint_kind, None)
    if prep_dict:
        if isinstance(prep_dict, str):
            prep_dict = literal_eval(prep_dict)
        return prep_dict

    ###otherwise try to construct it:
    ### read in provided attributes like constraint_kind-name, -mass, -error, -arxiv
    constraint_props = {
        key.removeprefix(constraint_kind + "_"): getattr(args, key)  ##cut identifier
        for key in dir(args)  ## search args for attrs
        if key.startswith(constraint_kind + "_")  ## related with kind
    }

    new_constraints = constraint_props.pop("name", None)
    if new_constraints:  ## there needs to be a unique label
        for k in list(constraint_props.keys()):
            v = constraint_props[k]
            if v is None:
                constraint_props.pop(k)
            elif len(v) != len(new_constraints):
                raise ValueError(
                    f"For {constraint_kind}, the number of entries for {k} ({len(v)}) does not match the number of names ({len(new_constraints)})!"
                )

        ext_dict = {}
        ### iterate through constrs.
        for i, name in enumerate(new_constraints):
            ext_dict[name] = {k: v[i] for k, v in constraint_props.items()}
        return ext_dict
    else:
        return None


class JointEoSConstraint:
    """Combines one or more EOS constraints, evaluating their combined
    log-likelihood against the curves of the last batch of EOSs converted by
    ``eos_converter``.

    ``log_likelihood(parameters)`` is only meaningful directly after
    ``parameter_conversion(parameters)`` (or another conversion of the same
    parameters by ``eos_converter``), as the constraints read the curves from the
    converter's state (``radii``, ``m_tov`` and ``batch_shape``).

    Parameters
    ----------
    *constraints: EoSConstraint | JointEoSConstraint | dict
        Constraints to combine. An ``EoSConstraint`` is used as-is, a
        ``JointEoSConstraint`` is flattened (its own constraints merged
        in), and a dict is expanded via ``initialise_from_dict``. Every
        constraint is attached to ``eos_converter``.
    eos_converter: EoSConverter
        Converts sampled EOS parameters to macroscopic properties and holds
        the curves the constraints are evaluated on.
    """

    def __init__(self, *constraints, eos_converter):
        self.constraints = self.initialise_constraints(constraints)
        self.eos_converter = eos_converter
        for constraint in self.constraints:
            constraint.eos_converter = eos_converter

    def __repr__(self):
        """Combined repr of the constraints, joined with "and"/commas."""
        if len(self.constraints) == 1:
            return f"{self.constraints[0].__repr__()}"
        elif len(self.constraints) == 2:
            return (
                f"{self.constraints[0].__repr__()} and {self.constraints[1].__repr__()}"
            )
        else:
            return f"{self.__class__.__name__} of {', '.join([cons.__repr__() for cons in self.constraints[:-1]])} and {self.constraints[-1].__repr__()}"

    def initialise_constraints(self, constraint_tuple):
        """Flatten the constructor's ``*constraints`` into a plain list of
        ``EoSConstraint`` instances, expanding any dicts along the way."""
        constraint_list = []
        for constraint in constraint_tuple:
            if isinstance(constraint, EoSConstraint):
                constraint_list.append(constraint)
            elif isinstance(constraint, JointEoSConstraint):
                constraint_list.extend(constraint.constraints)
            elif isinstance(constraint, dict):
                constraint_list.extend(self.initialise_from_dict(constraint))
        return constraint_list

    def initialise_from_dict(self, constraint_dict):
        """Build EoSConstraint instances from a
        ``{constraint_kind: {label: {property: value, ...}}}`` dict, as
        produced by ``compose_eos_constraints``.

        Parameters
        ----------
        constraint_dict: dict
            Keyed by "lower_mtov", "upper_mtov", or "mass_radius".

        Returns
        -------
        list of EoSConstraint
        """
        constraint_list = []
        for constraint_type, constr_class in zip(
            ["lower_mtov", "upper_mtov"], [LowerMTOVConstraint, UpperMTOVConstraint]
        ):
            for label, constraint in constraint_dict.get(constraint_type, {}).items():
                constraint_list.append(
                    constr_class(
                        measured_mass=constraint["mass"],
                        measure_error=constraint.get("error", 0.0),
                        name=label,
                        arxiv_ref=constraint.get("arxiv", None),
                        plot_kwargs=constraint.get("plot_kwargs", None),
                    )
                )
        for label, constraint in constraint_dict.get("mass_radius", {}).items():
            constraint_list.append(
                MassRadiusConstraint(
                    file_path=constraint.get(
                        "posterior", constraint.get("file_path", None)
                    ),
                    name=label,
                    arxiv_ref=constraint.get("arxiv", None),
                    plot_kwargs=constraint.get("plot_kwargs", None),
                )
            )
        return constraint_list

    def parameter_conversion(self, parameters):
        """Forward to ``self.eos_converter.parameter_conversion``."""
        return self.eos_converter.parameter_conversion(parameters)

    def log_likelihood(self, parameters):
        """Sum of the constraints' log-likelihoods for the EOSs currently held by
        ``self.eos_converter``, of shape ``eos_converter.batch_shape``."""
        return sum(con.log_likelihood(parameters) for con in self.constraints)

    def tabulated_eos_parameters(self, parameters=None):
        """``{"EOS": indices}`` of the tabulated EOSs to evaluate. ``parameters``
        may be such a dict, a count N (int, float or str) for the first N EOSs,
        or None (or 0) for all EOSs of the generator."""
        generator = self.eos_converter.generator
        if not isinstance(generator, TabulatedEoSGenerator):
            raise TypeError(
                "Weighting tabulated EOSs requires a TabulatedEoSGenerator, "
                f"got {type(generator).__name__}"
            )
        if isinstance(parameters, (str, int, float, np.integer)):
            parameters = int(parameters) or None
        if parameters is None:
            parameters = generator.Neos
        if not isinstance(parameters, dict):
            parameters = {"EOS": np.arange(parameters)}
        return parameters

    def log_weights(self, parameters=None, chunk_size=10000):
        """Combined log-likelihood of each tabulated EOS in ``parameters`` (see
        ``tabulated_eos_parameters``), evaluated in batches of ``chunk_size``
        on the resampled curves of the converter's generator.

        Returns
        -------
        np.ndarray
            Shape (n_eos,), in the order of ``parameters["EOS"]``.
        """
        parameters = self.tabulated_eos_parameters(parameters)
        eos_indices = np.ravel(parameters["EOS"]).astype(int)
        log_weights = np.empty(len(eos_indices))
        for start in range(0, len(eos_indices), chunk_size):
            chunk = {"EOS": eos_indices[start : start + chunk_size]}
            self.eos_converter.compute_macro_parameters(chunk)
            log_weights[start : start + chunk_size] = self.log_likelihood(chunk)
        return log_weights

    def write_sorted_eos(self, out_dir, eos_indices, weights):
        """Copy the tabulated EOS files ``eos_indices`` to ``out_dir`` as
        ``1.dat``, ``2.dat``, ... in ascending order of ``weights``.

        Returns
        -------
        sort_idcs: np.ndarray
            Indices into ``weights`` (and ``eos_indices``) in the written order.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        eos_files = self.eos_converter.generator.eos_files
        eos_indices = np.ravel(eos_indices).astype(int)
        sort_idcs = np.argsort(weights)
        for rank, idx in enumerate(sort_idcs):
            shutil.copy(eos_files[eos_indices[idx]], out_dir / f"{rank + 1}.dat")
        return sort_idcs

    def tabulate_weighted_eos(
        self, parameters, outdir, weight_path=None, normalise=True, chunk_size=10000
    ):
        """Evaluate this constraint on a set of tabulated EOSs and write them
        out sorted by weight, for use as a WeightedCategorical EOS prior.

        If ``outdir`` already contains a ``sorted`` directory and
        ``eos_weights.dat``, that cached result is returned as-is.

        Parameters
        ----------
        parameters: dict | str | int | float | None
            EOSs to evaluate, see ``tabulated_eos_parameters``.
        outdir: str
            Directory to write the sorted EOSs and weights into.
        weight_path: str, optional
            Path to prior weights (one per EOS of the generator) to combine
            with the new constraint weights; ignored if the file is missing.
        normalise: bool, default True
            Whether to normalise the resulting weights to sum to 1.
        chunk_size: int, default 10000
            Number of EOSs evaluated per batch.

        Returns
        -------
        weight_path: Path
            Path to the written weights file, ``outdir/eos_weights.dat``.
        file_path: Path
            Directory of sorted EOS files, named ``1.dat``, ``2.dat``, ...
            in ascending weight.
        n_eos: int
            Number of EOSs written.
        """
        file_path = Path(outdir, "sorted")
        new_weight_path = Path(outdir, "eos_weights.dat")
        if file_path.is_dir() and new_weight_path.is_file():
            return new_weight_path, file_path, len(list(file_path.iterdir()))

        parameters = self.tabulated_eos_parameters(parameters)
        eos_indices = np.ravel(parameters["EOS"]).astype(int)
        log_weights = self.log_weights(parameters, chunk_size)

        if isinstance(weight_path, (str, Path)):
            try:
                previous_weights = np.atleast_1d(np.loadtxt(weight_path))
                log_weights += np.log(previous_weights[eos_indices])
            except FileNotFoundError:
                pass

        if normalise:
            log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)

        sort_idcs = self.write_sorted_eos(file_path, eos_indices, weights)
        np.savetxt(new_weight_path, weights[sort_idcs])
        return new_weight_path, file_path, len(weights)


class EoSConstraint:
    """Base class for a single EOS constraint (mass-only or mass-radius).

    Parameters
    ----------
    name: str, optional
        Identifier of the measurement. If omitted, ``self.name`` falls
        back to the class name — but ``repr()`` doesn't: a known bug
        leaves ``self.base_repr`` as literal ``None`` in that case, so
        ``repr()`` renders as ``"None "`` instead.
    arxiv_ref: str, optional
        Identifier of a relevant source, appended to ``repr()`` if set.
    plot_kwargs: dict, optional
        Overrides for the default line/contour style used by ``plot``.
    """

    def __init__(self, name=None, arxiv_ref=None, plot_kwargs=None):
        self.repr_add = ""
        self.type = "macro"
        if name is None:
            self.name = self.__class__.__name__
            self.base_repr = self.name
        else:
            self.name = name
            self.base_repr = f"{self.__class__.__name__} based on {name}"
        self.arxiv_ref = arxiv_ref
        self.plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
        # set by JointEoSConstraint
        self.eos_converter = None

    def reshape_to_batch(self, values: np.ndarray):
        """``values`` of shape (n_eos,) in ``batch_shape``, a numpy scalar for a
        single sample."""
        return np.reshape(values, self.eos_converter.batch_shape)[()]

    def log_likelihood(self, parameters):
        """Log-likelihood of the EOSs currently held by ``self.eos_converter``,
        of shape ``eos_converter.batch_shape``."""
        raise NotImplementedError

    def __repr__(self):
        """``"{class_name} based on {name}"``, plus ``" (see arxiv:...)"``
        if set. Broken (renders ``"None "``) when ``name`` was omitted."""
        out = f"{self.base_repr} {self.repr_add}"
        if self.arxiv_ref:
            out = f"{out} (see arxiv:{self.arxiv_ref})"
        return out


class MassConstraint(EoSConstraint):
    """Base class for a Gaussian mass-bound constraint (see
    LowerMTOVConstraint/UpperMTOVConstraint for the actual bounds).

    Parameters
    ----------
    measured_mass: float
        Observed mass (in solar masses).
    measure_error: float
        1-sigma uncertainty of the measurement.
    name: str, optional
        Identifier of the measurement.
    arxiv_ref: str, optional
        Identifier of a relevant source.
    plot_kwargs: dict, optional
        Overrides for the default line style used by ``plot``.
    lognorm_method: callable
        ``norm.logcdf`` or ``norm.logsf``, set by subclasses to pick the
        bound's direction.
    """

    def __init__(
        self,
        measured_mass,
        measure_error,
        name=None,
        arxiv_ref=None,
        plot_kwargs=None,
        lognorm_method=None,
    ):
        super().__init__(name, arxiv_ref, plot_kwargs)
        self.mass = measured_mass
        self.error = measure_error
        self.repr_add = f"of {measured_mass}+-{measure_error} M_sun"
        self.lognorm_method = lognorm_method
        self.linestyle = "--"

    def __repr__(self):
        """``"{class_name} of {mass}+-{error} M_sun based on {name}"``."""
        out = f"{self.__class__.__name__} of {self.mass}+-{self.error} M_sun"
        if self.name != "Mass Constraint":
            out = f"{out} based on {self.name}"
        return out

    def log_likelihood(self, parameters):
        """``self.lognorm_method(m_tov, loc=self.mass, scale=self.error)``
        for the TOV mass (last grid mass) of each current EOS curve."""
        log_l = self.lognorm_method(
            self.eos_converter.m_tov, loc=self.mass, scale=self.error
        )
        return self.reshape_to_batch(log_l)

    def plot(self, ax, **kwargs):
        """Draw this constraint as a horizontal line at ``self.mass``, spanning
        the current x-limits of a mass-radius plot, labelled with ``self.name``.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Mass-radius axes to draw on (mass on y, radius on x).
        **kwargs
            Overrides for ``self.plot_kwargs`` (e.g. color, linestyle,
            linewidth), passed to ``ax.hlines``.

        Returns
        -------
        ax: matplotlib.axes.Axes
            The same axes, for chaining.
        """
        x_lim = ax.get_xlim()
        plot_kwargs = self.plot_kwargs | kwargs
        if "color" not in plot_kwargs:
            plot_kwargs["color"] = ax._get_lines.get_next_color()
        if "linestyle" not in plot_kwargs:
            plot_kwargs["linestyle"] = self.linestyle
        if "linewidth" not in plot_kwargs:
            plot_kwargs["linewidth"] = 2.5

        ax.hlines(self.mass, *x_lim, zorder=3, **plot_kwargs)
        ax.text(
            x_lim[0] + 0.05 * (x_lim[1] - x_lim[0]),
            self.mass,
            self.name,
            color=plot_kwargs["color"],
            fontsize=ax.xaxis.label.get_size(),
            va="center",
            zorder=20,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5),
        )
        # cmap = fading_cmap(plot_kwargs['color'])
        # levels = [0.95, 0.68]
        # for i, level in enumerate(levels):
        #     ax.fill_between(x_lim, self.mass - (i+1)*self.error, self.mass + (i+1)*self.error, color=cmap(0.8*level), zorder=2-i)
        return ax


class LowerMTOVConstraint(MassConstraint):
    """Constraint that an EOS supports at least a certain TOV mass.

    Parameters
    ----------
    measured_mass: float
        Observed mass (in solar masses).
    measure_error: float
        1-sigma uncertainty of the measurement.
    name: str, optional
        Identifier of the measurement.
    arxiv_ref: str, optional
        Identifier of a relevant source.
    plot_kwargs: dict, optional
        Overrides for the default line style used by ``plot``.
    """

    def __init__(
        self, measured_mass, measure_error, name=None, arxiv_ref=None, plot_kwargs=None
    ):
        super().__init__(
            measured_mass,
            measure_error,
            name,
            arxiv_ref,
            plot_kwargs,
            lognorm_method=norm.logcdf,
        )


class UpperMTOVConstraint(MassConstraint):
    """Constraint that an EOS supports at most a certain TOV mass
    (within Gaussian uncertainty).

    Parameters
    ----------
    measured_mass: float
        Observed mass (in solar masses).
    measure_error: float
        1-sigma uncertainty of the measurement.
    name: str, optional
        Identifier of the measurement.
    arxiv_ref: str, optional
        Identifier of a relevant source.
    plot_kwargs: dict, optional
        Overrides for the default line style used by ``plot``.
    """

    def __init__(
        self, measured_mass, measure_error, name=None, arxiv_ref=None, plot_kwargs=None
    ):
        super().__init__(
            measured_mass,
            measure_error,
            name,
            arxiv_ref,
            plot_kwargs,
            lognorm_method=norm.logsf,
        )
        self.linestyle = ":"


class MassRadiusConstraint(EoSConstraint):
    """Constraint that an EOS adheres to a certain mass-radius region.

    Parameters
    ----------
    mass_array: np.array, optional
        Array (or similar) with mass posterior of M-R measurement, must be
        specified along an equal-length radius_array.
    radius_array: np.array, optional
        Array (or similar) with radius posterior of M-R measurement, must
        be specified along an equal-length mass_array.
    weights: np.array, optional
        Weights for the M-R samples, aligned with mass_array/radius_array.
    file_path: str, optional
        Path to a data file containing radius and mass posteriors. If
        provided, mass_array and radius_array are ignored.
    name: str, optional
        Identifier of the measurement.
    arxiv_ref: str, optional
        Identifier of a relevant source.
    plot_kwargs: dict, optional
        Overrides for the default line style used by ``plot``.
    """

    def __init__(
        self,
        mass_array=None,
        radius_array=None,
        weights=None,
        file_path=None,
        name=None,
        arxiv_ref=None,
        plot_kwargs=None,
    ):
        super().__init__(name, arxiv_ref, plot_kwargs)
        if file_path:
            mass_array, radius_array, weights = self.read_data(file_path)
        elif mass_array is None or radius_array is None:
            raise ValueError(
                "Must provide data for masses and radii as arrays or file from which to load"
            )
        self.set_grid(mass_array, radius_array, weights)
        self.test_masses = np.linspace(1.2, 2.5, 151)

    def read_data(self, file_path):
        """Load an M-R(-weight) file, auto-detecting orientation and which
        column is mass vs. radius (masses are assumed <= 3, radii > 3).

        Parameters
        ----------
        file_path: str
            Whitespace-delimited file with 2 columns (mass, radius) or 3
            (mass, radius, weight), in either row or column orientation.

        Returns
        -------
        masses: np.ndarray
        radius: np.ndarray
        weights: np.ndarray or None
            None if the file had only 2 columns.
        """
        data = np.loadtxt(file_path, unpack=True)
        if data.shape[0] not in [2, 3]:
            data = data.T
        if data.shape[0] not in [2, 3]:
            raise ValueError(
                "Data file must have two or three columns for mass, radius (, weights)"
            )
        try:
            # if three columns it includes weights
            data_1, data_2, weights = data
        except ValueError:
            data_1, data_2 = data
            weights = None

        if (data_1 <= 3.0).any():
            # we assume radii in km and mass in solar masses. 3 km is an arbitrary-ish limit for neutron stars
            masses = data_1
            radius = data_2

        else:
            radius = data_1
            masses = data_2

        if not ((masses > 0).all() and (masses < 5).all() and (radius > 3).all()):
            min_mass = np.min(masses)
            max_mass = np.max(masses)
            median_mass = np.median(masses)
            min_radius = np.min(radius)
            max_radius = np.max(radius)
            median_radius = np.median(radius)
            raise ValueError(
                "Failed to properly identify mass and radius. Masses should be in solar masses and radii in km, "
                "but the identified values seem to fall outside reasonable ranges. "
                f"Identified mass range: {min_mass:.2f} - {max_mass:.2f} (median: {median_mass:.2f}), "
                f"Identified radius range: {min_radius:.2f} - {max_radius:.2f} (median: {median_radius:.2f}). "
                "Please check the input data file format."
            )

        return masses, radius, weights

    def set_grid(self, masses, radii, weights, mass_step=0.01, radius_step=0.03):
        """Bin M-R(-weight) samples into a smoothed 2D histogram, stored as
        ``self.histogram`` (with edges ``self.rad_edges``/``self.mass_edges``)
        for use as an approximate mass-radius pdf.

        Trims the 0.1%/99.9% tails of each axis and pads the range by 5%
        before binning. When combining multiple measurements, use the same
        step sizes for all of them.

        Parameters
        ----------
        masses: np.array
            Mass posterior samples of the M-R measurement.
        radii: np.array
            Radius posterior samples, same length as ``masses``.
        weights: np.array, optional
            Sample weights, same length as ``masses``.
        mass_step: float, default 0.01
            Mass grid step size, in solar masses.
        radius_step: float, default 0.03
            Radius grid step size, in km.
        """
        mass_bins = self.set_bins(masses, mass_step)
        rad_bins = self.set_bins(radii, radius_step)
        if 3 * len(mass_bins) * len(rad_bins) > len(masses):
            print(
                "Warning: The histogram might be to sparsely populated to get meaningful results."
            )

        histogram, self.rad_edges, self.mass_edges = np.histogram2d(
            radii, masses, bins=[rad_bins, mass_bins], weights=weights, density=True
        )
        drad = self.rad_edges[1] - self.rad_edges[0]
        dmass = self.mass_edges[1] - self.mass_edges[0]

        self.histogram = gaussian_filter(histogram * dmass * drad, sigma=3)

    def set_bins(self, array, step_size, sensitivity=0.001):
        """Bin edges for ``array``, from the [sensitivity, 1-sensitivity]
        quantile range padded by 5% on each side, stepped by ``step_size``.

        Parameters
        ----------
        array: np.array
        step_size: float
        sensitivity: float, default 0.001
            Quantile tail to trim before padding.

        Returns
        -------
        np.ndarray
            Bin edges.
        """
        low, high = np.quantile(array, [sensitivity, 1.0 - sensitivity])
        bins = np.arange(0.95 * low, 1.05 * high, step_size, dtype=np.float64)
        return bins

    def log_likelihood(self, parameters):
        """Log of this constraint's binned mass-radius pdf summed along each
        current EOS curve.

        Returns
        -------
        float or np.ndarray
            Shape ``eos_converter.batch_shape``.
        """
        test_radii, _ = self.eos_converter.curves_at(self.test_masses)
        n_rad, n_mass = self.histogram.shape
        mass_bins = np.searchsorted(self.mass_edges, self.test_masses, side="right") - 1
        rad_bins = np.searchsorted(self.rad_edges, test_radii, side="right") - 1
        inside = (
            (test_radii > 0.0)  # curves_at gives radius 0 outside the curve
            & (rad_bins >= 0)
            & (rad_bins < n_rad)
            & (mass_bins >= 0)
            & (mass_bins < n_mass)
        )
        probabilities = self.histogram[
            np.clip(rad_bins, 0, n_rad - 1), np.clip(mass_bins, 0, n_mass - 1)
        ]
        with np.errstate(divide="ignore"):
            log_l = np.log(np.where(inside, probabilities, 0.0).sum(axis=-1))
        return self.reshape_to_batch(log_l)

    def plot(self, ax, **kwargs):
        """Draw 50%/90% highest-posterior-density contours of this
        constraint's mass-radius pdf on ``ax``, labelling the 50% contour
        with ``self.name``.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            Mass-radius axes to draw on (mass on y, radius on x).
        **kwargs
            Overrides for ``self.plot_kwargs`` (e.g. ``color``, or
            ``manual`` — an (x, y) tuple or list of them for manual
            contour-label placement), passed to ``ax.contour``.

        Returns
        -------
        ax: matplotlib.axes.Axes
            The same axes, for chaining.
        """
        Xc = 0.5 * (self.rad_edges[:-1] + self.rad_edges[1:])
        Yc = 0.5 * (self.mass_edges[:-1] + self.mass_edges[1:])

        flat = self.histogram.flatten()
        order = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[order])
        levels = [0.9, 0.5]
        clevels = [flat[order][np.searchsorted(cumsum, p)] for p in levels]

        plot_kwargs = self.plot_kwargs | kwargs
        manual = plot_kwargs.get("manual", False)
        color = plot_kwargs.get("color", ax._get_lines.get_next_color())
        cmap = fading_cmap(color)
        colors = cmap(levels[::-1])
        contour = ax.contour(
            Xc, Yc, self.histogram.T, levels=clevels, colors=colors, linewidths=2
        )

        if isinstance(manual, bool) or manual is None:
            manual = False
        elif isinstance(manual, (list, tuple)):
            if len(manual) == 1:
                assert len(manual[0]) == 2, (
                    "Manual position for label must be a tuple of (x,y) coordinates"
                )
            elif (
                len(manual) == 2
                and isinstance(manual[0], (int, float, np.floating))
                and isinstance(manual[1], (int, float, np.floating))
            ):
                manual = [manual]

        ax.clabel(
            contour,
            levels=[clevels[-1]],
            fmt={clevels[-1]: self.name},
            inline=True,
            fontsize=ax.xaxis.label.get_size(),
            manual=manual,
            zorder=20,
        )
        return ax


class PulsarConstraint(LowerMTOVConstraint):
    """legacy synonym for general LowerMTOVConstraint"""


class MTOVUpperConstraint(UpperMTOVConstraint):
    """legacy synonym for general UpperMTOVConstraint"""


class JointConstraint(JointEoSConstraint):
    """legacy synonym for JointEoSConstraint"""


##### routines to organise a collection of tabulated EoSs
def tabulated_eos_constraint(eos_data, *constraints, Neos=None, mass_grid=None):
    """JointEoSConstraint of ``constraints`` on a set of tabulated EOSs, i.e.
    with an EoSConverter around a TabulatedEoSGenerator.

    Parameters
    ----------
    eos_data: str | Path
        Directory of ``{j}.dat`` files, or a glob pattern, see
        ``TabulatedEoSGenerator.read_eos_files``. Tables need columns radius
        [km], mass [M_sun] and Lambda (or ln Lambda).
    *constraints: EoSConstraint | JointEoSConstraint | dict | list
        Constraints to combine; lists and tuples are unpacked.
    Neos: int, optional
        Number of EOSs to read, default all.
    mass_grid: MassGrid | dict, optional
        Grid the tables are resampled onto, see ``TabulatedEoSGenerator``.

    Returns
    -------
    JointEoSConstraint
    """
    flat_constraints = []
    for constraint in constraints:
        if isinstance(constraint, (list, tuple)):
            flat_constraints.extend(constraint)
        else:
            flat_constraints.append(constraint)
    generator = TabulatedEoSGenerator(
        Namespace(eos_data=eos_data, Neos=Neos), mass_grid=mass_grid
    )
    eos_converter = EoSConverter(generator=generator, macro_only=True)
    return JointEoSConstraint(*flat_constraints, eos_converter=eos_converter)


def EOSConstraints2Prior(macro_eos_path, out_path, Constraint):
    """
    Given a directory of macroscopic EOSs and EOS constraint(s),
    returns sorted EOSs and the corresponding prior

    Parameters
    ----------
    macro_eos_path: str
        Path to the original EOSs: a directory of ``{j}.dat`` files with columns
        radius, mass, Lambda, or a glob pattern, see ``tabulated_eos_constraint``
    out_path: str
        Directory to write the sorted EOSs to, as ``1.dat``, ``2.dat``, ... in
        ascending prior weight
    Constraint: EoSConstraint | JointEoSConstraint | list
        The EOS constraint(s) to be considered

    Returns
    -------
    prior: bilby.core.Prior
        Prior on the EOSs
    logNorm: float
        log normalization constant for the EOS prior
    """

    joint_constraint = tabulated_eos_constraint(macro_eos_path, Constraint)
    parameters = joint_constraint.tabulated_eos_parameters()
    logLs = joint_constraint.log_weights(parameters)
    logNorm = logsumexp(logLs)

    weights = np.exp(logLs - logNorm)
    sort_idcs = joint_constraint.write_sorted_eos(out_path, parameters["EOS"], weights)
    prior = WeightedCategorical(len(weights), weights[sort_idcs], name="EOS")

    return prior, logNorm
