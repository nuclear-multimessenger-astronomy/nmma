import corner
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import seaborn
from matplotlib import pyplot as plt
from ast import literal_eval
import bilby

from ..core.conversion import (
    chirp_mass_and_eta_to_component_masses,
    tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities,
    label_mapping,
)
from ..core import utils, parsing
from ..core import plotting_utils as corepu
from .parser import corner_plot_parser

nmma_colors = corepu.fig_setup()


def setup_plot_quantities(
    posterior_samples,
    limits,
    plot_keys,
    injection,
    post_dir=None,
    default_labels={},
    **plot_kwargs,
):
    """
    Collect the quantities used to plot a set of posterior samples.

    Loads the samples, decides which parameters to show, and builds the
    labels, summary titles, axis limits and truth values shared by
    :func:`plot_histograms_only` and :func:`setup_corner_plot`.

    Parameters
    ----------
    posterior_samples : pandas.DataFrame, dict, str, pathlib.Path or argparse.Namespace
        Samples, or something :func:`nmma.core.utils.get_posteriors` can
        load them from.
    limits : list of tuple or None
        One ``(min, max)`` pair per entry of ``plot_keys``, widened in place
        to cover the samples. If None, each entry starts at ``(inf, -inf)``.
    plot_keys : list of str or None
        Parameters to show. If None, every column of the samples is used
        except ``log_likelihood`` and ``log_prior``.
    injection : dict, pandas.DataFrame, pandas.Series or None
        Truth values. A DataFrame is reduced to its first row.
    post_dir : str or pathlib.Path, default=None
        Directory to resolve ``posterior_samples`` against, passed through
        to :func:`nmma.core.utils.get_posteriors`.
    default_labels : dict, default={}
        Fallback axis labels, consulted for keys that
        :data:`nmma.core.conversion.label_mapping` does not cover.
    **plot_kwargs
        Only ``quantiles`` (passed to :func:`nmma.core.utils.sig_lims`) and
        ``label`` are read.

    Returns
    -------
    dict
        With keys ``samples`` (``(n_samples, n_keys)`` array), ``keys``,
        ``labels``, ``titles``, ``limits``, ``truths`` and ``best_fit``
        (the highest-``log_likelihood`` row, or all-None if the samples
        carry no ``log_likelihood`` column).
    """
    plot_quantities = {}
    # load samples
    posterior_samples = utils.get_posteriors(posterior_samples, post_dir)
    try:
        plot_quantities["best_fit"] = posterior_samples.iloc[
            posterior_samples["log_likelihood"].idxmax()
        ]
    except KeyError:
        plot_quantities["best_fit"] = {k: None for k in posterior_samples.columns}
    if plot_keys is None:
        plot_keys = posterior_samples.columns.tolist()  # show all we can
        for key in ["log_likelihood", "log_prior"]:
            if key in plot_keys:
                plot_keys.remove(key)  # but not the likelihood itself
    if limits is None:
        limits = [
            (np.inf, -np.inf) for _ in plot_keys
        ]  # will adjust more permissively later
    # find what to actually plot
    plot_samples, plot_labels, titles = [], [], []
    for i, k in enumerate(plot_keys):
        try:  # we have data to show
            show_data = posterior_samples[k].to_numpy()
            plot_samples.append(show_data)
            titles.append(utils.sig_lims(show_data, plot_kwargs.get("quantiles", None)))

            cur_min, cur_max = limits[i]
            limits[i] = (
                min(cur_min, np.amin(show_data)),
                max(cur_max, np.amax(show_data)),
            )

            label = label_mapping.get(k, default_labels.get(k, k))
            plot_labels.append(label)
        except KeyError:
            print(
                f"{plot_kwargs.get('label', 'Warning')} : key {k} was not found in the posterior samples; Will try to insert dummy plot."
            )
            cur_min, cur_max = limits[i]
            if (
                cur_min > cur_max
            ):  # meaning no data seen so far, so we have to take chances
                cur_min, cur_max = (
                    1e42,
                    -1e42,
                )  # just set crazy limits that we can still work on.
                limits[i] = (cur_min, cur_max)
            # some dummy values that should remain out of frame
            plot_samples.append(
                np.linspace(100 * cur_max, 1001 * cur_max, posterior_samples.shape[0])
            )
            plot_labels.append("")
            titles.append("")

    plot_quantities["labels"] = plot_labels
    plot_quantities["titles"] = titles
    plot_quantities["samples"] = np.column_stack(plot_samples)
    plot_quantities["limits"] = limits
    plot_quantities["keys"] = plot_keys
    if injection is not None:
        if isinstance(injection, pd.DataFrame):
            injection = injection.iloc[0]
        if isinstance(injection, pd.Series):
            injection = injection.to_dict()
        plot_quantities["truths"] = [injection.get(k, None) for k in plot_keys]
    else:
        plot_quantities["truths"] = None

    return plot_quantities


def plot_histograms_only(
    posterior_samples,
    limits=None,
    plot_keys=None,
    fig=None,
    injection=None,
    post_dir=None,
    default_labels={},
    best_fit=False,
    show_titles=True,
    prior=None,
    ncols=None,
    title_kwargs={},
    fig_kwargs={},
    **plot_kwargs,
):
    """
    Plot the 1D marginal histogram of each parameter on its own panel.

    Parameters
    ----------
    posterior_samples : pandas.DataFrame, dict, str, pathlib.Path or argparse.Namespace
        Samples, or something :func:`nmma.core.utils.get_posteriors` can
        load them from.
    limits : list of tuple, default=None
        One ``(min, max)`` pair per entry of ``plot_keys``, widened in place
        to cover the samples.
    plot_keys : list of str, default=None
        Parameters to show. Defaults to every column except
        ``log_likelihood`` and ``log_prior``.
    fig : matplotlib.figure.Figure, default=None
        Figure to draw into. If None, one is created with
        :func:`nmma.core.plotting_utils.setup_multi_axes`.
    injection : dict, pandas.DataFrame, pandas.Series, default=None
        Truth values. A scalar is drawn as a vertical line, a pair as a
        shaded band.
    post_dir : str or pathlib.Path, default=None
        Directory to resolve ``posterior_samples`` against.
    default_labels : dict, default={}
        Fallback axis labels.
    best_fit : bool, default=False
        Mark the highest-likelihood sample with a dash-dotted line.
    show_titles : bool, default=True
        Put the summary title above each panel. If False, the label is
        placed on the top axis instead.
    prior : dict, default=None
        Priors to overplot. ``bilby`` ``Constraint`` priors are skipped.
    ncols : int, default=None
        Panel columns, passed to
        :func:`nmma.core.plotting_utils.setup_multi_axes`.
    title_kwargs : dict, default={}
        Text properties for the panel titles. ``color`` and ``fontsize``
        default to the series colour and the axis title size.
    fig_kwargs : dict, default={}
        Passed to :func:`nmma.core.plotting_utils.setup_multi_axes`.
    **plot_kwargs
        Passed to ``Axes.hist``. ``color`` selects the series colour,
        otherwise the next colour of ``nmma_colors`` is taken; ``label``
        adds an entry to a figure-level legend.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.
    list of tuple
        The axis limits.
    """
    plot_quantities = setup_plot_quantities(
        posterior_samples,
        limits,
        plot_keys,
        injection,
        post_dir,
        default_labels,
        **plot_kwargs,
    )

    if fig is None:
        fig, axes = corepu.setup_multi_axes(
            len(plot_quantities["keys"]), ncols=ncols, **fig_kwargs
        )
    else:
        axes = fig.get_axes()
    label = plot_kwargs.pop("label", None)

    color = plot_kwargs.pop("color") if "color" in plot_kwargs else next(nmma_colors)
    for i, ax in enumerate(axes):
        if i >= len(plot_quantities["keys"]):
            ax.axis("off")  # Hide any extra subplots
            continue

        key = plot_quantities["keys"][i]
        ax.hist(
            plot_quantities["samples"][:, i],
            bins=50,
            density=True,
            color=color,
            alpha=0.7,
            histtype="step",
            **plot_kwargs,
        )

        if plot_quantities["truths"] is not None:
            injected = plot_quantities["truths"][i]
            if injected is None:
                pass
            elif isinstance(injected, (int, float, np.floating)) or len(injected) == 1:
                ax.axvline(injected, color="tab:orange", linestyle="--")
            elif len(injected) == 2:
                ax.fill_betweenx(
                    ax.get_ylim(),
                    injected[0],
                    injected[1],
                    color="tab:orange",
                    alpha=0.3,
                    step="post",
                )

        if best_fit is True:
            if plot_quantities["best_fit"][key] is not None:
                ax.axvline(
                    plot_quantities["best_fit"][key], color=color, linestyle="-."
                )

        xlim = plot_quantities["limits"][i]
        if isinstance(prior, dict):
            if key in prior and not isinstance(prior[key], bilby.core.prior.Constraint):
                _range = np.linspace(*xlim, 300)
                ax.plot(
                    _range,
                    prior[key].prob(_range),
                    color=color,
                    alpha=0.5,
                    linewidth=1,
                    linestyle="--",
                )

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, prune="both"))
        [l.set_rotation(45) for l in ax.get_xticklabels()]
        [l.set_rotation(45) for l in ax.get_xticklabels(minor=True)]
        ax.autoscale(enable=True, axis="x", tight=True)
        ax.set_yticks([])
        ax.set_yticklabels([])

        if show_titles:
            use_kwargs = title_kwargs.copy()
            if "color" not in title_kwargs:
                use_kwargs["color"] = color
            if "fontsize" not in title_kwargs:
                use_kwargs["fontsize"] = ax.title.get_fontsize()
            ax = prepare_titles(ax, plot_quantities, i, use_kwargs)
        else:
            ax.xaxis.set_label_position("top")
            # FIXME: show_titles=False branch reads title_kwargs['fontsize']; default {}
            # raises KeyError
            ax.set_xlabel(
                plot_quantities["labels"][i], fontsize=title_kwargs["fontsize"]
            )

    # allow joint legend
    if label:
        fig.legends.clear()
        fig.axes[0].plot([], [], label=label, color=color, **plot_kwargs)
        leg_elements = len(fig.axes[0].get_legend_handles_labels()[0])
        leg_ncols = leg_elements if leg_elements < 4 else 2
        fig.legend(
            ncols=leg_ncols,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            handlelength=2,
        )

    return fig, plot_quantities["limits"]


def plot_multi_corner(args, key_selection=None, save=False):

    """
    Draw a corner plot for each entry of ``args.posterior_files``, each
    into the same figure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``posterior_files``, ``label_name``,
        ``kwargs`` (a string evaluated with :func:`ast.literal_eval` and
        forwarded to ``corner.corner``), ``prior``, ``injection_json``,
        ``injection_num``, ``bestfit_params``, ``bestfit_json`` and
        ``verbose``.
    key_selection : list of str, default=None
        Restrict the plotted parameters to this subset of the prior, passed
        to
        :func:`nmma.core.plotting_utils.plotting_parameters_from_priors`.
    save : str or bool, default=False
        Path to write the figure to. If falsy, nothing is written.

    Returns
    -------
    The value returned by the last :func:`setup_corner_plot` call.

    Notes
    -----
    Truths are taken from ``injection_json`` at row ``injection_num`` if
    given, otherwise from ``bestfit_params``. Quantiles are fixed at
    ``[0.16, 0.5, 0.84]``.
    """
    plot_kwargs = literal_eval(args.kwargs)
    quantiles = [0.16, 0.5, 0.84]
    fig = None
    labels = (
        [lab for lab in args.label_name]
        if args.label_name is not None
        else [f for f in args.posterior_files]
    )
    for i, f in enumerate(args.posterior_files):
        # FIXME: Unpacking dict .items() into plot_keys/plot_labels yields tuples,
        # usually raises ValueError
        # FIXME: plot_multi_corner reads args.prior; parser defines prior_filename,
        # causing AttributeError
        plot_keys, plot_labels = corepu.plotting_parameters_from_priors(
            args.prior, keys=key_selection
        ).items()
        if args.injection_json is not None:
            truths = utils.read_injection_file(args.injection_json)
            truths = truths.iloc[args.injection_num].to_dict()
            truths = np.array([truths[k] for k in plot_keys])
            # FIXME: args.verbose read; corner_plot_parser defines no --verbose, causing
            # AttributeError
            if args.verbose:
                print("\nLoaded Injection:")
                print(f"Truths from injection: {truths}")
        elif args.bestfit_params is not None:
            # FIXME: args.bestfit_json read; parser only defines bestfit_params.
            # AttributeError.
            truths = utils.read_bestfit_from_json(
                args.bestfit_json, plot_keys, args.verbose
            )
        else:
            truths = None

        # FIXME: Tuple from setup_corner_plot assigned to fig; fig.savefig raises
        # AttributeError
        fig = setup_corner_plot(
            f,
            # FIXME: Empty list passed as limits; setup_plot_quantities indexes
            # limits[i], raising IndexError
            [],
            label=labels[i],
            # FIXME: truths lands in plot_kwargs, duplicating explicit truths at
            # corner_plot call
            truths=truths,
            fig=fig,
            quantiles=quantiles,
            plot_keys=plot_keys,
            default_labels=plot_labels,
            **plot_kwargs,
        )
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=300)
        print("\nSaved corner plot:", save)
    return fig


def setup_corner_plot(
    posterior_samples,
    limits=None,
    plot_keys=None,
    fig=None,
    injection=None,
    post_dir=None,
    default_labels={},
    prior=None,
    show_titles=True,
    best_fit=False,
    **plot_kwargs,
):

    """
    Draw a corner plot of the posterior samples.

    Wraps :func:`corner_plot` and then adjusts the diagonal panels: rescales
    them to their histogram height, optionally overplots the prior and the
    best fit, and rewrites the titles through :func:`prepare_titles`.

    Parameters
    ----------
    posterior_samples : pandas.DataFrame, dict, str, pathlib.Path or argparse.Namespace
        Samples, or something :func:`nmma.core.utils.get_posteriors` can
        load them from.
    limits : list of tuple, default=None
        One ``(min, max)`` pair per entry of ``plot_keys``, widened in place
        to cover the samples.
    plot_keys : list of str, default=None
        Parameters to show. Defaults to every column except
        ``log_likelihood`` and ``log_prior``.
    fig : matplotlib.figure.Figure, default=None
        Figure to draw into.
    injection : dict, pandas.DataFrame, pandas.Series, default=None
        Truth values, resolved by :func:`setup_plot_quantities` and passed
        to ``corner.corner`` as ``truths``.
    post_dir : str or pathlib.Path, default=None
        Directory to resolve ``posterior_samples`` against.
    default_labels : dict, default={}
        Fallback axis labels.
    prior : dict, default=None
        Priors to overplot on the diagonal. ``bilby`` ``Constraint`` priors
        are skipped.
    show_titles : bool or None, default=True
        True puts the summary title on the diagonal panels via
        :func:`prepare_titles`; None puts the label on the top axis
        instead; False leaves the titles alone.
    best_fit : bool, default=False
        Mark the highest-likelihood sample with a dash-dotted line.
    **plot_kwargs
        Passed to :func:`corner_plot`. ``color`` selects the series colour,
        otherwise the next colour of ``nmma_colors`` is taken; ``label``
        adds a legend entry; ``title_kwargs`` sets the title text
        properties.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.
    list of tuple
        The axis limits.
    """
    plot_quantities = setup_plot_quantities(
        posterior_samples,
        limits,
        plot_keys,
        injection,
        post_dir,
        default_labels,
        **plot_kwargs,
    )
    color = plot_kwargs.pop("color", next(nmma_colors))
    fig = corner_plot(
        plot_quantities["samples"],
        plot_quantities["labels"],
        plot_quantities["limits"],
        fig=fig,
        truths=plot_quantities["truths"],
        color=color,
        show_titles=False,
        **plot_kwargs,
    )

    axes = fig.get_axes()
    for i, key in enumerate(plot_quantities["keys"]):
        ax = axes[i * len(plot_quantities["keys"]) + i]
        max_height = max(np.max(poly.get_xy()[:, 1]) for poly in ax.patches)
        ax.set_ylim(top=1.05 * max_height)
        if isinstance(prior, dict):
            if key in prior and not isinstance(prior[key], bilby.core.prior.Constraint):
                _range = np.linspace(*(ax.get_xlim()), 300)
                ax.plot(_range, prior[key].prob(_range), color=color, alpha=0.6)

        # adjust titles
        title_kwargs = plot_kwargs.get("title_kwargs", {}).copy()
        if "color" not in title_kwargs:
            title_kwargs["color"] = color
        if "fontsize" not in title_kwargs:
            title_kwargs["fontsize"] = ax.title.get_fontsize()
        if show_titles:
            len_ax = len(plot_quantities["keys"])
            offset_ax = axes[len_ax * (len_ax - 1) + i]
            ax = prepare_titles(ax, plot_quantities, i, title_kwargs, offset_ax)
            ax.yaxis.label.set_visible(False)  #
        elif show_titles is None:
            ax.xaxis.set_label_position("top")
            ax.set_xlabel(
                plot_quantities["labels"][i], fontsize=title_kwargs["fontsize"]
            )
        else:
            pass

        if best_fit is True:
            if plot_quantities["best_fit"][key] is not None:
                ax.axvline(
                    plot_quantities["best_fit"][key], color=color, linestyle="-."
                )

    # allow joint legend
    if "label" in plot_kwargs:
        fig.legends.clear()
        fig.axes[i].plot([], [], label=plot_kwargs["label"], color=color)
        fig.axes[i].legend(
            loc="upper right",
            bbox_to_anchor=(1.05, 1.05),
            edgecolor="none",
            handlelength=1.5,
            fontsize=1.5 * title_kwargs["fontsize"],
        )
    return fig, plot_quantities["limits"]


def prepare_titles(ax, plot_quantities, i, title_kwargs, offset_ax=None):

    """
    Write the summary title for one panel.

    Folds any axis offset into the quoted numbers, then either sets the
    title directly (when this is the only text on the panel) or hands the
    panel to :func:`nmma.core.plotting_utils.arange_titles`, which
    positions the texts according to how many have accumulated.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Panel to title.
    plot_quantities : dict
        As returned by :func:`setup_plot_quantities`; ``titles`` and
        ``labels`` are read.
    i : int
        Index into ``titles`` and ``labels``.
    title_kwargs : dict
        Text properties for the title. The key ``move`` is consumed here,
        not passed on: it lists the indices, titles or labels whose text
        should be shifted upward.
    offset_ax : matplotlib.axes.Axes, default=None
        Axis whose tick formatter supplies the offset. Defaults to ``ax``.

    Returns
    -------
    matplotlib.axes.Axes
        The titled panel.
    """
    title = plot_quantities["titles"][i]
    if offset_ax is None:
        offset_ax = ax
    offset_ax.xaxis.get_ticklabels()
    new_title = corepu.format_title_offset(offset_ax, title)
    move = title_kwargs.pop("move", [])
    ax.text(0.0, 0.0, new_title, transform=ax.transAxes, **title_kwargs)
    if len(ax.texts) == 1:
        ax.set_title(
            f'{plot_quantities["labels"][i]}={ax.texts[0].get_text()}',
            color="black",
            fontsize=ax.texts[0].get_fontsize(),
        )
        ax.texts[0].set_visible(False)
    else:
        do_move = (
            True
            if (
                i in move
                or plot_quantities["titles"][i] in move
                or plot_quantities["labels"][i] in move
            )
            else False
        )
        ax.yaxis.set_label_position("left")
        ax.set_ylabel(plot_quantities["labels"][i], fontsize=ax.texts[0].get_fontsize())
        ax.texts[0].set_visible(True)
        ax = corepu.arange_titles(ax, move=do_move)

    return ax


def corner_plot(plot_samples, labels, limits, fig=None, save=False, **kwargs):
    """
    Draw a corner plot with the NMMA default styling.

    Thin wrapper around ``corner.corner``. The defaults set 50 bins, a
    smoothing of 1.3, quantiles ``[0.16, 0.5, 0.84]``, contour levels
    ``(0.10, 0.32, 0.68, 0.95)``, ``fill_contours=True``, and
    ``plot_density=False`` with ``plot_datapoints=False``.

    Parameters
    ----------
    plot_samples : numpy.ndarray
        Samples of shape ``(n_samples, n_parameters)``.
    labels : list of str
        Axis label per parameter.
    limits : list of tuple
        ``(min, max)`` range per parameter, passed as ``range``.
    fig : matplotlib.figure.Figure, default=None
        Figure to draw into.
    save : str or bool, default=False
        Path to write the figure to. If falsy, nothing is written.
    **kwargs
        Passed to ``corner.corner``, overriding any of the defaults above.

    Returns
    -------
    matplotlib.figure.Figure
        The figure drawn into.
    """
    default_kwargs = dict(
        bins=50,
        smooth=1.3,
        label_kwargs=dict(fontsize=16),
        show_titles=True,
        title_kwargs=dict(fontsize=16),
        color="C1",  # color='#0072C1',
        truth_color="tab:orange",
        quantiles=[0.16, 0.5, 0.84],
        levels=(0.10, 0.32, 0.68, 0.95),
        median_line=True,
        title_fmt=".2f",
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        max_n_ticks=4,
        hist_kwargs={"density": True},
    )
    default_kwargs.update(kwargs)

    fig = corner.corner(
        plot_samples, labels=labels, range=limits, fig=fig, **default_kwargs
    )
    if save:
        plt.savefig(save, bbox_inches="tight")
    return fig


def resampling_corner_plot(posterior_samples, solution, outdir, withNSBH):

    r"""
    Draw the corner plot summarising a GW-EM resampling run.

    Converts the sampled chirp mass and mass ratio to component masses and,
    for a BNS, reads the tidal deformabilities and the maximum mass from the
    EOS tables held on the solver: the deformabilities are interpolated at
    the component masses, the maximum mass is the last tabulated mass.

    Parameters
    ----------
    posterior_samples : pandas.DataFrame
        Resampling output, with columns ``chirp_mass``, ``mass_ratio``,
        ``alpha``, ``zeta`` and, for a BNS, ``EOS``.
    solution : EjectaResamplerMixIn
        The resampling solver, read for its ``EOS_masses_dict`` and
        ``EOS_lambda_dict`` tables.
    outdir : str
        Output directory, passed through to :func:`corner_plot`.
    withNSBH : bool
        True plots the NSBH parameters
        (:math:`\mathcal{M}_c, q, \alpha, \zeta`); False adds the
        effective tidal deformability and the maximum mass for a BNS.

    Returns
    -------
    None

    Notes
    -----
    For a BNS the 90th percentile of the effective tidal deformability is
    printed. The mass ratio is plotted inverted (:math:`m_1 / m_2`) and
    capped at 3; the maximum mass is capped at 2.7 solar masses.
    """
    mc = posterior_samples["chirp_mass"].to_numpy()
    invq = posterior_samples["mass_ratio"].to_numpy()
    alpha = posterior_samples["alpha"].to_numpy()
    zeta = posterior_samples["zeta"].to_numpy()

    eta = invq / np.power(1.0 + invq, 2)
    m1, m2 = chirp_mass_and_eta_to_component_masses(mc, eta)
    q = m1 / m2  # inverted

    if withNSBH:  # NSBH resampling result corner plot
        labels = [
            r"$\mathcal{M}_c[M_{\odot}]$",
            r"$q$",
            r"$\alpha[M_{\odot}]$",
            r"$\zeta$",
        ]
        plot_samples = np.vstack((mc, q, alpha, zeta))
        limits = (
            (np.amin(mc), np.amax(mc)),
            (np.amin(q), 3),
            (np.amin(alpha), np.amax(alpha)),
            (np.amin(zeta), np.amax(zeta)),
        )

    else:  # BNS resampling result corner plot
        labels = [
            r"$\mathcal{M}_c[M_{\odot}]$",
            r"$q$",
            r"$\tilde{\Lambda}$",
            r"$\alpha[M_{\odot}]$",
            r"$\zeta$",
            r"$M_{\rm{max}}[M_{\odot}]$",
        ]

        EOS = posterior_samples["EOS"].astype(int) + 1
        lambda1 = []
        lambda2 = []
        for i in range(len(EOS)):
            EOSsample = EOS[i]
            lam1, lam2 = np.interp(
                [m1[i], m2[i]],
                solution.EOS_masses_dict[EOSsample],
                solution.EOS_lambda_dict[EOSsample],
            )
            lambda1.append(lam1)
            lambda2.append(lam2)
        lambda1 = np.array(lambda1)
        lambda2 = np.array(lambda2)
        lambdaT, _ = tidal_deformabilities_and_mass_ratio_to_eff_tidal_deformabilities(
            lambda1, lambda2, q
        )

        MTOV = []
        for EOSsample in EOS:
            MTOV.append(solution.EOS_masses_dict[EOSsample][-1])
        MTOV = np.array(MTOV)

        print(
            "The 90% upper bound for lambdaT is {0}".format(np.quantile(lambdaT, 0.9))
        )
        plot_samples = np.vstack((mc, q, lambdaT, alpha, zeta, MTOV))
        limits = (
            (np.amin(mc), np.amax(mc)),
            (np.amin(q), 3),
            (np.amin(lambdaT), np.amax(lambdaT)),
            (np.amin(alpha), np.amax(alpha)),
            (np.amin(zeta), np.amax(zeta)),
            (np.amin(MTOV), 2.7),
        )
    # FIXME: outdir passed positionally as corner_plot's fig argument; nothing is saved
    corner_plot(plot_samples.T, labels, limits, outdir)


def plot_R14_trend(args):
    # load the data
    """
    Plot the R14 constraint as a function of the number of events.

    Draws two stacked panels: the median radius with its asymmetric error
    bars for the GW-only and the joint GW+EM data sets, and below it the
    mean of the two errors as a percentage of the median, on a log scale,
    with reference lines at 10, 5 and 1 per cent.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. Reads ``outdir`` and ``label`` to locate
        ``{outdir}/GW_EM_R14trend_{label}.dat``, ``gwR14trend`` to locate
        ``{gwR14trend}/GW_R14trend.dat``, and ``R14_true`` for the
        reference line marking the injected value.

    Returns
    -------
    None
        The figure is written to
        ``{outdir}/R14_trend_GW_EM_{label}.pdf``.
    """
    data_GWEM = pd.read_csv(
        f"{args.outdir}/GW_EM_R14trend_{args.label}.dat", header=0, delimiter=" "
    )
    data_GW = pd.read_csv(f"{args.gwR14trend}/GW_R14trend.dat", header=0, delimiter=" ")

    # initialise the figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.suptitle("Constrain EoS using EM + GW", fontname="Times New Roman Bold")

    ax1.set_ylabel(r"$R_{1.4} \ [{\rm km}]$")
    ax2.set_ylabel(r"$\delta R_{1.4} / R_{1.4} \ [\%]$")
    ax2.set_xlabel("Events")

    # set reference lines
    c = seaborn.color_palette("colorblind")
    ax1.axhline(args.R14_true, linestyle="--", color=c[1], label="Injected value")
    for y in [10, 5, 1]:
        ax2.axhline(y, color="grey", linestyle="--", alpha=0.5)
    ax2.set_yscale("log")

    # plot the data
    for label, color, data_set in zip(
        ["GW", "GW+EM"], [c[3], c[0]], [data_GW, data_GWEM]
    ):
        x_values = np.arange(1, len(data_set) + 1)
        ax1.errorbar(
            x_values,
            data_set.R14_med,
            yerr=[data_set.R14_lowerr, data_set.R14_uperr],
            label=label,
            color=color,
            fmt="o",
            capsize=5,
        )
        mean_error = np.mean([data_set.R14_lowerr, data_set.R14_uperr], axis=0)
        ax2.plot(x_values, mean_error / data_set.R14_med * 100, color=color, marker="o")

    # legend and limits
    ax1.legend()
    ax1.set_xlim([0.5, len(data_GWEM) + 0.5])
    ax2.set_xticks(np.arange(1, len(data_GWEM) + 1, 2))

    fig.tight_layout()
    # fig.subplots_adjust(hspace=0.1)
    plt.savefig(f"{args.outdir}/R14_trend_GW_EM_{args.label}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    args = parsing.nmma_base_parsing(corner_plot_parser)
    plot_multi_corner(args)
