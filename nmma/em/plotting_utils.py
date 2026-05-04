import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

params = {
    "backend": "pdf",
    "figure.figsize": [18, 25],
}
matplotlib.rcParams.update(params)
matplotlib.rcParams['text.usetex'] = (os.environ.get("CI") != 'true')
matplotlib.rcParams['figure.labelsize'] = 20
# matplotlib.rcParams['axes.titlesize'] = 'large'
if os.environ.get("CI") == 'true':
    matplotlib.use("agg")
from nmma.core.plotting_utils import fig_setup
nmma_colors = fig_setup()

##############################################
################# MAIN PLOTS #################
##############################################
def basic_em_analysis_plot(
        transient, plot_filters, mags_to_plot, error_dict, chi2_dict, 
        mismatches, sub_model_plot_props, xlim, ylim, save_path,
        ncol = 2, fig = None, shared_data = True, **kwargs):

    
    if fig:
        return add_em_analysis_plot(fig, transient, mags_to_plot, error_dict, mismatches, sub_model_plot_props, xlim, ylim, save_path,  shared_data,  **kwargs)

    time = mags_to_plot.pop("time")
    filter_names = list(plot_filters.keys())
    n_filters = len(filter_names)
    data_colors = plt.cm.plasma(np.linspace(0, 1, len(filter_names)))[::-1]
    color = kwargs.pop('color', next(nmma_colors))

    fig, axes = analysis_plot_geometry(filter_names, ncol=ncol)
    fig.supylabel("AB magnitude", rotation=90)
    fig.supxlabel("Time [days]")
    if xlim is None:
        xlim = get_time_limits_from_obs_data(transient)
    else:
        xlim = check_limit(xlim)
        
    if ylim is None:
        ylim = get_mag_limits_from_obs_data(transient, filter_names)
    else:
        shared_ylim = check_limit(ylim)
        ylim = {filt: shared_ylim for filt in filter_names}

    
    for cnt, filt in enumerate(filter_names):
        # summary plot
        row, col = divmod(cnt, ncol)
        ax_sum = axes[row, col]
        ax_sum.set_ylim(ylim[filt])
        ax_sum.set_xlim(xlim)
        if xlim[0] > 0:
            ax_sum.set_xscale('log')
        ax_sum.set_ylabel(plot_filters[filt])

        # adding the ax for the Delta
        divider = make_axes_locatable(ax_sum)
        ax_delta = divider.append_axes('bottom',
                                        size='40%',
                                        sharex=ax_sum,
                                        pad=0.1)
        ax_delta.set_ylabel(r"$\Delta$ mag")

        # ax_delta.set_yscale('log')
        # ax_delta.yaxis.set_major_locator(MaxNLocator(min_n_ticks=2))
        
        if cnt not in [n_filters - i-1 for i in range(ncol)]:  
            ax_sum.set_xticklabels([])
        else:
            plt.setp(ax_sum.get_xticklabels(), visible=False)
            # only show x labels on the lowest delta plots


        # configuring ax_sum

        if cnt not in [n_filters - i-1 for i in range(ncol)]:  # only show x labels on the last two plots
            ax_sum.set_xticklabels([])
        # plot the observations
        ax_sum, det_times = plot_observations(ax_sum, transient, data_colors[cnt], filter=filt)

        if det_times.size>0:

            # plot the mismatch between the model and the data
            diff_per_data, _ = mismatches[filt]
            ax_delta.axhline(0, linestyle='--', color='k')
            ax_delta.scatter(det_times, diff_per_data, color=color)
            ax_sum.plot([], [], label=fr'$\chi^2$ / d.o.f. = {round(chi2_dict[filt], 2)}', color=color)
            ax_sum.legend(loc='best', frameon=False, handlelength=0, handletextpad=0)
        

        # plot the best-fit lc with errors
        plot_bestfit_with_errors(ax_sum, time, mags_to_plot[filt], error_dict[filt], sub_model_plot_props, cnt, color)

        
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def add_em_analysis_plot(fig, 
        transient, mags_to_plot, error_dict, mismatches,
        sub_model_plot_props, xlim, ylim, save_path, shared_data = True, **kwargs):
    
    time = mags_to_plot.pop("time")
    filter_names = list(mags_to_plot.keys())
    n_axes = int(np.ceil(len(fig.axes)/2))

    data_colors = plt.cm.plasma(np.linspace(0, 1, len(filter_names)))[::-1]
    color = kwargs.pop('color', next(nmma_colors))

    if not shared_data:
        if xlim is None:
            add_xlim = get_time_limits_from_obs_data(transient)
            old_xlim = fig.axes[0].get_xlim()
            xlim = (min(add_xlim[0], old_xlim[0]), max(add_xlim[1], old_xlim[1]))
        if ylim is None:
            add_ylim = get_mag_limits_from_obs_data(transient, filter_names)
            ylim = {}
            for i, filt in enumerate(filter_names):
                old_ylim = fig.axes[i].get_ylim()
                ylim[filt] = (max(add_ylim[filt][0], old_ylim[0]), min(add_ylim[filt][1], old_ylim[1]))
                
    for cnt, filt in enumerate(filter_names):

        # summary plot
        ax_sum = fig.axes[cnt]
        ax_delta = fig.axes[cnt+n_axes]
        
        # plot the observations
        ax_sum, det_times = plot_observations(ax_sum, transient, data_colors[cnt], filter=filt)
        if not shared_data:
            ax_sum.set_xlim(xlim)
            ax_sum.set_ylim(ylim[filt])
            ax_delta.set_xlim(xlim)

        if det_times.size>0:            
            # plot the mismatch between the model and the data
            diff_per_data, _ = mismatches[filt]
            delta_ylim = ax_delta.get_ylim()
            ylim = (min(delta_ylim[0], 0.9*min(diff_per_data)), 
                    max(delta_ylim[1], 1.1*max(diff_per_data)))
            ax_delta.set_ylim(ylim)
            ax_delta.axhline(0, linestyle='--', color='k')
            ax_delta.scatter(det_times, diff_per_data, color=color)
            
            ax_sum.get_legend().remove()
            # ax_sum.plot([], [], label=round(chi2_dict[filt], 2), color=color)
            # ax_sum.legend(loc='best', frameon=False, handlelength=0, handletextpad=0, labelcolor='linecolor')
            
        plot_bestfit_with_errors(ax_sum, time, mags_to_plot[filt], error_dict[filt], sub_model_plot_props, cnt, color)
        
        
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig

def plot_bestfit_with_errors(ax_sum, time, mag_plot, error_budget, 
        sub_model_plot_props, cnt, color):

    label = 'combined' if sub_model_plot_props is not None else ""
    ax_sum.plot(time, mag_plot,
        color=color, linewidth=3, linestyle="--")
    ax_sum.fill_between(time,
        mag_plot + error_budget,
        mag_plot - error_budget,
        facecolor=color,
        alpha=0.2,
        label=label,
        )

    if sub_model_plot_props is not None:
        ## plot additional lcs for each sub_model
        for model_name, prop_dict in sub_model_plot_props.items():
            mag_plot = prop_dict['plot_mags'][cnt]
            mag_err = prop_dict['plot_errors'][cnt]
            plot_times = prop_dict['plot_times']
            ax_sum.plot(plot_times, mag_plot,
                color=color, linewidth=3, linestyle="--")
            ax_sum.fill_between(plot_times,
                mag_plot + mag_err,
                mag_plot - mag_err,
                facecolor=prop_dict['color'],
                alpha=0.2,
                label=model_name,
            )


def bolometric_lc_plot(transient, time, lc, save_path, color = "coral"):
    matplotlib.rcParams.update(
        {'font.size': 12, 'font.family': 'serif'
        }
    )
    fig, ax = plt.subplots(1, 1)
    ax, _ = plot_observations(ax, transient, markersize=12)

    ### plot the bestfit model
    ax.plot(time, lc,
            color=color, linewidth=3, linestyle="--")

    ax.set_ylabel("L [erg / s]")
    ax.set_xlabel("Time [days]")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualise_model_performance(training_data, training_model, light_curve_model, data_type):
    """Function to visualise training success and model performance by 
    comparing the model's light curves or spectra to the training data."""
    # we can plot an example where we compare the model performance
    # to the grid points

    training = next(iter(training_data.values()))  # get the first entry
    data = {param: training[param] for param in training_model.model_parameters}
    data["redshift"] = 0
    plotName = os.path.join(
        training_model.outdir, "injection_" + training_model.model + "_lightcurves.png"
    )
    sample_times = training_model.sample_times
    filters = training_model.filters

    if data_type == "photometry":
        mag = light_curve_model.generate_lightcurve(sample_times, data)
        lc_comparison_plot(mag, training["data"], filters, sample_times, plotName, 
                           ylabel_kwargs=dict( fontsize=30, rotation=0, labelpad=14))

    elif data_type == "spectroscopy":
        spec = light_curve_model.generate_spectra(
            sample_times, training_model.filters, data)
        
        training_data =np.log10(training["data"])
        interpolated_data = np.log10(np.array([spec[key] for key in filters]))
        residual = interpolated_data - training_data
        norm_residual = np.log10(np.abs(residual/training_data))
        plot_entries = {"Original": training_data.T, 
                            "Interpolated": interpolated_data,
                        "(Original - Interpolated) / Interpolated": norm_residual.T}
        def spec_plot_func(fig, ax, XX, YY, plot_data, label):
            if label == "(Original - Interpolated) / Interpolated":
                vmin = -3
                vmax = 0
                cbar_label = "Relative Difference"
            else:
                vmin = -10
                vmax = -2
                cbar_label = "log10(Flux)"
            return spec_subplot(fig, ax, XX, YY, plot_data, label, vmin=vmin, vmax=vmax, cbar_label=cbar_label)
        
        basic_spec_plot(
            mesh_X=sample_times,
            mesh_Y=filters,
            spec_func = spec_plot_func,
            plot_entries=plot_entries,
            save_path=plotName,
            figsize=(32, 14))  

def chi2_hists_from_dict(chi2_dict, outpath):
    matplotlib.rcParams.update(
        {"font.size": 16, "font.family": "Serif"}
    )
    for filt, chi2_array in chi2_dict.items():
        plt.figure()
        plt.xlabel(r"$\chi^2 / {\rm d.o.f.}$")
        plt.ylabel("Count")
        plt.hist(chi2_array, label=filt, bins=51, histtype="step")
        plt.legend()
        plt.savefig(f"{outpath}/{filt}.pdf", bbox_inches="tight")
        plt.close()

def plot_benchmark_percentiles( model, model_benchmarks, outdir):
        fig, ax = plt.subplots(figsize=(12, 8))

        filts = list(model_benchmarks.keys())
        filter_benchmarks = list(model_benchmarks.values())

        pctls_25 = [x[1] for x in filter_benchmarks]
        pctls_50 = [x[2] for x in filter_benchmarks]
        pctls_75 = [x[3] for x in filter_benchmarks]
        pctls_100= [x[4] for x in filter_benchmarks]
        plt.tight_layout()
        ax.bar(filts, pctls_75, label="<= 75th", color="red", hatch="\\")
        ax.bar(filts, pctls_50, label="<= 50th", color="slategray")
        ax.bar(filts, pctls_25, label="<= 25th", color="blue", hatch="/")
        ax.set_ylim(0, 1.1 * np.max(pctls_75))
        ax.tick_params(axis="x", labelrotation=75)
        ax.set_xlabel("Filter")
        ax.set_ylabel(r"Reduced $\chi^{2}$")
        ax.legend(
            title=f"Percentile\n (max $\chi^{2}$ = {np.round(np.max(pctls_100),1)})",  # noqa
            loc=2,
        )
        ax.set_title(f"{model} benchmark percentiles")
        plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.95)

        fig.savefig(f"{outdir}/benchmark_percentiles_{model}.pdf", bbox_inches="tight")

###################################################
################# PLOT STRUCTURES #################
###################################################

def basic_photo_lc_plot(
        plot_fc, filters, save_path, fontsize=30, figsize=(15, 18), colorbar=False, xlim = [0,14], ylim = [-12, -18], n_yticks = 4, ylabel_kwargs = dict(fontsize=30, rotation=90, labelpad=8),**kwargs):

    fig = plt.figure(figsize=figsize)
    ncols = 1
    nrows = int(np.ceil(len(filters) / ncols))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

    xlim = check_limit(xlim)
    ylim = check_limit(ylim)

    axs = []
    for ii, filt in enumerate(filters):
        loc_x, loc_y = np.divmod(ii, nrows)
        loc_x, loc_y = int(loc_x), int(loc_y)
        ax = fig.add_subplot(gs[loc_y, loc_x])

        ax, cb = plot_fc(ax, filt, ii)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if ii == len(filters) - 1:
            ax.set_xticks([np.ceil(x) for x in np.linspace(xlim[0], xlim[1], 8)])
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        ax.set_yticks([np.ceil(x) for x in np.linspace(ylim[0], ylim[1], n_yticks)])
        
        ax.set_ylabel(filt, **ylabel_kwargs)

        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.grid(which="both", alpha=0.5)
        axs.append(ax)

    if colorbar:
        fig.colorbar(cb, ax=axs, location="right", shrink=0.6)

    fig.text(0.45, 0.05, "Time [days]", fontsize=fontsize)
    ylabel = "Absolute Magnitude"

    fig.text(0.01, 0.5, ylabel, fontsize=fontsize,
        va="center", rotation="vertical")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def lc_comparison_plot(mag_dict, training_data, filters, sample_times, save_path,  **kwargs):
    def lc_plot_fc(ax, filt, ii):
        ax.plot(sample_times, training_data[:, ii], "k--", label="grid")
        ax.plot(sample_times, mag_dict[filt], "b-", label="interpolated")
        if ii == 0:
            ax.legend(fontsize=kwargs.get("fontsize", 30)//2)
        return ax, None
    
    basic_photo_lc_plot(lc_plot_fc, filters, save_path=save_path,  **kwargs)


def lc_plot_with_histogram(filters, data_dict, sample_times, save_path, percentiles = (10, 50, 90) , **kwargs):
    def lc_hist_fc(ax, filt, ii):
        plot_data = data_dict[filt]

        bins = np.linspace(-20, 30, 100)
        def return_hist(x):
            hist, _ = np.histogram(x, bins=bins)
            return hist

        hist = np.apply_along_axis(return_hist, -1, plot_data.T)
        bins = (bins[1:] + bins[:-1]) / 2.0

        X, Y = np.meshgrid(sample_times, bins)
        hist = hist.astype(np.float64)
        hist[hist == 0.0] = np.nan

        cb = ax.pcolormesh(X, Y, hist.T, shading="auto", cmap="cividis", alpha=0.7)
        # plot 10th, 50th, 90th percentiles
        for pct in percentiles:
            ax.plot(sample_times, np.nanpercentile(plot_data, pct, axis=0), "k--")

        return ax, cb
    basic_photo_lc_plot(lc_hist_fc, filters, save_path, **kwargs)

        
def spec_subplot(
        fig, ax, X, Y, Z, data_label, vmin =-10, vmax =-2, fontsize=30, cbar_label="log10(Flux)"
        ):
    pcm = ax.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax)
    ax.set_title(data_label, fontsize=fontsize)
    ax.set_xlabel("Time [days]", fontsize=fontsize)
    ax.set_ylabel("Wavelength [AA]", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(which="both", alpha=0.5)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    return fig, ax

def basic_spec_plot(
        mesh_X, mesh_Y, spec_func, plot_entries, save_path, figsize=(32, 14)):
    
    XX, YY = np.meshgrid(mesh_X, mesh_Y)
    fig = plt.figure(figsize=figsize)
    ncols = len(plot_entries)
    nrows = 1
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)
    for ii, (label, plot_data) in enumerate(plot_entries.items()):
        loc_x, loc_y = np.divmod(ii, nrows)
        loc_x, loc_y = int(loc_x), int(loc_y)
        ax = fig.add_subplot(gs[loc_x,loc_y])
        fig, ax = spec_func(fig, ax, XX, YY, plot_data, label)


    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


##############################################
############### HELPER FUNCTIONS #############
##############################################
def check_limit(lim):
    if isinstance(lim, str):
        lim = lim.split(",")
    lim = [float(val) for val in lim]
    assert len(lim) == 2, f"{lim} is no valid plot-limit." 
    return lim

def plot_observations(ax, transient, color="k",**kwargs):
    obs_times, obs_lc, obs_unc = transient.light_curve_times, transient.light_curves, transient.light_curve_uncertainties
    if 'filter' in kwargs:
        filt = kwargs.pop('filter')
        obs_lc = obs_lc[filt]
        obs_unc = obs_unc[filt]
        obs_times = obs_times[filt]
    # obs_times+= transient.trigger_time 
    detections = np.isfinite(obs_unc) ## does not include nans or infs
    ax.errorbar(obs_times[detections], obs_lc[detections], obs_unc[detections], fmt="o", color =color, **kwargs)

    non_detections = np.isinf(obs_unc) ## does only include +-inf, not nans
    ax.errorbar(obs_times[non_detections], obs_lc[non_detections], fmt="v", color=color, **kwargs)
    return ax, obs_times[detections]

def analysis_plot_geometry(filters_to_plot, ncol=2):
    # NOTE Should this be the preferred geometry for the plots?
    # set up the geometry for the all-in-one figure
    wspace = 0.6  # All in inches.
    hspace = 0.3
    lspace = 1.0
    bspace = 0.7
    trspace = 0.2
    hpanel = 2.25
    wpanel = 3.

    nrow = int(np.ceil(len(filters_to_plot) / ncol))

    figsize = (1.5 * (lspace + wpanel * ncol + wspace * (ncol - 1) + trspace),
                1.5 * (bspace + hpanel * nrow + hspace * (nrow - 1) + trspace))
    # Create the figure and axes.
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=lspace / figsize[0],
                        bottom=bspace / figsize[1],
                        right = 1. - trspace / figsize[0],
                        top =   1. - trspace / figsize[1],
                        wspace=wspace / wpanel,
                        hspace=hspace / hpanel)

    if len(filters_to_plot) % ncol:
        for i in range(len(filters_to_plot) % ncol, ncol):
            axes[-1, i-ncol].axis('off')
    return fig, axes



def get_time_limits_from_obs_data(transient):
    """
    A function that goes through the lc data and finds the time range that encompasses all data points.
    """

    xmin = np.min([t_arr.min() for t_arr in transient.light_curve_times.values()])
    xmax = np.max([t_arr.max() for t_arr in transient.light_curve_times.values()])

    return (0.9*xmin, 1.1*xmax)

def get_mag_limits_from_obs_data(transient, filter_names):
    """
    A function that goes through the lc data and finds the magnitude range for each filter.
    """

    ylim = {}
    for filt in filter_names:
        min_mag = transient.light_curves[filt].min()
        max_mag = transient.light_curves[filt].max()
        ylim[filt] = (min(1.05*max_mag, 1+ max_mag), max(min_mag-1, 0.95*min_mag))
    return ylim

