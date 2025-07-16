import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

matplotlib.use("agg")
params = {
    "backend": "pdf",
    "axes.labelsize": 42,
    "legend.fontsize": 42,
    "xtick.labelsize": 42,
    "ytick.labelsize": 42,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": [18, 25],
    "font.size": 16,
}
matplotlib.rcParams.update(params)

##############################################
################# MAIN PLOTS #################
##############################################
def basic_em_analysis_plot(
        transient, mags_to_plot, error_dict, chi2_dict, mismatches,
        sub_model_plot_props, xlim, ylim, save_path,
        ncol = 2):


    xlim = check_limit(xlim)
    ylim = check_limit(ylim)

    time = mags_to_plot.pop("time")
    filter_names = list(mags_to_plot.keys())

    fig, axes = analysis_plot_geometry(filter_names, ncol=ncol)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(filter_names)))[::-1]
    for cnt, filt in enumerate(filter_names):

        # summary plot
        row, col = divmod(cnt, ncol)
        ax_sum = axes[row, col]
        # adding the ax for the Delta
        divider = make_axes_locatable(ax_sum)
        ax_delta = divider.append_axes('bottom',
                                        size='30%',
                                        sharex=ax_sum)

        # configuring ax_sum
        ax_sum.set_ylabel("AB magnitude", rotation=90)
        ax_delta.set_ylabel(r"$\Delta (\sigma)$")
        if cnt == len(filter_names)-1:
            ax_delta.set_xlabel("Time [days]")
        else:
            ax_delta.set_xticklabels([])

        # plot the observations
        ax_sum, det_times = plot_observations(ax_sum, transient, colors[cnt], filter=filt)

        # plot the mismatch between the model and the data
        diff_per_data, sigma_per_data = mismatches[filt]
        ax_delta.axhline(0, linestyle='--', color='k')
        ax_delta.scatter(det_times, diff_per_data, # / sigma_per_data,  # FIXME: Bug?
                         color=colors[cnt])

        # plot the best-fit lc with errors
        mag_plot = mags_to_plot[filt]
        error_budget = error_dict[filt]
        label = 'combined' if sub_model_plot_props is not None else ""

        ax_sum.plot(time, mag_plot,
            color='coral', linewidth=3, linestyle="--")
        ax_sum.fill_between(time,
            mag_plot + error_budget,
            mag_plot - error_budget,
            facecolor='coral',
            alpha=0.2,
            label=label,
            )

        if sub_model_plot_props is not None:
            ## plot additional lcs for each sub_model
            for model_name, prop_dict in sub_model_plot_props:
                mag_plot = prop_dict['plot_mags'][cnt]
                plot_times = prop_dict['plot_times'][cnt]
                ax_sum.plot(plot_times, mag_plot,
                    color='coral', linewidth=3, linestyle="--")
                ax_sum.fill_between(plot_times,
                    mag_plot + error_budget,
                    mag_plot - error_budget,
                    facecolor=prop_dict['color'],
                    alpha=0.2,
                    label=model_name,
                )

        ax_sum.set_title(f'{filt}: ' + fr'$\chi^2 / d.o.f. = {round(chi2_dict[filt], 2)}$')
        ax_sum.set_xlim(xlim)
        ax_sum.set_ylim(ylim)
        # ax_delta.set_xlim(xlim)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def bolometric_lc_plot(transient, lbol_dict, save_path, color = "coral"):
    matplotlib.rcParams.update(
        {'font.size': 12,
        'text.usetex': True,
        'font.family': 'Times New Roman'}
    )
    fig, ax = plt.subplots(1, 1)
    ax, _ = plot_observations(ax, transient, markersize=12)

    ### plot the bestfit model
    ax.plot(lbol_dict['time'], lbol_dict['lbol'],
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
        {"font.size": 16, "text.usetex": True, "font.family": "Times New Roman"}
    )
    for filt, chi2_array in chi2_dict.items():
        plt.figure()
        plt.xlabel(r"$\chi^2 / {\rm d.o.f.}$")
        plt.ylabel("Count")
        plt.hist(chi2_array, label=filt, bins=51, histtype="step")
        plt.legend()
        plt.savefig(f"{outpath}/{filt}.pdf", bbox_inches="tight")
        plt.close()

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

        hist = np.apply_along_axis(lambda x: return_hist(x), -1, plot_data.T)
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
        lim = [float(val) for val in lim.split(",")]
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
    fig, axes = plt.subplots(nrow, ncol)

    figsize = (1.5 * (lspace + wpanel * ncol + wspace * (ncol - 1) + trspace),
                1.5 * (bspace + hpanel * nrow + hspace * (nrow - 1) + trspace))
    # Create the figure and axes.
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    fig.subplots_adjust(left=lspace / figsize[0],
                        bottom=bspace / figsize[1],
                        right=1. - trspace / figsize[0],
                        top=1. - trspace / figsize[1],
                        wspace=wspace / wpanel,
                        hspace=hspace / hpanel)

    if len(filters_to_plot) % 2:
        axes[-1, -1].axis('off')
    return fig, axes

