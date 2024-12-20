import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
}


matplotlib.rcParams.update(params)
def check_limit(lim):
    if isinstance(lim, str):
        lim = [float(val) for val in lim.split(",")]
    assert len(lim) == 2, f"{lim} is no valid plot-limit." 
    return lim

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
