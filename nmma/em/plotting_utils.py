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

def  lc_plot(filters, data_dict, sample_times, plotpath, percentiles = (10, 50, 90), 
             xlim = [0,14], ylim = [-12, -18], n_yticks =8,colorbar= False,
             ylabel_kwargs = dict(fontsize=30, rotation=90, labelpad=8),  **fig_kwargs):
    

    fig = plt.figure()
    ncols = 1
    nrows = len(filters)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.6, hspace=0.5)

    axs = []
    for ii, filt in enumerate(filters):
        loc_x, loc_y = np.divmod(ii, nrows)
        loc_x, loc_y = int(loc_x), int(loc_y)
        ax = fig.add_subplot(gs[loc_y, loc_x])

        plot_data = data_dict[filt]

        bins = np.linspace(-20, 30, 100)
        def return_hist(x):
            hist, bin_edges = np.histogram(x, bins=bins)
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
        xlim = check_limit(xlim)
        ax.set_xlim(xlim)
        ylim = check_limit(ylim)
        ax.set_ylim(ylim)

        ax.set_ylabel(filt, ylabel_kwargs)

        if ii == len(filters) - 1:
            ax.set_xticks([np.ceil(x) for x in np.linspace(xlim[0], xlim[1], 8)])
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        ax.set_yticks([np.ceil(x) for x in np.linspace(ylim[0], ylim[1], n_yticks)])
        ax.tick_params(axis="x", labelsize=35)
        ax.tick_params(axis="y", labelsize=35)
        ax.grid(which="both", alpha=0.5)

        axs.append(ax)

    if colorbar:
        fig.colorbar(cb, ax=axs, location="right", shrink=0.6)
    fig.text(0.42, 0.05, r"Time [days]", fontsize=36)
    fig.text(
        0.01,
        0.5,
        r"Absolute Magnitude",
        va="center",
        rotation="vertical",
        fontsize=36,
    )

    plt.tight_layout()
    plt.savefig(plotpath, bbox_inches="tight")
    plt.close()

def check_limit(lim):
    if isinstance(lim, str):
        lim = [float(val) for val in lim.split(",")]
    assert len(lim) == 2, f"{lim} is no valid plot-limit." 
    return lim