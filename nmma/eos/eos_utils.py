import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from nmma.core.conversion import ascending, interpolate_on_mass_nodes


@dataclass
class TOVDataset:
    params: np.ndarray  # (n_samples, n_params)
    mass: np.ndarray  # (n_samples, max_curve_length), M_sun
    radius: np.ndarray  # km
    lam: np.ndarray  # dimensionless tidal deformability

    def __len__(self):
        return len(self.params)

    def subset(self, indices):
        return TOVDataset(
            self.params[indices],
            self.mass[indices],
            self.radius[indices],
            self.lam[indices],
        )

    def fingerprint(self):
        "Short hash to identify the data a model was trained on"
        digest = hashlib.sha1(np.ascontiguousarray(self.params).tobytes())
        digest.update(str(self.mass.shape).encode())
        return digest.hexdigest()[:12]

    def prepare_targets(self, grid, mtov_range=(2.0, 3.0)):
        """Interpolate the raw curves onto the mass grid.
        Returns the dataset indices of usable samples and the radius [km], ln(Lambda)
        and M_tov targets of those samples."""
        n_samples = len(self)
        radius = np.empty((n_samples, grid.n_points))
        ln_lam = np.empty((n_samples, grid.n_points))
        m_tov = np.full(n_samples, np.nan)
        self.rejected = {
            "invalid or non-monotonic curve": 0,
            "M_tov outside range": 0,
            f"curve starts above m_min={grid.m_min}": 0,
        }
        for i in range(n_samples):
            branch = self.stable_branch(i)
            if branch is None:
                self.rejected["invalid or non-monotonic curve"] += 1
                continue
            m, r, l = branch
            if not mtov_range[0] <= m[-1] <= mtov_range[1]:
                self.rejected["M_tov outside range"] += 1
                continue
            if m[0] > grid.m_min:
                self.rejected[f"curve starts above m_min={grid.m_min}"] += 1
                continue
            radius[i], ln_lam[i] = grid.resample(m, r, np.log(l))
            m_tov[i] = m[-1]

        good = np.flatnonzero(np.isfinite(m_tov))
        print(f"Prepared targets for {len(good)}/{n_samples} samples")
        print(f"Rejected: {self.rejected}")
        targets = np.hstack([radius[good], ln_lam[good], m_tov[good][:, None]])
        return good, targets

    def stable_branch(self, i):
        """Valid part of the raw solver curve of sample i up to the maximum mass, as (m, r, lam),
        or None if unusable"""
        mass, radius, lam = self.mass[i], self.radius[i], self.lam[i]
        valid = np.isfinite(mass) & np.isfinite(radius) & np.isfinite(lam) & (mass > 0)
        m, r, l = mass[valid], radius[valid], lam[valid]
        if len(m) < 2:
            return None
        i_max = np.argmax(m)
        m, r, l = m[: i_max + 1], r[: i_max + 1], l[: i_max + 1]
        if not ascending(m) or np.any(l <= 0) or np.any(r <= 0):
            return None
        return m, r, l

    def reference_values(self, ref_masses, order=2):
        """R and ln(Lambda) of the raw solver curves at the reference masses (NaN
        outside the stable branch), interpolated in sqrt(M_tov - m) by polynomials
        of ``order``, see ``interpolate_on_mass_nodes``."""
        R = np.full((len(self), len(ref_masses)), np.nan)
        ln_L = np.full_like(R, np.nan)
        for i in range(len(self)):
            branch = self.stable_branch(i)
            if branch is None:
                continue
            m, r, l = branch
            R[i], ln_L[i] = interpolate_on_mass_nodes(
                ref_masses, m, r, np.log(l), order=order
            )
        return R, ln_L

    def test_vs_generator(self, generator, ref_masses=(1.4, 1.6, 1.8, 2.0)) -> dict:
        """Compare the curves of an EoSGenerator for ``params`` (in generator input
        order) with the raw curves of this dataset at ``ref_masses``.

        Returns
        -------
        dict
            Per-sample errors at each reference mass ('diff': R [km] and ln Lambda,
            'relative', 'abs_rel' for R and Lambda), true and predicted TOV masses
            and a 'summary' of the absolute relative errors (median, 95th percentile).
        """
        ref_masses = np.asarray(ref_masses, dtype=float)
        curves, m_tov_pred = generator.predict_curves(self.params)
        predictions = generator.mass_grid.interpolate(ref_masses, m_tov_pred, curves)
        truths = self.reference_values(ref_masses)

        characteristics = {mode: {} for mode in ("diff", "relative", "abs_rel")}
        summary = {}
        for j, mass in enumerate(ref_masses):
            key = str(mass)
            for mode in characteristics:
                characteristics[mode][key] = {}
            summary[key] = {}
            for obs, pred, truth in zip(("rad", "lam"), predictions, truths):
                relative = relative_error(obs, pred[:, j], truth[:, j])
                characteristics["diff"][key][obs] = pred[:, j] - truth[:, j]
                characteristics["relative"][key][obs] = relative
                characteristics["abs_rel"][key][obs] = np.abs(relative)
                summary[key][obs] = {"model": error_summary(relative, truth[:, j])}

        m_tov_true = np.array([np.nanmax(mass) for mass in self.mass])
        mtov_error = np.abs(m_tov_pred - m_tov_true)
        summary["m_tov"] = {
            "median_abs": float(np.nanmedian(mtov_error)),
            "p95_abs": float(np.nanpercentile(mtov_error, 95)),
            "max_abs": float(np.nanmax(mtov_error)),
        }
        characteristics.update(
            {"m_tov": m_tov_true, "m_tov_pred": m_tov_pred, "summary": summary}
        )
        return characteristics


def relative_error(obs, pred, truth):
    "Relative error of R, or of Lambda from ln(Lambda) values"
    if obs == "lam":
        return np.expm1(pred - truth)
    return (pred - truth) / truth


def error_summary(relative, truth):
    """Median, 95th percentile and max of the absolute relative error.
    'missing' counts samples where a true value exists but the prediction
     is NaN (predicted M_tov below the mass)."""
    rel = np.abs(relative[np.isfinite(truth)])
    missing = int(np.sum(~np.isfinite(rel)))
    rel = rel[np.isfinite(rel)]
    if not len(rel):
        return {"median": None, "p95": None, "max": None, "missing": missing, "n": 0}
    return {
        "median": float(np.median(rel)),
        "p95": float(np.percentile(rel, 95)),
        "max": float(np.max(rel)),
        "missing": missing,
        "n": int(len(rel)),
    }


def plot_performance(
    generator, test_data, characteristics, n_show=40, ref_mass="1.4", save=None, seed=0
):
    """Overview of the performance of an EoSGenerator on a test set.

    Top row: M-R and M-ln(Lambda) curves of n_show random test samples (lines: solver,
    stars: generator on its mass grid; top axis of the Lambda panel in Lambda) and the signed
    relative M_tov error.
    Bottom row: signed relative errors of R and Lambda at ref_mass (positive up, negative
    down) with median and 95th percentile of the emulator error and the medians of grid floor
    and PCA floor, and a legend of these lines. All error panels state median and 95th percentile.

    Parameters:
    generator: EoSGenerator
        with a mass grid; its parameters are the columns of test_data.params
    test_data: TOVDataset
        test samples in the order used for `characteristics`
    characteristics: dict
        output of TOVDataset.test_vs_generator or TOVEmulatorTrainer.evaluate
    ref_mass: str
        key of the reference mass in characteristics
    save: str or Path, optional
        file to store the figure in (e.g. a .pdf); the figure is closed afterwards
    """
    ref_mass = str(ref_mass)
    rng = np.random.default_rng(seed)
    show = np.sort(
        rng.choice(len(test_data), min(n_show, len(test_data)), replace=False)
    )
    R_pred, ln_L_pred, m_tov_pred = generator(
        generator.parameter_dict(test_data.params[show])
    )
    m_pred = generator.mass_grid.masses(m_tov_pred)
    m_min = generator.mass_grid.m_min

    m_tov = np.asarray(characteristics["m_tov"], dtype=float)
    mtov_rel = (np.asarray(characteristics["m_tov_pred"], dtype=float) - m_tov) / m_tov
    summary = characteristics.get("summary", {})

    with _plot_style():
        fig, axs = plt.subplots(
            2, 3, figsize=(15, 8), width_ratios=[1, 1, 0.3], height_ratios=[3, 1]
        )
        ax_r, ax_l, ax_m, ax_rh, ax_lh, ax_text = axs.flatten()

        for j, i in enumerate(show):
            colour = plt.get_cmap("tab20")(j % 20)
            branch = test_data.stable_branch(i)
            if branch is not None:
                m, r, l = branch
                keep = m >= m_min - 0.05
                ax_r.plot(r[keep], m[keep], color=colour)
                ax_l.plot(np.log(l[keep]), m[keep], color=colour)
            ax_r.plot(R_pred[j], m_pred[j], "*", color=colour, markersize=4)
            ax_l.plot(ln_L_pred[j], m_pred[j], "*", color=colour, markersize=4)

        m_top = max(np.max(m_pred), np.max(m_tov[show])) + 0.05
        ax_r.set_ylim([m_min, m_top])
        ax_r.set_ylabel(r"Mass [$M_{\odot}$]")
        ax_r.set_xlabel("Radius [km]")
        ax_l.set_ylim(ax_r.get_ylim())
        ax_l.set_yticklabels([])
        ax_l.set_xlabel(r"$\ln\Lambda$")
        # corresponding Lambda values on a log scale at the top
        ax_lam = ax_l.secondary_xaxis("top", functions=(np.exp, np.log))
        ax_lam.set_xscale("log")
        ax_lam.set_xlabel(r"$\Lambda$", labelpad=10)

        # signed M_tov error, horizontal histogram as in the old routine
        mtov_abs = np.abs(mtov_rel)
        bins = np.geomspace(max(min(mtov_abs), 1e-7), max(mtov_abs), 30)
        hist_kwargs = dict(bins=bins, alpha=0.5, color="blue", orientation="horizontal")
        plus, minus = mtov_rel[mtov_rel >= 0], np.abs(mtov_rel[mtov_rel < 0])
        ax_m.hist(plus, **hist_kwargs)
        ax_m.hist(minus, weights=-np.ones_like(minus), **hist_kwargs)
        ax_m.axvline(0.0, color="black", linewidth=0.8)
        ax_m.set_ylabel(r"Relative $M_{\rm TOV}$ error")
        ax_m.set_yscale("log")
        ax_m.yaxis.set_label_position("right")
        ax_m.yaxis.tick_right()
        ax_m.tick_params(
            axis="both", which="both", bottom=False, top=False, left=False, right=True
        )
        ax_m.set_xticklabels([])
        median, p95 = np.percentile(mtov_abs, [50, 95])
        ax_m.axhline(median, color="black", linestyle="-", linewidth=1.2)
        ax_m.axhline(p95, color="black", linestyle="-.", linewidth=1.2)
        _error_levels_text(ax_m, median, p95, x=0.5, ha="center", fontsize=11)

        for ax, obs, label in [
            (ax_rh, "rad", rf"$R_{{{ref_mass}}}$"),
            (ax_lh, "lam", rf"$\Lambda_{{{ref_mass}}}$"),
        ]:
            relative = np.asarray(
                characteristics["relative"][ref_mass][obs], dtype=float
            )
            _plot_signed_distribution(ax, relative, summary.get(ref_mass, {}).get(obs))
            ax.set_xlabel(f"Relative {label} error")
        ax_rh.set_ylabel(r"Count [a.u.]")
        ax_lh.set_yticklabels([])

        _line_legend(ax_text)

        plt.tight_layout()
        if save:
            fig.savefig(Path(save), bbox_inches="tight", dpi=300)
            plt.close(fig)
    return fig


def _plot_signed_distribution(ax, relative, obs_summary=None, n_bins=40):
    """Log-binned |relative error|, positive errors upwards and negative errors downwards,
    normalised to the larger maximum (as plot_distribution(mode='rel') in nep_emulator_tools).
    The range follows the errors (0.1th percentile to maximum, extended to show the floor
    medians); median and 95th percentile are drawn and written out."""
    relative = relative[np.isfinite(relative)]
    absolute = np.abs(relative)
    floors = [
        obs_summary[name]["median"]
        for name in ("grid_floor", "pca_floor")
        if obs_summary and name in obs_summary and obs_summary[name]["median"]
    ]
    low = min([np.percentile(absolute[absolute > 0], 0.1), *floors]) / 1.5
    high = absolute.max() * 1.5
    bins = np.geomspace(low, high, n_bins)

    plus, _ = np.histogram(relative[relative >= 0], bins)
    minus, _ = np.histogram(-relative[relative < 0], bins)
    norm = max(plus.max(), minus.max(), 1)
    ax.stairs(plus / norm, bins, fill=True, alpha=0.5, color="blue")
    ax.stairs(-minus / norm, bins, fill=True, alpha=0.5, color="blue")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(axis="y", which="both", left=False, right=False)

    median, p95 = np.percentile(absolute, [50, 95])
    ax.axvline(median, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(p95, color="black", linestyle="-.", linewidth=1.2)
    if obs_summary:
        for name, style in [("grid_floor", ":"), ("pca_floor", "--")]:
            if name in obs_summary and obs_summary[name]["median"]:
                ax.axvline(
                    obs_summary[name]["median"],
                    color="black",
                    linestyle=style,
                    linewidth=1.2,
                )
    _error_levels_text(ax, median, p95)


def _percentile_label(percentile):
    return (
        rf"{percentile}\textsuperscript{{th}}"
        if matplotlib.rcParams["text.usetex"]
        else f"{percentile}th"
    )


def _error_levels_text(ax, median, p95, x=0.02, ha="left", fontsize=14):
    "Median and 95th percentile written into the upper part of an error panel"
    text = f"median: {median:.1e}\n{_percentile_label(95)}: {p95:.1e}"
    ax.text(
        x,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha=ha,
        fontsize=fontsize,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
    )


def _line_legend(ax, fontsize=16):
    "Meaning of the line styles in the error panels"
    ax.set_axis_off()
    handles = [
        Line2D([], [], color="black", linestyle=style, linewidth=1.2, label=label)
        for style, label in [
            ("-", "median"),
            ("-.", f"{_percentile_label(95)} percentile"),
            (":", "grid floor (median)"),
            ("--", "PCA floor (median)"),
        ]
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=fontsize,
        handlelength=1.8,
        borderaxespad=0.0,
    )


def _plot_style():
    "Plot style of the old routine, without changing global rcParams; LaTeX only if available"
    rc = {
        "font.size": 22,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
    }
    if shutil.which("latex"):
        rc["text.usetex"] = True
    return matplotlib.rc_context(rc)


def print_summary(summary):
    "Table of the error summary of TOVDataset.test_vs_generator / TOVEmulatorTrainer.evaluate"
    names = ("model", "grid_floor", "pca_floor")
    print(
        "Absolute relative errors (median / 95th percentile) against raw solver curves:"
    )
    print(
        f"{'mass':>6} {'obs':>4} {'model':>22} {'grid floor':>22} {'PCA floor':>22} {'missing':>8}"
    )
    for key, entry in summary.items():
        if key in ("m_tov", "timing_us"):
            continue
        for obs, values in entry.items():
            cols = []
            for name in names:
                s = values.get(name)
                cols.append(
                    f"{s['median']:.2e} / {s['p95']:.2e}" if s and s["n"] else "n/a"
                )
            print(
                f"{key:>6} {obs:>4} {cols[0]:>22} {cols[1]:>22} {cols[2]:>22} {values['model']['missing']:>8}"
            )
    m = summary["m_tov"]
    print(
        f"M_tov absolute error [M_sun]: median {m['median_abs']:.2e}, 95% {m['p95_abs']:.2e}, max {m['max_abs']:.2e}"
    )
    if "timing_us" in summary:
        print("Single-sample call times:")
        for name, value in summary["timing_us"].items():
            print(f"{name:>30}: {value:10.1f} us per call")
