import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from scipy.special import softmax

# Times New Roman for every text of the plots: labels, numbers of the ticks, legend, mathematical symbols
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams["mathtext.cal"] = "Times New Roman:italic"
plt.rcParams["mathtext.fallback"] = "stix"   # symbols missing in Times New Roman (e.g. the sun symbol) from STIX, a font made to match Times

color_mock = '#6E1E63'
color_recovered = '#cdbac5'


# =================================================================================
# Recovered profile from the MCMC samples
# =================================================================================

def geometric_median(X, sample_size=None, seed=0):
    """
    Geometric median restricted to the samples: the sample that minimizes the sum of the distances from all the other samples
    """
    if sample_size is None:
        sample_size = X.shape[0]
    rng = np.random.default_rng(seed)
    subset = rng.choice(X.shape[0], size=sample_size, replace=False)
    sums = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        sums[i] = np.linalg.norm(X[subset] - X[i], axis=1).sum()
    return X[np.argmin(sums)]


# =================================================================================
# Earth mover distance (from analysis_ring_statistics.py)
# =================================================================================

def rebin_mock_on_bins(y_grid_pc, Sigma_y_mock, y_edges):
    """
    Gas column per bin of the mock profile, obtained by interpolating Sigma(y)
    at the bin edges. Same units as `Sigma_geometric_median`.
    """
    Sigma_at_edges = np.interp(y_edges, y_grid_pc, Sigma_y_mock)
    return np.diff(Sigma_at_edges)


def earth_mover_distance(Sigma_bins_mock, Sigma_bins_recovered, y_edges):
    """
    L1 distance between the two cumulative mass profiles, both rescaled to
    unit mass, so that the result is a distance in pc:
        EMD = int |F(y) - F_recovered(y)| dy
    """
    cdf_mock      = np.concatenate(([0.0], np.cumsum(Sigma_bins_mock      / Sigma_bins_mock.sum())))
    cdf_recovered = np.concatenate(([0.0], np.cumsum(Sigma_bins_recovered / Sigma_bins_recovered.sum())))
    return float(np.trapezoid(np.abs(cdf_mock - cdf_recovered), y_edges))


# =================================================================================
# Common parts of the two plots
# =================================================================================

def add_star_band(ax, y_stars_pc, M_bins, y_min, y_max):
    """
    Coloured band below the plot with the number of stars per bin, and its colorbar
    """
    counts, bin_edges = np.histogram(y_stars_pc, bins=M_bins, range=(y_min, y_max))
    cmap = plt.cm.Oranges
    norm_cb = Normalize(vmin=counts.min(), vmax=counts.max())
    band_height = 0.05   # height of the band, fraction of the height of the axis

    for i in range(len(counts)):
        ax.fill_between([bin_edges[i], bin_edges[i+1]], -band_height, 0,
                        transform=ax.get_xaxis_transform(), color=cmap(norm_cb(counts[i])), zorder=2, clip_on=False)

    # black frame around the band
    rect = Rectangle((y_min, -band_height), y_max - y_min, band_height, transform=ax.get_xaxis_transform(),
                     fill=False, edgecolor='black', linewidth=0.8, zorder=3, clip_on=False)
    ax.add_patch(rect)
    ax.tick_params(axis='x', which='both', direction='out', pad=6)
    ax.spines['bottom'].set_position(('axes', -band_height))

    sm = ScalarMappable(cmap=cmap, norm=norm_cb)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Number of stars per bin")
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def add_text(ax, n_stars, res, emd, y_sun, y_box, va_box):
    """
    To Sun (at height y_sun, fraction of the axis) and box (top or bottom right) with number of stars, resolution and EMD
    """
    ax.annotate("← To Sun", xy=(0.02, y_sun), xycoords='axes fraction', ha='left', va='bottom', fontsize=9, color='k')
    ax.text(0.97, y_box, f"{n_stars} stars\nResolution = {res:g} pc\nEMD = {emd:.1f} pc",
            transform=ax.transAxes, ha='right', va=va_box, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))


# =================================================================================
# Plots
# =================================================================================

def plot_results(path_backend, Sigma_tot, rho_mock, y_grid_pc, Sigma_y_mock, los_unit_pc, rho_scale,
                 y_edges, y_centers, res, y_stars_pc, n_stars, l_pc, b_pc,
                 y_min, y_max, path_plots, plot_label):
    """
    Plots of the mock (true) and recovered gas profile and of the cumulative mass along the los. Saved as pdf in path_plots:
    Gas_Profile_{plot_label}.pdf and Cumulative_mass_{plot_label}.pdf
    los_unit_pc: unit of length (in pc) of the gas columns, 1000 if they are in density*kpc (Ring), 1 if in density*pc (Fractal).
    Units of the plots: y in pc, density in M_sun pc^-3, cumulative mass in M_sun pc^-2.
    """
    M_bins = len(y_centers)
    res_los = res / los_unit_pc   # bin width in the unit of the gas columns
    title = rf"Mock profile vs recovered profile - $(\ell, b)$ = ({l_pc:g}, {b_pc:g}) pc"

    # ========= MCMC samples =============
    idata = az.from_netcdf(path_backend)
    vartheta_raw = idata.posterior["vartheta"].values
    vartheta_flat = vartheta_raw.reshape(-1, vartheta_raw.shape[-1])    # shape: (n_chains * n_steps, M_bins)
    Sigma_samples = Sigma_tot * softmax(vartheta_flat, axis=1)          # gas column of every bin, every sample
    Sigma_percentiles = np.percentile(Sigma_samples, [16, 84], axis=0)
    Sigma_geometric_median = geometric_median(Sigma_samples)

    # ========= EMD =============
    Sigma_bins_mock = rebin_mock_on_bins(y_grid_pc, Sigma_y_mock, y_edges)
    emd = earth_mover_distance(Sigma_bins_mock, Sigma_geometric_median, y_edges)

    # -------------------------------
    # Gas profile
    # -------------------------------
    rho_recovered = (Sigma_geometric_median / res_los) * rho_scale
    rho_16 = (Sigma_percentiles[0, :] / res_los) * rho_scale
    rho_84 = (Sigma_percentiles[1, :] / res_los) * rho_scale
    y_step = np.repeat(y_edges, 2)[1:-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y_grid_pc, rho_mock*rho_scale, lw=2.5, color=color_mock, zorder=4,   # above the bars
            path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()])
    ax.bar(y_centers, rho_recovered, width=res, align='center', color=color_recovered, alpha=0.9)
    ax.plot(y_step, np.repeat(rho_recovered, 2), color='k', lw=1.0)
    ax.fill_between(y_step, np.repeat(rho_16, 2), np.repeat(rho_84, 2),
                    facecolor='none', hatch='/////', edgecolor='k', linewidth=0.0, zorder=3)
    add_star_band(ax, y_stars_pc, M_bins, y_min, y_max)
    add_text(ax, n_stars, res, emd, 0.5, 0.97, 'top')

    handles = [Line2D([0], [0], color=color_mock, lw=2.5, path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()]),
               Patch(facecolor=color_recovered, edgecolor='k', alpha=0.9),
               Patch(facecolor='none', edgecolor='k', hatch='/////', linewidth=0.0)]
    ax.legend(handles, ["Mock profile", "Recovered profile", "16-84 percentiles"], loc='upper left', fontsize=10)
    ax.set_xlabel("Position on line of sight [pc]")
    ax.set_ylabel(r"Density [M$_\odot$ pc$^{-3}$]")
    ax.set_xlim(y_min, y_max)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_plots/f"Gas_Profile_{plot_label}.pdf")
    plt.close()

    # -------------------------------
    # Cumulative mass
    # -------------------------------
    dy = y_grid_pc[1] - y_grid_pc[0]
    mass_cum_mock = np.cumsum(rho_mock * dy) * rho_scale                                  # M_sun pc^-2
    mass_cum_recovered = np.cumsum(Sigma_geometric_median * los_unit_pc) * rho_scale       # M_sun pc^-2, mass up to the right edge of every bin
    mass_cum_samples = np.cumsum(Sigma_samples * los_unit_pc, axis=1) * rho_scale          # cumulative mass of every sample
    mass_cum_percentiles = np.percentile(mass_cum_samples, [16, 84], axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y_grid_pc, mass_cum_mock, lw=7, color=color_mock,
            path_effects=[pe.Stroke(linewidth=9, foreground='k'), pe.Normal()])
    ax.fill_between(y_edges[1:], mass_cum_percentiles[0, :], mass_cum_percentiles[1, :],
                    facecolor='none', hatch='/////', edgecolor='k', linewidth=0.0, zorder=3)
    ax.plot(y_edges[1:], mass_cum_recovered, linestyle='-', lw=1, color='k',
            marker='o', ms=10, markerfacecolor=color_recovered, markeredgecolor='k', zorder=4)
    add_star_band(ax, y_stars_pc, M_bins, y_min, y_max)
    add_text(ax, n_stars, res, emd, 0.1, 0.05, 'bottom')

    handles = [Line2D([0], [0], color=color_mock, lw=7, path_effects=[pe.Stroke(linewidth=9, foreground='k'), pe.Normal()]),
               Line2D([0], [0], linestyle='-', lw=1, color='k', marker='o', ms=10, markerfacecolor=color_recovered, markeredgecolor='k'),
               Patch(facecolor='none', edgecolor='k', hatch='/////', linewidth=0.0)]
    ax.legend(handles, ["Mock cumulative mass", "Recovered cumulative mass", "16-84 percentiles"], loc='upper left', fontsize=10)
    ax.set_xlabel("Position on line of sight [pc]")
    ax.set_ylabel(r"Cumulative mass [M$_\odot$ pc$^{-2}$]")
    ax.set_xlim(y_min, y_max)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path_plots/f"Cumulative_mass_{plot_label}.pdf")
    plt.close()

    return Sigma_geometric_median, Sigma_percentiles, emd
