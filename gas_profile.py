import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

# =================================================================================
# Gas profile of the Ring50 model
# =================================================================================

def ring(y_min, y_max, l_pc, b_pc, rho_0=3.0, R_R=0.05, W_R=0.01, H_R=0.02, **kwargs):
    """
    Mock gas density of the Ring50 model along the los (l_pc, b_pc), on a fine grid.
    y is the coordinate along the los, x the one along l, z the one along b (as in the paper). Lengths of the ring in kpc.
    """
    y_grid_pc = np.linspace(y_min, y_max, 1000)
    y_grid_kpc = np.sort(y_grid_pc * 0.001) # kpc

    x_kpc = l_pc*0.001  # kpc
    z_kpc = b_pc*0.001  # kpc

    R = np.sqrt(y_grid_kpc**2 + x_kpc**2)
    rho_ring = rho_0 * np.exp(-((R - R_R)**2) / (2 * W_R**2)) * np.exp(-z_kpc**2 / (2 * H_R**2))
    return rho_ring, y_grid_kpc


# =================================================================================
# sigma_fov: dispersion of the gas column inside the cell (from SigmaObs_Ring.py)
# =================================================================================

def plot_cell(Sigma_sub_cells, sub_cells_l, sub_cells_b, l_pc, b_pc, sigma_perc, path_cell_plot):
    """
    Plot of the cell divided into sub-cells, each coloured with its gas column (M_sun pc^-2).
    Used also for the Fractal (create_fractal.py).
    """
    d = sub_cells_l[1] - sub_cells_l[0]   # distance between two sub-cells, pc
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Sigma_sub_cells.T, origin='lower', aspect='equal', cmap='viridis',
                   extent=[sub_cells_l[0]-d/2, sub_cells_l[-1]+d/2, sub_cells_b[0]-d/2, sub_cells_b[-1]+d/2])
    plt.colorbar(im, ax=ax, label=r"Gas column of the sub-cell [M$_\odot$ pc$^{-2}$]")
    ax.text(0.97, 0.97, f"std/mean = {sigma_perc:.2f} %", transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.85))
    ax.set_xlabel(r"$\ell$ [pc]")
    ax.set_ylabel(r"$b$ [pc]")
    ax.set_title(rf"Gas column in the cell - $(\ell, b)$ = ({l_pc:g}, {b_pc:g}) pc")
    plt.tight_layout()
    plt.savefig(path_cell_plot)
    plt.close()


def compute_mean_and_sigma(gas_density, params, l_pc, b_pc, rho_scale, path_mean_and_sigma, path_cell_plot,
                           cell_size_pc=10.0, n_sub_cells_side=100):
    """
    For the cell centred on (l_pc, b_pc) the gas column is computed on a regular grid of n_sub_cells_side x n_sub_cells_side
    sub-cells inside the cell (cell of cell_size_pc x cell_size_pc).
    Saves:
    - one row (l_pc, b_pc, mean of the column, std of the column, std/mean*100)
    - a plot of the cell with the gas column of every sub-cell (M_sun pc^-2, normalized with rho_scale)
    """
    params_sub_cell = dict(params)   # copy, so that the params of the main are not modified

    sub_cells_l = np.linspace(l_pc - cell_size_pc/2, l_pc + cell_size_pc/2, n_sub_cells_side)
    sub_cells_b = np.linspace(b_pc - cell_size_pc/2, b_pc + cell_size_pc/2, n_sub_cells_side)
    grid_l, grid_b = np.meshgrid(sub_cells_l, sub_cells_b, indexing="ij")
    sub_cells = np.column_stack([grid_l.ravel(), grid_b.ravel()])

    Sigma_tot_sub_cells = np.zeros([len(sub_cells)])
    for i in range(len(sub_cells)):
        params_sub_cell["l_pc"] = sub_cells[i, 0]
        params_sub_cell["b_pc"] = sub_cells[i, 1]
        rho, y_grid_kpc = gas_density(**params_sub_cell)
        dy = y_grid_kpc[1] - y_grid_kpc[0]
        Sigma_tot_sub_cells[i] = np.sum(rho * dy)   # same as the last value of the cumulative integral

    mean_column = np.mean(Sigma_tot_sub_cells)
    std_column = np.std(Sigma_tot_sub_cells, ddof=1)
    results = np.array([(l_pc, b_pc, mean_column, std_column, std_column / mean_column * 100)])
    np.save(path_mean_and_sigma, results)

    print(f"Sigma of the cell ({l_pc:g}, {b_pc:g}) pc: {std_column / mean_column * 100:.2f} %  ({len(sub_cells)} sub-cells)")

    # Plot of the cell: gas column of every sub-cell (density*kpc -> M_sun pc^-2)
    Sigma_sub_cells = Sigma_tot_sub_cells.reshape(n_sub_cells_side, n_sub_cells_side) * 1000.0 * rho_scale
    plot_cell(Sigma_sub_cells, sub_cells_l, sub_cells_b, l_pc, b_pc, std_column / mean_column * 100, path_cell_plot)

    return results


# =================================================================================
# Normalization of the Ring to the mass of the CMZ
# =================================================================================

def compute_norm_ring(params, M_CMZ):
    """
    Factor rho_scale that rescales the density of the ring, computed as for the figures of the paper.
    Units of the result: M_sun pc^-3 per unit of density of the ring.
    NOTE: dblquad calls the function as f(y, x), with y the inner variable: here R takes the values in [-5H_R, 5H_R]
    and z those in [0, R_R+5W_R]. The integral covers only the half of the ring with z >= 0, so the rescaled ring
    has a total mass of 2 M_CMZ (as in the figures of the paper).
    """
    rho_0 = params["rho_0"]
    R_R = params["R_R"] * 1000.0   # pc
    W_R = params["W_R"] * 1000.0   # pc
    H_R = params["H_R"] * 1000.0   # pc

    def ring_cylindrical(R, z):
        return rho_0 * np.exp(-((R - R_R)**2) / (2 * W_R**2)) * np.exp(-z**2 / (2 * H_R**2)) * R

    integral, err = dblquad(ring_cylindrical, 0, R_R + 5*W_R, lambda R: -5*H_R, lambda R: 5*H_R)
    m_tot = 2 * np.pi * integral
    return M_CMZ / m_tot
