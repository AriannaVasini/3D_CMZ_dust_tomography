import numpy as np
import matplotlib.pyplot as plt
from gas_profile import plot_cell


def fractal_ISM(y_min, y_max, n_grid_fractal, alpha, seed, **kwargs):
    """
    3D fractal field F(x) of the paper: F = 1 + [M - 1] = M, with M = exp(G - sigma_fractal^2/2)
    and G a random-phase field with power spectrum P(k) ~ k^-alpha (mean 0, variance 1).
    """
    N = n_grid_fractal
    L = np.abs(y_max-y_min)
    dx = L / N
    dy = dx
    dz = dx

    kx = np.fft.fftfreq(N, d=dx)   # length N with step 1/L
    ky = np.fft.fftfreq(N, d=dy)   # length N with step 1/L
    kz = np.fft.fftfreq(N, d=dz)

    # k_mod computed without the three full meshgrids (same result, less memory)
    k_mod = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)

    # power spectrum:
    P_k = np.zeros((N, N, N))
    mask = (k_mod != 0)
    P_k[mask] = k_mod[mask]**(-alpha)
    del k_mod, mask   # free memory

    # amplitude
    A_k = np.sqrt(P_k)
    del P_k   # free memory

    # random phases: fixed seed, so that the same fractal is generated every time
    rng = np.random.RandomState(seed)
    phi = rng.uniform(0, 2*np.pi, size=(N, N, N))

    A_k[0,0,0] = 0.0
    phi[0,0,0] = 0.0
    
    if N % 2 == 0:
        phi[N//2, :, :] = 0.0
    if N % 2 == 0:
        phi[:, N//2, :] = 0.0
    if N % 2 == 0:
        phi[:, :, N//2] = 0.0

    G_tilde = A_k * np.exp(1j * phi)
    del A_k, phi   # free memory

    # Hermitian symmetry, so that the field in real space is real
    for i in range(N):
        for j in range(N):
            for q in range(N):
                i2 = (-i) % N
                j2 = (-j) % N
                q2 = (-q) % N
                G_tilde[i2, j2, q2] = np.conj(G_tilde[i, j, q])
    G = np.fft.ifftn(G_tilde)
    del G_tilde   # free memory
    G = G.real

    # Mean of G = 0, so that the mean of exp(G), i.e. M, is 1
    sigma_G = np.std(G)
    G = (G - G.mean()) / sigma_G

    sigma_fractal = 1.0
    M = np.exp(G - 0.5 * sigma_fractal**2)
    F = M   # F = 1 + [M - 1] in the paper

    grid_fractal_pc = np.linspace(-L/2, L/2, N)   # the same grid along l, b and the los
    
    return F, grid_fractal_pc


def ring_background(y_min, y_max, coords, n_grid_fractal, rho_0=3.0, R_R=0.05, W_R=0.01, H_R=0.02, **kwargs):
    # Ring with gaussian background (sigma_R = 1 kpc along the los), base of the Fractal model (rho_RB * F in the paper).
    # It is computed on the same n_grid_fractal points of the fractal along the los,
    # so that it can be multiplied point by point with the fractal.
    # (The Ring50 model uses ring() of gas_profile.py)
    l_pc = coords[:, 0][:, None]
    b_pc = coords[:, 1][:, None]

    y_grid_pc = np.linspace(y_min, y_max, n_grid_fractal)
    y_grid_kpc = np.sort(y_grid_pc * 0.001)[None, :]

    x_kpc = l_pc * 0.001
    z_kpc = b_pc * 0.001

    sigma_R = 1.0   # kpc, width of the gaussian background along the los (paper)
    background = np.exp(-(y_grid_kpc)**2/(2*sigma_R**2))

    R = np.sqrt(y_grid_kpc**2 + x_kpc**2)
    rho_RB = rho_0 * np.exp(-((R - R_R)**2) / (2 * W_R**2)) * np.exp(-z_kpc**2 / (2 * H_R**2)) + background
    return rho_RB, y_grid_kpc[0]


def cumulative_column(rho, y_grid_kpc):
    # Sigma(y) of every sub-cell (one row per sub-cell)
    dy = y_grid_kpc[1] - y_grid_kpc[0]
    Sigma_y = np.cumsum(rho * dy, axis=1)
    return Sigma_y


def create_fractal(params_fractal):
    """
    Generates the 3D fractal field (n_grid_fractal^3 points). With the same params_fractal (seed included) it is always the same.
    """
    F_field, grid_fractal_pc = fractal_ISM(**params_fractal)
    return F_field, grid_fractal_pc


def compute_mean_and_sigma_fractal(F_field, grid_fractal_pc, params, params_fractal, l_pc, b_pc, rho_scale,
                                   path_mean_and_sigma, path_rho_cell, path_grid_fractal, path_cell_plot, cell_size_pc=10.0):
    """
    For the cell centred on (l_pc, b_pc) the sub-cells are the points of the fractal grid inside the cell
    (cell of cell_size_pc x cell_size_pc). On every sub-cell the gas density along the los is rho_RB * F.
    Saves:
    - one row (l_pc, b_pc, mean of the column, std of the column, std/mean*100)
    - the mean gas density of the cell along the los (the mock profile of the los)
    - the grid of the fractal (pc)
    - a plot of the cell with the gas column of every sub-cell (M_sun pc^-2, normalized with rho_scale)
    """
    mask_l = (grid_fractal_pc >= l_pc - cell_size_pc/2) & (grid_fractal_pc <= l_pc + cell_size_pc/2)
    mask_b = (grid_fractal_pc >= b_pc - cell_size_pc/2) & (grid_fractal_pc <= b_pc + cell_size_pc/2)
    
    sub_cells_l = grid_fractal_pc[mask_l]   #pc
    sub_cells_b = grid_fractal_pc[mask_b]   #pc

    grid_l, grid_b = np.meshgrid(sub_cells_l, sub_cells_b, indexing="ij")  #pc
    sub_cells = np.column_stack([grid_l.ravel(), grid_b.ravel()])
    
    rho_RB, y_grid_kpc = ring_background(coords=sub_cells, n_grid_fractal=params_fractal["n_grid_fractal"], **params)
    IL, IB = np.meshgrid(np.where(mask_l)[0], np.where(mask_b)[0], indexing="ij")
    F_los = F_field[IL.ravel(), IB.ravel(), :]  # shape: (n_sub_cells, n_grid_fractal)
    rho_sub_cells = F_los * rho_RB              # shape: (n_sub_cells, n_grid_fractal)
    Sigma_y_sub_cells = cumulative_column(rho_sub_cells, y_grid_kpc)

    rho_cell = np.mean(rho_sub_cells, axis=0)   # shape: (n_grid_fractal,)
    
    mean_column = np.mean(Sigma_y_sub_cells[:, -1])
    std_column = np.std(Sigma_y_sub_cells[:, -1], ddof=1)
    results = np.array([(l_pc, b_pc, mean_column, std_column, std_column / mean_column * 100)])

    np.save(path_mean_and_sigma, results)
    np.save(path_rho_cell, rho_cell)
    np.save(path_grid_fractal, grid_fractal_pc)

    print(f"Sigma of the cell ({l_pc:g}, {b_pc:g}) pc: {std_column / mean_column * 100:.2f} %  ({len(sub_cells)} sub-cells)")

    # Plot of the cell: gas column of every sub-cell (density*kpc -> M_sun pc^-2)
    Sigma_sub_cells = Sigma_y_sub_cells[:, -1].reshape(len(sub_cells_l), len(sub_cells_b)) * 1000.0 * rho_scale
    plot_cell(Sigma_sub_cells, sub_cells_l, sub_cells_b, l_pc, b_pc, std_column / mean_column * 100, path_cell_plot)

    return results, rho_cell


def compute_norm_fractal(F_field, grid_fractal_pc, params, params_fractal, M_CMZ, path_rho_scale, R_cyl=80.0, b_max=50.0):
    """
    Factor rho_scale that rescales rho_RB * F so that its mass inside the cylinder
    sqrt(l^2 + y^2) <= R_cyl (plane of longitude l and los y) and |b| <= b_max (pc) is M_CMZ (M_sun).
    The mass is the sum over the points of the fractal grid inside the volume.
    Units of the result: M_sun pc^-3 per unit of density of the ring. Saved in path_rho_scale.
    """
    d = grid_fractal_pc[1] - grid_fractal_pc[0]   # step of the grid, pc
    idx_l = np.where(np.abs(grid_fractal_pc) <= R_cyl)[0]
    idx_b = np.where(np.abs(grid_fractal_pc) <= b_max)[0]

    # rho_RB on all the (l, b) of the volume, along the whole los
    grid_l, grid_b = np.meshgrid(grid_fractal_pc[idx_l], grid_fractal_pc[idx_b], indexing="ij")
    coords = np.column_stack([grid_l.ravel(), grid_b.ravel()])
    rho_RB, y_grid_kpc = ring_background(coords=coords, n_grid_fractal=params_fractal["n_grid_fractal"], **params)
    rho_RB = rho_RB.reshape(len(idx_l), len(idx_b), -1)            # (l, b, y)

    F_box = F_field[idx_l[0]:idx_l[-1]+1, idx_b[0]:idx_b[-1]+1, :]      # (l, b, y)
    inside = np.sqrt(grid_fractal_pc[idx_l][:, None]**2 + grid_fractal_pc[None, :]**2) <= R_cyl   # (l, y)

    M_box = np.sum(rho_RB * F_box * inside[:, None, :]) * d**3
    rho_scale = M_CMZ / M_box
    np.save(path_rho_scale, rho_scale)
    return rho_scale
