import numpy as np
from scipy.interpolate import interp1d

# I don't need @jax.jit for these, original version is good

# ==================================================================================================================================================================
# Functions to be computed ONCE before starting the mcmc
# ==================================================================================================================================================================

def compute_log_P_y(y_edges, P_y, bin_width_pc):
    """
    # Pre-computes log P(y) (stellar density along the los) at every point of the grid
    # as the average of P(y) between the centers of the previous and the following bins
    # The function is called just once
    """
    log_P_y = []
    n_sub=50 # numbers of evaluations for trapezoid    
    for i, y_pc in enumerate(y_edges):
        y_kpc = y_pc * 0.001
        # half-width to the left
        if i == 0:
            half_left_kpc = 0  # or some boundary assumption
        else:
            half_left_kpc = 0.5 * bin_width_pc[i-1] * 0.001
        # half-width to the right
        if i == len(y_edges)-1:
            half_right_kpc = 0  # or some boundary assumption
        else:
            half_right_kpc = 0.5 * bin_width_pc[i] * 0.001
        a, b = y_kpc - half_left_kpc, y_kpc + half_right_kpc
        if b > a:
            ys = np.linspace(a, b, n_sub)
            vals = P_y(ys)
            integral = np.trapezoid(vals, ys)
            avg = integral / (b - a)
            log_P_y.append(np.log(avg) if avg > 0 else -np.inf)
        else:
            log_P_y.append(-np.inf)
    return np.array(log_P_y)


def compute_log_P_vl(y_edges, v_l_stars, P_vl_y):
    """
    # Pre-computes log P(v_l|y) for all the stars in all the possible positions
    # The function is called just once
    """
    y_edges_kpc = y_edges * 1e-3  # convert pc → kpc once
    log_P_vl = np.log(P_vl_y.ev(y_edges_kpc[:, None], v_l_stars[None, :])).T
    return log_P_vl


def build_W(n_grid,M_bins):
    """
    # Pre-computes the matrix W with shape (# of positions on the grid, # of bins)
    # The elements w_i(y) of the matrix are 1 if the bin i is in front of the position y (between the Sun and y)
    # The elements w_i(y) of the matrix are 0 if the bin i is behind the position y
    # The function is called just once
    """
    W = np.zeros((n_grid, M_bins), dtype=float)
    frac=float(M_bins/(n_grid-1))
    n_chunks=int((n_grid-1)/M_bins)
    sub_W = np.zeros((n_chunks,M_bins),dtype=float)
    for i in range (0,M_bins):
        for j in range (0,n_chunks):
            if i == 0:
                sub_W[j,i]=frac*(j+1)
            else:
                sub_W[j,0:i]=1.0
                sub_W[j,i]=frac*(j+1)
        W[(n_chunks*i)+1:(n_chunks*(i+1))+1]=sub_W
    return W # (M+1 x M_bins)


def compute_Sigma_y(rho, y_grid_kpc):
    """
    # Sigma(y): cumulative integral of the gas density along the los
    """
    dy = y_grid_kpc[1] - y_grid_kpc[0]
    Sigma_y = np.cumsum(rho * dy, axis=0)
    
    return Sigma_y
    

def compute_Sigma_obs(Sigma_y, y_grid_kpc, y_stars): # y_stars in kpc
    """
    # Sigma_obs: gas column in front of every star of the sample
    """
    last_value = Sigma_y[-1]
    interp_Sigma_y = interp1d(y_grid_kpc, Sigma_y,kind='linear',bounds_error=False,fill_value=(0.0,last_value)) 
    Sigma_obs_stars = interp_Sigma_y(y_stars)
    return Sigma_obs_stars
