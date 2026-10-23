############################################################################################################################################################################
#
#                                                                                              MAIN
#
# Reconstruction of the gas distribution along a line of sight (los) from velocities and gas columns.
# Two mock gas distributions can be used: Ring50 and Fractal (Fractal3, Fractal2, Fractal1 of the paper). Choose everything in the SETTINGS section below.
# The stellar density P(y) and the velocity likelihood P(v_l|y) are provided in Input/P_y.pkl and Input/P_vl_y.pkl;
# all the other inputs are generated the first time they are needed.
# Names as in the paper: y is the coordinate along the los, (l, b) the centre of the cell on the sky, both in pc.
#
############################################################################################################################################################################



import numpy as np
import joblib
import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import arviz as az
from pathlib import Path
import numpyro
import matplotlib.pyplot as plt



############################################################################################################################################################################
#
#                                                                                              SETTINGS
#                                                                           (all the choices of the user are collected here)
#
############################################################################################################################################################################

# === Mock gas distribution ===
# "Ring50" or "Fractal" (the value of alpha in params_fractal gives Fractal3, Fractal2 or Fractal1 of the paper)
gas_model    =  "Ring50"
new_profile  =  False    # True: the gas profile on the los is regenerated even if its files already exist

# === Lines of sight: centre of the cell (l, b) in pc. Cells of 10 pc x 10 pc ===
# l from -185 to +185 pc, b from -65 to +65 pc, in steps of 10 pc (the Galactic centre is at the corner of 4 cells)
l_list  =  [-5]
b_list  =  [-5]

# === Extent of the los ===
y_min  =  -200.0 #pc   (not below -210 pc: limit of the provided velocity likelihood)
y_max  =  200.0  #pc   (not above +210 pc)

# === Parameters of the ring (kpc) ===
params = {
    "y_min": y_min,
    "y_max": y_max,
    "rho_0": 3.0,
    "H_R": 0.02,
    "R_R": 0.05,
    "W_R": 0.01
}

# === Parameters of the fractal ===
# The fractal is regenerated only when the files of the los are missing. With the same seed it is always the same fractal.
params_fractal = {
    "y_min": y_min,
    "y_max": y_max,
    "n_grid_fractal": 400,
    "alpha": 3.0,
    "seed": 0
}

# === Stars and resolution ===
n_stars_pool  =  10000                      # stars generated with AGAMA in the cell of each los (pool to draw from)
list_n_stars  =  [100,200,300,400,500,600]  # stars drawn from the pool and used in the mcmc
list_res      =  [20,8,4]                   # resolution: width of the bins along the los (pc)

# === Error on the gas column ===
a_fov = 0.1

# === Mass of the CMZ (normalization of the gas density in the plots) ===
M_CMZ = 2e7   # M_sun

# === MCMC ===
n_warmup            =  500
n_samples           =  1000
n_chain             =  4
target_accept_prob  =  0.8
max_tree_depth      =  10
seed                =  0



############################################################################################################################################################################
#
#                                                                                              INTRODUCTION AND INITIALIZATION
#
############################################################################################################################################################################

numpyro.set_host_device_count(n_chain)   # one device per chain, so that the chains run in parallel

cell_size_pc = 10.0   # size of the cells on the sky (fixed by the provided P(y) and P(v_l|y))

# === Check of the settings ===
if gas_model not in ["Ring50", "Fractal"]:
    raise ValueError(f'gas_model must be "Ring50" or "Fractal", not "{gas_model}"')
if n_stars_pool < max(list_n_stars):
    raise ValueError(f"n_stars_pool ({n_stars_pool}) must be at least the largest value of list_n_stars ({max(list_n_stars)})")
for res in list_res:
    if not np.isclose((y_max - y_min) / res, round((y_max - y_min) / res)):
        raise ValueError(f"The resolution {res} pc does not divide the los from {y_min} to {y_max} pc into an integer number of bins")
for l_pc in l_list:
    if not (-185 <= l_pc <= 185 and (l_pc - 5) % 10 == 0):
        raise ValueError(f"l = {l_pc} pc is not the centre of a cell: allowed values are -185, -175, ..., +185 pc")
for b_pc in b_list:
    if not (-65 <= b_pc <= 65 and (b_pc - 5) % 10 == 0):
        raise ValueError(f"b = {b_pc} pc is not the centre of a cell: allowed values are -65, -55, ..., +65 pc")

# === General Path ===
path_main     =  Path(__file__).resolve().parent   # main folder of the repository
path_input    =  path_main / "Input"
path_result   =  path_main / "Results"
os.makedirs(path_input, exist_ok=True)

# =================================================================================
# FUNCTIONS (from external files)
# =================================================================================

from precompute_jax_compatible import compute_log_P_y, compute_log_P_vl, build_W, compute_Sigma_y, compute_Sigma_obs
from posterior_jax_compatible import log_prob, log_posterior, log_prior, log_likelihood_allstars
from gas_profile import ring, compute_mean_and_sigma, compute_norm_ring
from agama_model import build_agama_model, generate_stars
from create_fractal import create_fractal, compute_mean_and_sigma_fractal, compute_norm_fractal
from plot_results import plot_results

# =================================================================================
# GLOBAL INPUTS
# =================================================================================

# === Stellar density P(y) and velocity likelihood P(v_l|y) (provided) ===
# Dictionaries with key (l, b) in pc. The NSD model is symmetric for l -> -l and b -> -b,
# so only the quadrant l < 0, b < 0 is stored: the los (l, b) uses the key (-|l|, -|b|)
P_y_all    = joblib.load(path_input / "P_y.pkl")
P_vl_y_all = joblib.load(path_input / "P_vl_y.pkl")

def key_quadrant(l_pc, b_pc):
    return (-abs(int(round(l_pc))), -abs(int(round(b_pc))))

los_missing = [f"({l_pc:g}, {b_pc:g})" for b_pc in b_list for l_pc in l_list
               if key_quadrant(l_pc, b_pc) not in P_y_all or key_quadrant(l_pc, b_pc) not in P_vl_y_all]
if los_missing:
    raise FileNotFoundError(f"P(y) and P(v_l|y) are not available for the lines of sight (l, b) = {', '.join(los_missing)} pc")

gm = None        # AGAMA model: built only if some star file is missing
F_field = None   # fractal: generated only if the fractal files of some los are missing


for b_pc in b_list:
    for l_pc in l_list:

        # === Paths to files ===
        los_name        =  f"l{l_pc:g}_b{b_pc:g}"
        path_input_los  =  path_input      / f"Input_{los_name}"
        path_result_los =  path_result     / f"Results_{los_name}"
        os.makedirs(path_input_los, exist_ok=True)
        os.makedirs(path_result_los, exist_ok=True)

        # =================================================================================
        # LOADING QUANTITIES FROM EXTERNAL FILES
        # =================================================================================

        P_y    = P_y_all[key_quadrant(l_pc, b_pc)]
        P_vl_y = P_vl_y_all[key_quadrant(l_pc, b_pc)]
    
    
    
        # =================================================================================
        # CHOOSE the MOCK GAS PROFILE
        # =================================================================================
    
        params["l_pc"] = l_pc
        params["b_pc"] = b_pc
        
        """ NAMES """
        # rho_mock: mock gas density on the los. Ring50, or mean of rho_RB * F over the cell (Fractal)
        # sigma_fov: dispersion of the gas column inside the cell of the los
       
        if gas_model == "Fractal":

            # FOLDERS
            model_name                         = f"Fractal{params_fractal['alpha']:g}"
            path_input_fractal                 = path_input                   / "Fractal"
            path_input_gas_model               = path_input_los               / model_name
            path_result_gas_model              = path_result_los              / model_name
    
            os.makedirs(path_input_fractal, exist_ok=True)
            os.makedirs(path_input_gas_model, exist_ok=True)
            os.makedirs(path_result_gas_model, exist_ok=True)
    
            # FILES
            path_grid_fractal                  = path_input_gas_model         / f"y_grid_fractal_pc.npy"
            path_mean_and_sigma                = path_input_gas_model         / f"mean_and_sigma_{model_name}.npy"
            path_rho_cell                      = path_input_gas_model         / f"rho_cell_{model_name}.npy"
            path_cell_plot                     = path_input_gas_model         / f"Sigma_cell_{model_name}.pdf"
            path_rho_scale                     = path_input_fractal           / f"rho_scale_{model_name}.npy"
            path_rho_mock                      = path_result_gas_model        / f"rho_mock.npy"
            path_Sigma_y_mock                  = path_result_gas_model        / f"Sigma_y_mock.npy"

            # The fractal is generated (fixed seed: always the same) only if the files of this los or the normalization are missing.
            # Mean and sigma are computed only for this los
            files_los_missing  = not (path_grid_fractal.exists() and path_mean_and_sigma.exists() and path_rho_cell.exists())
            if files_los_missing or not path_rho_scale.exists():
                if F_field is None:
                    F_field, grid_fractal_pc = create_fractal(params_fractal)
                # Normalization: rho_RB * F has mass M_CMZ inside the cylinder R <= 80 pc, |b| <= 50 pc
                if not path_rho_scale.exists():
                    compute_norm_fractal(F_field, grid_fractal_pc, params, params_fractal, M_CMZ, path_rho_scale)
                rho_scale = np.load(path_rho_scale)
                if files_los_missing:
                    compute_mean_and_sigma_fractal(F_field, grid_fractal_pc, params, params_fractal, l_pc, b_pc, rho_scale,
                                                   path_mean_and_sigma, path_rho_cell, path_grid_fractal, path_cell_plot, cell_size_pc)

            rho_scale = np.load(path_rho_scale)

            # Quantities for the error and mock profile of the cell
            mean_and_sigma = np.load(path_mean_and_sigma)
            rho_cell       = np.load(path_rho_cell)

            files_required                     = [path_rho_mock, path_Sigma_y_mock]
            files_missing                      = [p for p in files_required if not os.path.exists(p)]
    
        elif gas_model == "Ring50":

            # FOLDERS
            model_name                    = "Ring50"
            path_input_gas_model          = path_input_los  / model_name
            path_result_gas_model         = path_result_los / model_name

            os.makedirs(path_input_gas_model, exist_ok=True)
            os.makedirs(path_result_gas_model, exist_ok=True)
            
            # FILES
            path_mean_and_sigma           = path_input_gas_model/f"mean_and_sigma_{model_name}.npy"
            path_cell_plot                = path_input_gas_model/f"Sigma_cell_{model_name}.pdf"
            path_rho_mock                 = path_result_gas_model/f"rho_mock.npy"
            path_y_grid_kpc               = path_result_gas_model/f"y_grid_kpc.npy"
            path_Sigma_y_mock             = path_result_gas_model/f"Sigma_y_mock.npy"

            # Normalization: the whole ring has mass M_CMZ
            rho_scale = compute_norm_ring(params, M_CMZ)

            # Mean and sigma are computed only for this los, only if the file is missing
            if not path_mean_and_sigma.exists():
                compute_mean_and_sigma(ring, params, l_pc, b_pc, rho_scale, path_mean_and_sigma, path_cell_plot, cell_size_pc)

            # Quantities for the error
            mean_and_sigma = np.load(path_mean_and_sigma)

            files_required                = [path_rho_mock, path_y_grid_kpc, path_Sigma_y_mock]
            files_missing                 = [p for p in files_required if not os.path.exists(p)]

        if new_profile or files_missing:
            if files_missing:
                print(f"⚠️ Missing files: {files_missing} — generating new profile.")
            else:
                print("Generating new profile as requested.")

            if gas_model == "Fractal":
                y_grid_fractal_pc = np.load(path_grid_fractal)
                rho_mock = rho_cell
                Sigma_y_mock = compute_Sigma_y(rho_mock,y_grid_fractal_pc)
                np.save(path_rho_mock,rho_mock)
                np.save(path_Sigma_y_mock,Sigma_y_mock)

            elif gas_model == "Ring50":
                rho_mock, y_grid_kpc = ring(**params)
                Sigma_y_mock = compute_Sigma_y(rho_mock,y_grid_kpc)
                np.save(path_rho_mock,rho_mock)
                np.save(path_y_grid_kpc,y_grid_kpc)
                np.save(path_Sigma_y_mock,Sigma_y_mock)
                print("RING50: New gas profile generated")

        else:
            print("Loading already existing profile.")
            rho_mock          =  np.load(path_rho_mock)
            Sigma_y_mock      =  np.load(path_Sigma_y_mock)
            if gas_model == "Fractal":
                y_grid_fractal_pc  =  np.load(path_grid_fractal)
            else:
                y_grid_kpc         =  np.load(path_y_grid_kpc)
            print("✅ Loaded existing gas profile from disk.")
        
        Sigma_tot = Sigma_y_mock[-1]
        print("SIGMA TOT",Sigma_tot)

        # sigma_fov: std of the gas column inside the cell (the file has std/mean*100 in the last column)
        sigma_fov = mean_and_sigma[0, 4] * 0.01 * Sigma_tot
        print("SIGMA FOV / SIGMA TOT", mean_and_sigma[0, 4] * 0.01)
        print("a_fov", a_fov)
    
        # =================================================================================
        # Section of mcmc DEPENDENT on n_stars
        # =================================================================================
    
        for res in list_res:
        
            # === Bins along the los (independent of n_stars) ===
            M_bins = int(round((y_max - y_min) / res))   # M in the paper
            y_edges = np.linspace(y_min, y_max, M_bins+1) # bin edges = grid of positions for the stars
            y_centers = 0.5 * (y_edges[1:] + y_edges[:-1])
            n_grid = y_edges.shape[0]
        
            for n_stars in list_n_stars:
                # === Paths to files depending on n_stars ===
                path_star_bin = path_result_gas_model/f"{n_stars}Stars_{res:g}pc"
                os.makedirs(path_star_bin, exist_ok=True)
                path_plots = path_star_bin / f"Plots"
                os.makedirs(path_plots, exist_ok=True)
    
                # === STARS ===
                id_file = os.path.join(path_input_los, f"{n_stars}_id_stars.npy")
                v_file = os.path.join(path_input_los, f"{n_stars}_v_l_stars.npy")
                y_kpc_file = os.path.join(path_input_los, f"{n_stars}_y_stars_kpc.npy")   
                y_pc_file = os.path.join(path_input_los, f"{n_stars}_y_stars_pc.npy")  
                Sigma_obs_file = os.path.join(path_input_gas_model, f"{n_stars}_Sigma_obs_stars.npy")
        
                if os.path.exists(id_file) and os.path.exists(v_file):
                    id_stars = np.load(id_file)
                    v_l_stars = np.load(v_file)
                    y_stars_kpc = np.load(y_kpc_file)
                    print("Stars loaded from input files")
                    if os.path.exists(Sigma_obs_file):
                        Sigma_obs_stars = np.load(Sigma_obs_file)
                        print("Column densities of stars loaded from input file")
                    else:    
                        y_stars_pc = y_stars_kpc * 1000.0
    
                        # Extract gas (the grid of the gas profile is in kpc for the Ring50, in pc for the Fractal)
                        if gas_model == "Fractal":
                            Sigma_obs_stars = compute_Sigma_obs(Sigma_y_mock, y_grid_fractal_pc, y_stars_pc)
                        elif gas_model == "Ring50":
                            Sigma_obs_stars = compute_Sigma_obs(Sigma_y_mock, y_grid_kpc, y_stars_kpc)
    
                        np.save(Sigma_obs_file, Sigma_obs_stars)
                        print("Column densities of stars newly computed")        
                else:
                    stars_pool_file = os.path.join(path_input_los, f"stars_pool_{n_stars_pool}.npy")
                    if os.path.exists(stars_pool_file):
                        stars_pool = np.load(stars_pool_file)
                        print("Loading pre-existing stars_pool file", stars_pool.shape)
                    else:
                        if gm is None:
                            gm = build_agama_model()
                        stars_pool = generate_stars(gm, l_pc, b_pc, y_min, y_max, n_stars_pool, stars_pool_file, cell_size_pc)
                        print("First time running this los: new stars_pool file with", len(stars_pool), "stars")
    
                    # Exclude stars before and after y_min and y_max
                    y_min_kpc = y_min * 0.001
                    y_max_kpc = y_max * 0.001
                    mask_position = (stars_pool[:,0].astype(float) >= y_min_kpc) & (stars_pool[:,0].astype(float) <= y_max_kpc)
                    stars_pool_los = stars_pool[mask_position]
                    id_stars_pool = stars_pool_los[:,3].astype(int)
    
                    id_stars = np.random.choice(id_stars_pool[:], size=n_stars, replace=False)
                    y_stars_kpc = np.array([stars_pool_los[stars_pool_los[:, 3] == i, 0].item() for i in id_stars])
                    y_stars_pc = y_stars_kpc * 1000.0
    
                    # Extract velocities
                    v_l_stars = np.array([stars_pool_los[stars_pool_los[:, 3] == i, 4].item() for i in id_stars])
                    # Extract gas (the grid of the gas profile is in kpc for the Ring50, in pc for the Fractal)
                    if gas_model == "Fractal":
                        Sigma_obs_stars = compute_Sigma_obs(Sigma_y_mock, y_grid_fractal_pc, y_stars_pc)
                    elif gas_model == "Ring50":
                        Sigma_obs_stars = compute_Sigma_obs(Sigma_y_mock, y_grid_kpc, y_stars_kpc)
                    
                    np.save(id_file, id_stars)
                    np.save(v_file, v_l_stars)
                    np.save(Sigma_obs_file, Sigma_obs_stars)
                    np.save(y_kpc_file, y_stars_kpc)
                    np.save(y_pc_file, y_stars_pc)
    
    
                log_P_y = jnp.array(compute_log_P_y(y_edges, P_y, np.diff(y_edges))) 
                log_P_vl = jnp.array(compute_log_P_vl(y_edges, v_l_stars, P_vl_y))
                W = jnp.array(build_W(n_grid,M_bins))
                
                
    
                model_args = {
                "y_edges": y_edges,
                "Sigma_obs_stars": Sigma_obs_stars,
                "v_l_stars": v_l_stars,
                "log_P_y": log_P_y,
                "W": W,
                "Sigma_tot": Sigma_tot,
                "log_P_vl": log_P_vl,
                "M_bins": M_bins,
                "sigma_fov":sigma_fov,
                "a_fov":a_fov,
            
                }
                
                path_backend = path_star_bin / f"mcmc_backend.h5" 
                # === Saving bin parameters ===
                bins_file = os.path.join(path_star_bin, f"parameters.npz")    
                np.savez(bins_file,
                    M_bins=M_bins,
                    res=res,
                    y_edges=y_edges,
                    y_centers=y_centers)
            
                # =================================================================================
                # BINNED MODEL
                # =================================================================================
    
                def model_binned(y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, Sigma_tot, log_P_vl, M_bins, sigma_fov, a_fov):
    
                    import numpyro
                    import numpyro.distributions as dist
        
                    vartheta = numpyro.sample("vartheta", dist.Normal(0.,1.0).expand([M_bins]))
    
                        
                    log_post, Sigma_tot_check = log_prob(vartheta, y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, Sigma_tot, log_P_vl, M_bins, sigma_fov, a_fov)
                    print("SIGMA TOT CHECK", Sigma_tot_check)
        
                    numpyro.factor("log posterior det",log_post)
                    numpyro.deterministic("log posterior safe det",log_post)
    
    
                # =================================================================================
                # Function to call if I want to run the mcmc
                # =================================================================================
                def run_inference(model,model_args,n_warmup=500,n_samples=1000,n_chain=4,
                                  target_accept_prob=0.8,max_tree_depth=10,seed=0):
        
                    from numpyro.infer import init_to_median, MCMC, NUTS
                    import jax
        
                    init_strategy = init_to_median()
    
    
                    backend_file = str(path_backend)
        
                    nuts_kernel = NUTS(model,
                                       target_accept_prob=target_accept_prob,
                                       max_tree_depth=max_tree_depth,
                                       init_strategy=init_strategy)
    
                    mcmc = MCMC(nuts_kernel,
                                num_warmup=n_warmup,
                                num_samples=n_samples,
                                num_chains=n_chain,
                                progress_bar=True)
    
                    rng_key = jax.random.PRNGKey(seed)
                    model_args_tuple = (y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, Sigma_tot, log_P_vl, M_bins, sigma_fov, a_fov)
    
                    init_state = nuts_kernel.init(rng_key,num_warmup=n_warmup, model_args=model_args_tuple, model_kwargs={})
                    init_params = init_state.z
        
                    mcmc.run(rng_key, **model_args)   # post-warmup phase
    
    
    
                    # Save the samples
                    idata = az.from_numpyro(mcmc)
                    idata.to_netcdf(path_backend)
    
                    # -------------------------------
                    # Summary
                    # -------------------------------
                    summary_df = az.summary(idata)
                    print(summary_df)       # print whole summary
                    # convert to array
                    summary_array = summary_df.to_numpy()
                    np.save(path_star_bin/"summary_array.npy", summary_array)
                    print("Shape summary array:", summary_array.shape)
    

                # =================================================================================
                # MAIN (I just call the function)
                # =================================================================================
    
                run_inference(model_binned, model_args,
                              n_warmup=n_warmup,n_samples=n_samples,n_chain=n_chain,
                                target_accept_prob=target_accept_prob,max_tree_depth=max_tree_depth,seed=seed)

                # =================================================================================
                # PLOTS (gas profile and cumulative mass)
                # =================================================================================

                if gas_model == "Fractal":
                    y_grid_pc   = y_grid_fractal_pc      # pc
                    los_unit_pc = 1.0                    # gas columns of the Fractal are in density*pc
                    plot_label  = f"{params_fractal['alpha']:g}_{n_stars}Stars_{res:g}pc"
                elif gas_model == "Ring50":
                    y_grid_pc   = y_grid_kpc * 1000.0    # pc
                    los_unit_pc = 1000.0                 # gas columns of the Ring50 are in density*kpc
                    plot_label  = f"{n_stars}Stars_{res:g}pc"

                plot_results(path_backend, Sigma_tot, rho_mock, y_grid_pc, Sigma_y_mock, los_unit_pc, rho_scale,
                             y_edges, y_centers, res, y_stars_kpc*1000.0, n_stars, l_pc, b_pc,
                             y_min, y_max, path_plots, plot_label)
