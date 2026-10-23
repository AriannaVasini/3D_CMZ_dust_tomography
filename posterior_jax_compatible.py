# jax compatible likelihood
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.nn import softmax


@jax.jit
def log_prob(vartheta, y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, Sigma_tot, log_P_vl, M_bins, sigma_fov, a_fov):
    """
    Transforms vartheta into Sigma (softmax) and calls posterior function
    """
    Sigma_bins = Sigma_tot * softmax(vartheta)
    # log|det J| of the softmax transformation (paper): det J = Sigma_tot^(M-1) prod_i e^vartheta_i / (sum_i e^vartheta_i)^M
    log_det_J = jnp.sum(vartheta) - M_bins * logsumexp(vartheta) + (M_bins - 1) * jnp.log(Sigma_tot)
    logpost=log_posterior(Sigma_bins, y_edges, Sigma_obs_stars, v_l_stars, log_det_J, log_P_y, W, Sigma_tot, log_P_vl, sigma_fov, a_fov)
    logpost = jnp.where(logpost is None, -jnp.inf, logpost)
    
    # Check for theta values
    theta_limit = 1e3
    mask = jnp.any(jnp.abs(vartheta) > theta_limit)
    logpost = jnp.where(mask, -jnp.inf, logpost)
    
    return logpost, Sigma_tot


@jax.jit
def log_posterior(Sigma_bins, y_edges, Sigma_obs_stars, v_l_stars, log_det_J, log_P_y, W, Sigma_tot, log_P_vl, sigma_fov, a_fov):
    """
    Calls prior and likelihood. Puts them together
    """
    lp = log_prior(Sigma_bins,Sigma_tot)
    ll = log_likelihood_allstars(Sigma_bins, y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, log_P_vl, Sigma_tot, sigma_fov, a_fov) 

    logpost = jnp.where(jnp.isfinite(lp) & jnp.isfinite(ll), ll + lp + log_det_J,-jnp.inf)
    return logpost


@jax.jit
def log_prior(Sigma_bins,Sigma_tot):
    """
    Prior over Sigma, uniform
    """
    mask = jnp.all(Sigma_bins >= 0.0) & jnp.isclose(jnp.sum(Sigma_bins), Sigma_tot, atol=1e-6)
    logprior = jnp.where(mask, 0.0, -jnp.inf)
    
    return logprior


@jax.jit
def log_likelihood_allstars(Sigma_bins, y_edges, Sigma_obs_stars, v_l_stars, log_P_y, W, log_P_vl, Sigma_tot, sigma_fov, a_fov):
    """
    Computes the likelihood for all stars
    """
    # Width of the gaussian (paper): sigma_j = sqrt(a^2 sigma_fov^2 + sigma_obs,j^2)
    sigma_obs = 0.02 * Sigma_tot   # observational error on Sigma_obs, the same for every star (paper: 0.02 Sigma_tot)
    sigma_j = jnp.sqrt((a_fov * sigma_fov)**2 + sigma_obs**2)

    Sigma_y = W @ Sigma_bins                       # Sigma(y) at every point of the grid
    Sigma_obs = Sigma_obs_stars[:, None]           # shape: (N, 1) (N number of stars)
    log_P_Sigma_obs = -0.5 * ((Sigma_obs - Sigma_y[None, :])** 2 / sigma_j**2)  - 0.5 * jnp.log(2 * jnp.pi * sigma_j**2)  # shape: (N, M+1)
    log_total = log_P_Sigma_obs + log_P_y[None, :] + log_P_vl # shape: (N, M+1)  
    # Use log-sum-exp trick for numerical stability
    log_like_one_star= logsumexp(log_total, axis=1)
    log_like_total = jnp.sum(log_like_one_star)
    
    return log_like_total
