
__all__ = ["get_parameter_bounds", "optim_svi", "get_mean_models", "scale_pdic"]

import numpy as np
import jax.numpy as jnp
from astropy.stats import sigma_clipped_stats
import numpyro
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_value, init_to_sample
import celerite2, tinygp
from celerite2.jax import terms as jax_terms


def get_parameter_bounds(sf, slope_max=0.2, zeta_max=10., model='coelho'):
    ones = jnp.ones(sf.sm.Norder)

    vsini_max = sf.ccfvbroad
    rvmean = sigma_clipped_stats(sf.ccfrvlist)[0]
    if vsini_max < 20:
        rvmin, rvmax = rvmean - 5., rvmean + 5.
    else:
        rvmin, rvmax = rvmean - 0.5 * vsini_max, rvmean + 0.5 * vsini_max

    param_bounds = {
        "teff": [3500., 7000.],
        "logg": [3., 5.],
        "feh": [-1, 0.5],
        "vsini": [0, vsini_max],
        "zeta": [0, zeta_max],
        "q1": [0, 1],
        "q2": [0, 1],
        "norm": [0.8*ones, 1.2*ones],
        "slope": [-slope_max*ones, slope_max*ones],
        "rv": [rvmin*ones, rvmax*ones],
        "wavres": [sf.wavresmin, sf.wavresmax]
    }

    if model == 'bosz':
        param_bounds["alpha"] = [-0.25, 0.25]
        param_bounds["carbon"] = [-0.25, 0.25]
        param_bounds['vmic'] = [0., 4.]
    else:
        param_bounds["alpha"] = [0, 0.4]

    return param_bounds


def optim_svi(numpyro_model, step_size, num_steps, p_initial=None, **kwargs):
    """optimization using Stochastic Variational Inference (SVI)

        Args:
            numpyro_model: numpyro model
            step_size: step size for optimization
            num_steps: # of steps for optimization
            p_initial: initial parameter set (dict); if None, use init_to_sample to initialize

        Returns:
            p_fit: optimized parameter set

    """
    optimizer = numpyro.optim.Adam(step_size=step_size)
    
    if p_initial is None:
        guide = AutoLaplaceApproximation(numpyro_model, init_loc_fn=init_to_sample)
    else:
        guide = AutoLaplaceApproximation(numpyro_model, init_loc_fn=init_to_value(values=p_initial))

    # SVI object
    svi = SVI(numpyro_model, guide, optimizer, loss=Trace_ELBO(), **kwargs)

    # run the optimizer and get the posterior median
    svi_result = svi.run(random.PRNGKey(0), num_steps)
    params_svi = svi_result.params
    p_fit = guide.median(params_svi)

    return p_fit


def get_mean_models(samples, sf):
    ms = np.mean(samples['fluxmodel'], axis=0)
    lna, lnc, lnsigma = np.mean(samples['lna']), np.mean(samples['lnc']), np.mean(samples['lnsigma'])
    
    sm = sf.sm
    idx = ~(sm.mask_obs+sm.mask_fit>0)
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    diags = sm.error_obs**2 + jnp.exp(2*lnsigma)

    mgps = []
    for j in range(len(idx)):
        idxj = idx[j]
        res = np.mean(samples['flux_residual%d'%j], axis=0)
        if not sm.gpu:
            gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
            gp.compute(sm.wav_obs[j][idxj], diag=diags[j][idxj])
            mgp = gp.predict(res, t=sm.wav_obs[j])
        else:
            gp = tinygp.GaussianProcess(kernel, sm.wav_obs[j][idxj], diag=diags[j][idxj], mean=0.0)
            mgp = gp.predict(res, X_test=sm.wav_obs[j])
        mgps.append(mgp)

    return ms, np.array(mgps) + ms


def scale_pdic(pdic, param_bounds):
    """scale parameters using bounds
    
        Args:
            pdic: dict of physical parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict of scaled parameters
    
    """
    pdic_scaled = {}
    for key, (pmin, pmax) in param_bounds.items():
        if np.allclose(pmin, pmax):
            continue
        pdic_scaled[key+"_scaled"] = (pdic[key] - pmin) / (pmax - pmin)
    return pdic_scaled

'''
def unscale_pdic(pdic_scaled, param_bounds):
    """unscale parameters using bounds
    
        Args:
            pdic: dict of scaled parameters
            param_bounds: dictionary of (lower bound array, upper bound array)

        Returns:
            dict of physical parameters in original scales
    
    """
    pdic = {}
    for key in param_bounds.keys():
        pdic[key] = param_bounds[key][0] + (param_bounds[key][1] - param_bounds[key][0]) * pdic_scaled[key+"_scaled"]
    return pdic

def information(sf, pdic, keys):
    """compute Fisher information matrix for iid gaussian likelihood

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters; keys must contain {ecosw, esinw, period, tic, lnode, cosi} and {mass or lnmass}
            keys: parameter keys for computing fisher matrix

        Returns:
            information matrix computed as grad.T Sigma_inv grad

    """
    assert set(keys).issubset(pdic.keys()), "all keys must be included in pdic.keys()."
    mask_all = sf.sm.mask_obs + (sf.sm.mask_fit > 0.)
    func = lambda p: sf.sm.fluxmodel_multiorder(p).ravel()[~mask_all.ravel()]
    jacobian_pytree = jacfwd(func)(pdic)
    jacobian = jnp.hstack([np.atleast_2d(jacobian_pytree[key]).T if jacobian_pytree[key].ndim==1 else jacobian_pytree[key] for key in keys])
    sigma_inv = jnp.diag(1. / sf.sm.error_obs.ravel()[~mask_all.ravel()]**2)
    information_matrix = jacobian.T@sigma_inv@jacobian
    return information_matrix


def scaled_information(sf, pdic, param_bounds, keys):
    """compute Fisher information matrix for iid gaussian likelihood

        Args:
            jttv: JaxTTV object
            pdic: dict containing parameters; keys must contain {ecosw, esinw, period, tic, lnode, cosi} and {mass or lnmass}
            keys: parameter keys for computing fisher matrix

        Returns:
            information matrix computed as grad.T Sigma_inv grad

    """
    assert set(keys).issubset(pdic.keys()), "all keys must be included in pdic.keys()."
    mask_all = sf.sm.mask_obs + (sf.sm.mask_fit > 0.)
    func = lambda p: sf.sm.fluxmodel_multiorder(p).ravel()[~mask_all.ravel()]
    jacobian_pytree = jacfwd(func)(pdic)
    jacobian = jnp.hstack([np.atleast_2d(jacobian_pytree[key]).T*(param_bounds[key][1] - param_bounds[key][0]) if jacobian_pytree[key].ndim==1 else jacobian_pytree[key]*(param_bounds[key][1] - param_bounds[key][0]) for key in keys])
    sigma_inv = jnp.diag(1. / sf.sm.error_obs.ravel()[~mask_all.ravel()]**2)
    information_matrix = jacobian.T@sigma_inv@jacobian
    return information_matrix
'''