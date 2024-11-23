__all__ = ["model_single", "model_sb2", "get_mean_models"]

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import tinygp
from numpyro.infer import init_to_value
import celerite2
from celerite2.jax import terms as jax_terms
from .utils import *


def model_single(sf, param_bounds, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False, zeta_max=10., slope_max=0.2, lnc_max=2., logg_min=3., fit_dilution=False, physical_logg_max=False, save_pred=False):
    """model for a single star

        Args:
            sf: SpecFit class
            param_bounds: dict of parameter bounds
            empirical_vmacro: if True, empirical vmacro-Teff relation is assumed
            single_wavres: if True, wavelength resolution is assumed to be common among orders
            fit_dilution: if True, f_dilution / f_star is fitted (assumed to be <1)
            physical_logg_max: if True, max of logg is determined as a function of Teff
            save_pred: if True, GP predictions are also saved

    """
    _sm = sf.sm
    par = {}

    for key in param_bounds.keys():
        if key == 'logg' and physical_logg_max:
            continue
        if key == 'zeta' and empirical_vmacro:
            continue
        if key == 'wavres':
            continue
        par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(
            jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][0])))
        par[key] = numpyro.deterministic(
            key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])

    if physical_logg_max:
        logg_max = -2.34638497e-08 * \
            par["teff"]**2 + 1.58069918e-04*par["teff"] + \
            4.53251890  # valid for 4500-7000K
        par["logg"] = numpyro.sample(
            "logg", dist.Uniform(param_bounds["logg"][0], logg_max))

    if empirical_vmacro:
        par["zeta"] = numpyro.deterministic(
            "zeta", 3.98 + (par["teff"] - 5770.) / 650.)

    par['u1'] = numpyro.deterministic("u1", 2*jnp.sqrt(par["q1"])*par["q2"])
    par['u2'] = numpyro.deterministic("u2", jnp.sqrt(par["q1"])-par["u1"])

    ones = jnp.ones(_sm.Norder)
    # wavres_min = wavres_max
    if param_bounds['wavres'][0][0] == param_bounds['wavres'][1][0]:
        par['wavres'] = numpyro.deterministic(
            "wavres", param_bounds['wavres'][0])
    # wavres_min != wavres_max, order-independent wavres
    elif single_wavres:
        wavres_single = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0][0], high=param_bounds['wavres'][1][0]))
        par['wavres'] = ones * wavres_single
    # wavres_min != wavres_max, order-dependent wavres
    else:
        par['wavres'] = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0], high=param_bounds['wavres'][1]))

    # dilution
    if fit_dilution:
        par['dilution'] = numpyro.sample("dilution", dist.Uniform())
    else:
        par['dilution'] = numpyro.deterministic("dilution", par['teff']*0.)

    fluxmodel = numpyro.deterministic(
        "fluxmodel", _sm.fluxmodel_multiorder(par))

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=-0.5))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=lnc_max))
    if _sm.gpu:
        kernel = jnp.exp(2*lna) * tinygp.kernels.Matern32(jnp.exp(lnc))
    else:
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample(
        "lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = _sm.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = _sm.mask_obs + (_sm.mask_fit > 0)
    idx = ~mask_all
    for j in range(len(fluxmodel)):
        idxj = idx[j]
        if _sm.gpu:
            gp = tinygp.GaussianProcess(
                kernel, _sm.wav_obs[j][idxj], diag=diags[j][idxj], mean=0.0)
        else:
            gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
            gp.compute(_sm.wav_obs[j][idxj], diag=diags[j][idxj])
        flux_residual = numpyro.deterministic(
            "flux_residual%d" % j, _sm.flux_obs[j][idxj] - fluxmodel[j][idxj])
        numpyro.sample("obs%d" % j, gp.numpyro_dist(), obs=flux_residual)
        if save_pred:
            numpyro.deterministic("pred%d" % j, gp.predict(
                flux_residual, t=_sm.wav_obs[j]))


def model_sb2(sf, param_bounds, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False, zeta_max=10., slope_max=0.2, lnc_max=2., logg_min=3., physical_logg_max=False, save_pred=False):
    """model for SB2
    """
    _sm = sf.sm
    par = {}

    for key in param_bounds.keys():
        if key == 'logg' and physical_logg_max:
            continue
        if key == 'zeta' and empirical_vmacro:
            continue
        if key == 'wavres':
            continue
        if key in ['norm', 'slope', 'rv1', 'drv']:
            par[key+"_scaled"] = numpyro.sample(key+"_scaled", dist.Uniform(
                jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][0])))
            par[key] = numpyro.deterministic(
                key, par[key+"_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])
        else:
            par[key+"1_scaled"] = numpyro.sample(key+"1_scaled", dist.Uniform(
                jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][0])))
            par[key+"1"] = numpyro.deterministic(
                key+"1", par[key+"1_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])

            par[key+"2_scaled"] = numpyro.sample(key+"2_scaled", dist.Uniform(
                jnp.zeros_like(param_bounds[key][0]), jnp.ones_like(param_bounds[key][0])))
            par[key+"2"] = numpyro.deterministic(
                key+"2", par[key+"2_scaled"] * (param_bounds[key][1] - param_bounds[key][0]) + param_bounds[key][0])

    par["rv2"] = numpyro.deterministic("rv2", par["rv1"] + par["drv"])

    if physical_logg_max:
        logg1_max = -2.34638497e-08 * \
            par["teff1"]**2 + 1.58069918e-04*par["teff1"] + \
            4.53251890  # valid for 4500-7000K
        logg2_max = -2.34638497e-08 * \
            par["teff2"]**2 + 1.58069918e-04*par["teff2"] + \
            4.53251890  # valid for 4500-7000K
        par["logg1"] = numpyro.sample(
            "logg1", dist.Uniform(param_bounds["logg"][0], logg1_max))
        par["logg2"] = numpyro.sample(
            "logg2", dist.Uniform(param_bounds["logg"][0], logg2_max))

    if empirical_vmacro:
        par["zeta1"] = numpyro.deterministic(
            "zeta1", 3.98 + (par["teff1"] - 5770.) / 650.)
        par["zeta2"] = numpyro.deterministic(
            "zeta2", 3.98 + (par["teff2"] - 5770.) / 650.)

    par['u11'] = numpyro.deterministic(
        "u11", 2*jnp.sqrt(par["q11"])*par["q21"])
    par['u21'] = numpyro.deterministic("u21", jnp.sqrt(par["q11"])-par["u11"])

    par['u12'] = numpyro.deterministic(
        "u12", 2*jnp.sqrt(par["q12"])*par["q22"])
    par['u22'] = numpyro.deterministic("u22", jnp.sqrt(par["q12"])-par["u12"])

    ones = jnp.ones(_sm.Norder)
    # wavres_min = wavres_max
    if param_bounds['wavres'][0][0] == param_bounds['wavres'][1][0]:
        par['wavres'] = numpyro.deterministic(
            "wavres", param_bounds['wavres'][0])
    # wavres_min != wavres_max, order-independent wavres
    elif single_wavres:
        wavres_single = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0][0], high=param_bounds['wavres'][1][0]))
        par['wavres'] = ones * wavres_single
    # wavres_min != wavres_max, order-dependent wavres
    else:
        par['wavres'] = numpyro.sample("wavres", dist.Uniform(
            low=param_bounds['wavres'][0], high=param_bounds['wavres'][1]))

    # flux ratio (now assumed to be common among orders)
    par['f2_ftot'] = numpyro.sample("f2_ftot", dist.Uniform())
    par['f2_f1'] = numpyro.deterministic(
        "f2_f1", par['f2_ftot']/(1. - par['f2_ftot']))

    fluxmodel = numpyro.deterministic(
        "fluxmodel", _sm.fluxmodel_multiorder(par))

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=-0.5))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=lnc_max))
    if _sm.gpu:
        kernel = jnp.exp(2*lna) * tinygp.kernels.Matern32(jnp.exp(lnc))
    else:
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample(
        "lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = _sm.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = _sm.mask_obs + (_sm.mask_fit > 0)
    idx = ~mask_all
    for j in range(len(fluxmodel)):
        idxj = idx[j]
        if _sm.gpu:
            gp = tinygp.GaussianProcess(
                kernel, _sm.wav_obs[j][idxj], diag=diags[j][idxj], mean=0.0)
        else:
            gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
            gp.compute(_sm.wav_obs[j][idxj], diag=diags[j][idxj])
        flux_residual = numpyro.deterministic(
            "flux_residual%d" % j, _sm.flux_obs[j][idxj] - fluxmodel[j][idxj])
        numpyro.sample("obs%d" % j, gp.numpyro_dist(), obs=flux_residual)
        if save_pred:
            numpyro.deterministic("pred%d" % j, gp.predict(
                flux_residual, t=_sm.wav_obs[j]))


def get_mean_models(samples, sf):
    """compute mean GP predictions for posterior samples

        Args:
            samples: posterior samples from numpyro MCMC
            sf: SpecFit class

        Returns:
            mean of physical flux models
            GP prediction

    """
    ms = np.mean(samples['fluxmodel'], axis=0)
    lna, lnc, lnsigma = np.mean(samples['lna']), np.mean(
        samples['lnc']), np.mean(samples['lnsigma'])

    sm = sf.sm
    idx = ~(sm.mask_obs + sm.mask_fit > 0)
    if not sm.gpu:
        kernel = jax_terms.Matern32Term(
            sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    else:
        kernel = jnp.exp(2*lna) * tinygp.kernels.Matern32(jnp.exp(lnc))
    diags = sm.error_obs**2 + jnp.exp(2*lnsigma)

    mgps = []
    for j in range(len(idx)):
        idxj = idx[j]
        res = np.mean(samples['flux_residual%d' % j], axis=0)
        if not sm.gpu:
            gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
            gp.compute(sm.wav_obs[j][idxj], diag=diags[j][idxj])
            mgp = gp.predict(res, t=sm.wav_obs[j])
        else:
            gp = tinygp.GaussianProcess(
                kernel, sm.wav_obs[j][idxj], diag=diags[j][idxj], mean=0.0)
            mgp = gp.predict(res, X_test=sm.wav_obs[j])
        mgps.append(mgp)

    return ms, np.array(mgps) + ms
