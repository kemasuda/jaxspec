__all__ = ["model", "model_sb2", "initialize_HMC", "get_mean_models", "model_sb2_numpyrogp"]

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value
import celerite2
from celerite2.jax import terms as jax_terms
from .utils import *


def model(sf, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False, zeta_max=10., slope_max=0.2, 
        teff_prior=None, logg_prior=None, feh_prior=None, physical_logg_max=False):
    """ standard model
    """
    self = sf.sm

    teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
    if physical_logg_max:
        logg_max = -2.34638497e-08*teff**2 + 1.58069918e-04*teff + 4.53251890 # valid for 4500-7000K
    else:
        logg_max = 5.
    logg = numpyro.sample("logg", dist.Uniform(3., logg_max))
    feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))

    if teff_prior is not None:
        mu, sig = teff_prior
        log_prior = -0.5 * (teff - mu)**2 / sig**2
        numpyro.factor("logprior_teff", log_prior)

    if logg_prior is not None:
        mu, sig = logg_prior
        log_prior = -0.5 * (logg - mu)**2 / sig**2
        numpyro.factor("logprior_logg", log_prior)

    if feh_prior is not None:
        mu, sig = feh_prior
        log_prior = -0.5 * (feh - mu)**2 / sig**2
        numpyro.factor("logprior_feh", log_prior)

    alpha = numpyro.sample("alpha", dist.Uniform(0., 0.4))
    vsini = numpyro.sample("vsini", dist.Uniform(0, sf.ccfvbroad))
    if empirical_vmacro:
        zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
    else:
        zeta = numpyro.sample("zeta", dist.Uniform(0., zeta_max))
    q1 = numpyro.sample("q1", dist.Uniform(0, 1))
    q2 = numpyro.sample("q2", dist.Uniform(0, 1))
    u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
    u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

    ones = jnp.ones(self.Norder)
    if sf.wavresmin[0] == sf.wavresmax[0]:
        wavres = numpyro.deterministic("res", jnp.array(sf.wavresmin))#[0] * ones)
    elif single_wavres:
        wavres_single = numpyro.sample("res", dist.Uniform(low=sf.wavresmin[0], high=sf.wavresmax[0]))
        wavres = ones * wavres_single
    else:
        wavres = numpyro.sample("res", dist.Uniform(low=jnp.array(sf.wavresmin), high=jnp.array(sf.wavresmax))) # output shape becomes () when the name is "wavres"...????

    # linear baseline: quadratic is not better
    c0 = numpyro.sample("norm", dist.Uniform(low=0.8*ones, high=1.2*ones))
    c1 = numpyro.sample("slope", dist.Uniform(low=-slope_max*ones, high=slope_max*ones))
    rv = numpyro.sample("rv", dist.Uniform(low=sf.rvbounds[0]*ones, high=sf.rvbounds[1]*ones))

    fluxmodel = numpyro.deterministic("fluxmodel",
        self.fluxmodel_multiorder(c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2)
        )

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=2))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = self.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = self.mask_obs + (self.mask_fit > 0)
    idx = ~mask_all
    # order-by-order gp
    for j in range(len(fluxmodel)):
        idxj = idx[j]
        gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
        gp.compute(self.wav_obs[j][idxj], diag=diags[j][idxj])
        flux_residual = numpyro.deterministic("flux_residual%d"%j, self.flux_obs[j][idxj] - fluxmodel[j][idxj])
        numpyro.sample("obs%d"%j, gp.numpyro_dist(), obs=flux_residual)
    """
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(self.wav_obs[idx].ravel(), diag=diags[idx].ravel())
    flux_residual = numpyro.deterministic("flux_residual", self.flux_obs[idx].ravel() - fluxmodel[idx].ravel())
    numpyro.sample("obs", gp.numpyro_dist(), obs=flux_residual)
    """



def model_sb2(sf, empirical_vmacro=False, lnsigma_max=-3, vsinimax=30., single_wavres=False,
        rv1bounds=None, drvbounds=None,
        teff_prior=None, logg_prior=None, feh_prior=None, physical_logg_max=False, lncmin=-5):
    """ SB2 model
    """
    self = sf.sm

    teff1 = numpyro.sample("teff1", dist.Uniform(3500, 7000))
    teff2 = numpyro.sample("teff2", dist.Uniform(3500, 7000))
    if physical_logg_max:
        logg_max1 = -2.34638497e-08*teff1**2 + 1.58069918e-04*teff1 + 4.53251890 # valid for 4500-7000K
        logg_max2 = -2.34638497e-08*teff2**2 + 1.58069918e-04*teff2 + 4.53251890
    else:
        logg_max1, logg_max2 = 5., 5.
    logg1 = numpyro.sample("logg1", dist.Uniform(3., logg_max1))
    logg2 = numpyro.sample("logg2", dist.Uniform(3., logg_max2))
    feh1 = numpyro.sample("feh1", dist.Uniform(-1, 0.5))
    feh2 = numpyro.sample("feh2", dist.Uniform(-1, 0.5))

    alpha1 = numpyro.sample("alpha1", dist.Uniform(0., 0.4))
    alpha2 = numpyro.sample("alpha2", dist.Uniform(0., 0.4))
    vsini1 = numpyro.sample("vsini1", dist.Uniform(0, vsinimax))
    vsini2 = numpyro.sample("vsini2", dist.Uniform(0, vsinimax))
    if empirical_vmacro:
        zeta1 = numpyro.deterministic("zeta1", 3.98 + (teff1 - 5770.) / 650.)
        zeta2 = numpyro.deterministic("zeta2", 3.98 + (teff2 - 5770.) / 650.)
    else:
        zeta1 = numpyro.sample("zeta1", dist.Uniform(0., 10.))
        zeta2 = numpyro.sample("zeta2", dist.Uniform(0., 10.))
    q1 = numpyro.sample("q1", dist.Uniform(0, 1))
    q2 = numpyro.sample("q2", dist.Uniform(0, 1))
    u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
    u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

    ones = jnp.ones(self.Norder)
    if sf.wavresmin[0] == sf.wavresmax[0]:
        wavres = numpyro.deterministic("res", jnp.array(sf.wavresmin))#[0] * ones)
    elif single_wavres:
        wavres_single = numpyro.sample("res", dist.Uniform(low=sf.wavresmin[0], high=sf.wavresmax[0]))
        wavres = ones * wavres_single
    else:
        wavres = numpyro.sample("res", dist.Uniform(low=jnp.array(sf.wavresmin), high=jnp.array(sf.wavresmax))) # output shape becomes () when the name is "wavres"...????

    # linear baseline: quadratic is not much better
    c0 = numpyro.sample("norm", dist.Uniform(low=0.8*ones, high=1.2*ones))
    c1 = numpyro.sample("slope", dist.Uniform(low=-0.1*ones, high=0.1*ones))
    #rv1 = numpyro.sample("rv1", dist.Uniform(low=sf.rv1bounds[0]*ones, high=sf.rv1bounds[1]*ones))
    #drv = numpyro.sample("drv", dist.Uniform(low=sf.drvbounds[0]*ones, high=sf.drvbounds[1]*ones))
    if rv1bounds is None:
        rv1min, rv1max = sf.rv1bounds
    else:
        rv1min, rv1max = rv1bounds
    rv1 = numpyro.sample("rv1", dist.Uniform(low=rv1min*ones, high=rv1max*ones))
    if drvbounds is None:
        drvmin, drvmax = sf.drvbounds
    else:
        drvmin, drvmax = drvbounds
    drv = numpyro.sample("drv", dist.Uniform(low=drvmin, high=drvmax))
    rv2 = numpyro.deterministic("rv2", rv1 + drv)

    fluxmodel = numpyro.deterministic("fluxmodel",
        self.fluxmodel_multiorder(c0, c1, teff1, teff2, logg1, logg2, feh1, feh2, alpha1, alpha2, vsini1, vsini2, zeta1, zeta2, wavres, rv1, rv2, u1, u2)
        )

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
    lnc = numpyro.sample("lnc", dist.Uniform(low=lncmin, high=2))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = self.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = self.mask_obs + (self.mask_fit > 0)
    idx = ~mask_all
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(self.wav_obs[idx].ravel(), diag=diags[idx].ravel())
    flux_residual = numpyro.deterministic("flux_residual", self.flux_obs[idx].ravel() - fluxmodel[idx].ravel())
    numpyro.sample("obs", gp.numpyro_dist(), obs=flux_residual)

def cov_rbf(x, tau, alpha, diag):
    dx = x[:,None] - x[None, :]
    return alpha**2 * jnp.exp(-0.5 * (dx / tau)**2) + jnp.diag(diag)


def model_sb2_numpyrogp(sf, empirical_vmacro=False, lnsigma_max=-3, vsinimax=30., single_wavres=False,
        teff_prior=None, logg_prior=None, feh_prior=None, physical_logg_max=False):
    """ SB2 model
    """
    self = sf.sm

    teff1 = numpyro.sample("teff1", dist.Uniform(3500, 7000))
    teff2 = numpyro.sample("teff2", dist.Uniform(3500, 7000))
    if physical_logg_max:
        logg_max1 = -2.34638497e-08*teff1**2 + 1.58069918e-04*teff1 + 4.53251890 # valid for 4500-7000K
        logg_max2 = -2.34638497e-08*teff2**2 + 1.58069918e-04*teff2 + 4.53251890
    else:
        logg_max1, logg_max2 = 5., 5.
    logg1 = numpyro.sample("logg1", dist.Uniform(3., logg_max1))
    logg2 = numpyro.sample("logg2", dist.Uniform(3., logg_max2))
    feh1 = numpyro.sample("feh1", dist.Uniform(-1, 0.5))
    feh2 = numpyro.sample("feh2", dist.Uniform(-1, 0.5))

    alpha1 = numpyro.sample("alpha1", dist.Uniform(0., 0.4))
    alpha2 = numpyro.sample("alpha2", dist.Uniform(0., 0.4))
    vsini1 = numpyro.sample("vsini1", dist.Uniform(0, vsinimax))
    vsini2 = numpyro.sample("vsini2", dist.Uniform(0, vsinimax))
    if empirical_vmacro:
        zeta1 = numpyro.deterministic("zeta1", 3.98 + (teff1 - 5770.) / 650.)
        zeta2 = numpyro.deterministic("zeta2", 3.98 + (teff2 - 5770.) / 650.)
    else:
        zeta1 = numpyro.sample("zeta1", dist.Uniform(0., 10.))
        zeta2 = numpyro.sample("zeta2", dist.Uniform(0., 10.))
    q1 = numpyro.sample("q1", dist.Uniform(0, 1))
    q2 = numpyro.sample("q2", dist.Uniform(0, 1))
    u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
    u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

    ones = jnp.ones(self.Norder)
    if sf.wavresmin[0] == sf.wavresmax[0]:
        wavres = numpyro.deterministic("res", jnp.array(sf.wavresmin))#[0] * ones)
    elif single_wavres:
        wavres_single = numpyro.sample("res", dist.Uniform(low=sf.wavresmin[0], high=sf.wavresmax[0]))
        wavres = ones * wavres_single
    else:
        wavres = numpyro.sample("res", dist.Uniform(low=jnp.array(sf.wavresmin), high=jnp.array(sf.wavresmax))) # output shape becomes () when the name is "wavres"...????

    # linear baseline: quadratic is not much better
    c0 = numpyro.sample("norm", dist.Uniform(low=0.8*ones, high=1.2*ones))
    c1 = numpyro.sample("slope", dist.Uniform(low=-0.1*ones, high=0.1*ones))
    rv1 = numpyro.sample("rv1", dist.Uniform(low=sf.rv1bounds[0]*ones, high=sf.rv1bounds[1]*ones))
    drv = numpyro.sample("drv", dist.Uniform(low=sf.drvbounds[0]*ones, high=sf.drvbounds[1]*ones))
    rv2 = numpyro.deterministic("rv2", rv1 + drv)

    fluxmodel = numpyro.deterministic("fluxmodel",
        self.fluxmodel_multiorder(c0, c1, teff1, teff2, logg1, logg2, feh1, feh2, alpha1, alpha2, vsini1, vsini2, zeta1, zeta2, wavres, rv1, rv2, u1, u2)
        )

    lna = numpyro.sample("lna", dist.Uniform(-5, 0))
    lntau = numpyro.sample("lntau", dist.Uniform(-5, 2))
    lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = self.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = self.mask_obs + (self.mask_fit > 0)
    idx = ~mask_all
    cov = cov_rbf(self.wav_obs[idx].ravel(), jnp.exp(lntau), jnp.exp(lna), diags[idx].ravel())
    flux_residual = numpyro.deterministic("flux_residual", self.flux_obs[idx].ravel() - fluxmodel[idx].ravel())
    numpyro.sample("obs", dist.MultivariateNormal(loc=0., covariance_matrix=cov), obs=flux_residual)


def initialize_HMC(sf, keys=None, vals=None, drop_keys=None, init_order_rv=True):
    """ initialize HMC
    """
    params_center = 0.5*(sf.bounds[0]+sf.bounds[1])
    params_opt_shift = 0.99*sf.params_opt + 0.01*params_center
    pdict_init = dict(zip(sf.pnames, params_opt_shift))
    if init_order_rv:
        pdict_init['rv'] = jnp.array([np.mean(sf.rvbounds)]*len(sf.ccfrvlist))
    del pdict_init['u1'], pdict_init['u2']
    if keys is not None:
        for k, v in zip(keys, vals):
            pdict_init[k] = v
    if drop_keys is not None:
        for k in drop_keys:
            pdict_init.pop(k)
    print ("# initial parameters for HMC:")
    for key in pdict_init.keys():
        print (key, pdict_init[key])
    init_strategy = init_to_value(values=pdict_init)
    return init_strategy


"""
def get_mean_models(samples, sf, keytag=''):
    ms = np.mean(samples['fluxmodel'+keytag], axis=0)
    mres = np.mean(samples['flux_residual'+keytag], axis=0)
    lna, lnc, lnsigma = np.mean(samples['lna']), np.mean(samples['lnc']), np.mean(samples['lnsigma'])

    sm = sf.sm
    idx = ~(sm.mask_obs+sm.mask_fit>0)
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    diags = sm.error_obs**2 + jnp.exp(2*lnsigma)
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(sm.wav_obs[idx].ravel(), diag=diags[idx].ravel())
    mgps = np.array([gp.predict(mres.ravel(), t=wobs) for wobs in sm.wav_obs]) + ms

    return ms, mgps
"""


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


def model_pair(sf1, sf2, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False):
    """ model for a pair of stars assuming common Teff, logg, feh, alpha, zeta but different vsini and rv
    rv1 is adjusted in each order, and drv is common
    """
    self1, self2 = sf1.sm, sf2.sm

    teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
    logg = numpyro.sample("logg", dist.Uniform(3., 5.))
    feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))
    alpha = numpyro.sample("alpha", dist.Uniform(0., 0.4))
    vsini1 = numpyro.sample("vsini1", dist.Uniform(0, sf1.ccfvbroad))
    vsini2 = numpyro.sample("vsini2", dist.Uniform(0, sf2.ccfvbroad))
    if empirical_vmacro:
        zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
    else:
        zeta = numpyro.sample("zeta", dist.Uniform(0., 10.))
    q1 = numpyro.sample("q1", dist.Uniform(0, 1))
    q2 = numpyro.sample("q2", dist.Uniform(0, 1))
    u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
    u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

    ones = jnp.ones(self1.Norder)
    if single_wavres:
        wavres_single = numpyro.sample("res", dist.Uniform(low=sf1.wavresmin[0], high=sf1.wavresmax[0]))
        wavres = ones * wavres_single
    else:
        wavres = numpyro.sample("res", dist.Uniform(low=jnp.array(sf1.wavresmin), high=jnp.array(sf1.wavresmax))) # output shape becomes () when the name is "wavres"...????
    c0 = numpyro.sample("norm", dist.Uniform(low=0.8*ones, high=1.2*ones))
    c1 = numpyro.sample("slope", dist.Uniform(low=-0.1*ones, high=0.1*ones))
    rv1 = numpyro.sample("rv", dist.Uniform(low=sf1.rvbounds[0]*ones, high=sf1.rvbounds[1]*ones))
    drv = numpyro.sample("drv", dist.Uniform(low=-10, high=10))
    rv2 = numpyro.deterministic("rv2", rv1 + drv * ones)

    fluxmodel1 = numpyro.deterministic("fluxmodel1",
        fluxmodel_orders(self1, self1.wav_obs, c0, c1, teff, logg, feh, alpha, vsini1, zeta, wavres, rv1, u1, u2)
        )
    fluxmodel2 = numpyro.deterministic("fluxmodel2",
        fluxmodel_orders(self2, self2.wav_obs, c0, c1, teff, logg, feh, alpha, vsini2, zeta, wavres, rv2, u1, u2)
        )

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=2))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))

    diags1 = self1.error_obs**2 + jnp.exp(2*lnsigma)
    diags2 = self2.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all1 = self1.mask_obs + (self1.mask_fit > 0)
    idx1 = ~mask_all1
    gp1 = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp1.compute(self1.wav_obs[idx1].ravel(), diag=diags1[idx1].ravel())
    fres1 = self1.flux_obs[idx1].ravel() - fluxmodel1[idx1].ravel()
    numpyro.sample("obs1", gp1.numpyro_dist(), obs=fres1)

    mask_all2 = self2.mask_obs + (self2.mask_fit > 0)
    idx2 = ~mask_all2
    gp2 = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp2.compute(self2.wav_obs[idx2].ravel(), diag=diags2[idx2].ravel())
    fres2 = self2.flux_obs[idx2].ravel() - fluxmodel2[idx2].ravel()
    numpyro.sample("obs2", gp2.numpyro_dist(), obs=fres2)

    numpyro.deterministic("flux_residual1", fres1)
    numpyro.deterministic("flux_residual2", fres2)
