__all__ = ["fluxmodel_orders", "model", "initialize_HMC", "pairmodel"]

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value
import celerite2
from celerite2.jax import terms as jax_terms
from jaxspec.utils import *


def fluxmodel_orders(self, wav_out, c0, c1, teff, logg, feh, alpha, vsini, zeta, res, rv, u1, u2):
    flux_rest = self.sg.values(teff, logg, feh, alpha, self.wavgrid)
    flux_base = c0[:,jnp.newaxis] + c1[:,jnp.newaxis] * (wav_out - jnp.mean(self.wav_obs, axis=1)[:,jnp.newaxis]) / self.wav_obs_range[:,jnp.newaxis]
    flux_phys = flux_base * broaden_and_shift_vmap_full(wav_out, self.wavgrid, flux_rest, vsini, zeta, get_beta(res), rv, self.varr, u1, u2)
    return flux_phys


def model(sf, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False):
    self = sf.sm

    teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
    logg = numpyro.sample("logg", dist.Uniform(3., 5.))
    feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))
    alpha = numpyro.sample("alpha", dist.Uniform(0., 0.4))
    vsini = numpyro.sample("vsini", dist.Uniform(0, sf.ccfvbroad))
    if empirical_vmacro:
        zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
    else:
        zeta = numpyro.sample("zeta", dist.Uniform(0., 10.))
    q1 = numpyro.sample("q1", dist.Uniform(0, 1))
    q2 = numpyro.sample("q2", dist.Uniform(0, 1))
    u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
    u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

    ones = jnp.ones(self.Norder)
    if sf.wavresmin[0] == sf.wavresmax[0]:
        wavres = numpyro.deterministic("res", sf.wavresmin[0] * ones)
    elif single_wavres:
        wavres_single = numpyro.sample("res", dist.Uniform(low=sf.wavresmin[0], high=sf.wavresmax[0]))
        wavres = ones * wavres_single
    else:
        wavres = numpyro.sample("res", dist.Uniform(low=jnp.array(sf.wavresmin), high=jnp.array(sf.wavresmax))) # output shape becomes () when the name is "wavres"...????
    c0 = numpyro.sample("norm", dist.Uniform(low=0.8*ones, high=1.2*ones))
    c1 = numpyro.sample("slope", dist.Uniform(low=-0.1*ones, high=0.1*ones))
    rv = numpyro.sample("rv", dist.Uniform(low=sf.rvbounds[0]*ones, high=sf.rvbounds[1]*ones))
    """
    with numpyro.plate("orders", sf.sm.Norder):
        c0 = numpyro.sample("c0", dist.Uniform(0.8, 1.2))
        c1 = numpyro.sample("c1", dist.Uniform(-0.1, 0.1))
        rv = numpyro.sample("rv", dist.Uniform(sf.rvbounds[0], sf.rvbounds[1]))
    """

    fluxmodel = numpyro.deterministic("fluxmodel",
        fluxmodel_orders(self, self.wav_obs, c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2)
        )

    lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
    lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=2))
    kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
    lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
    diags = self.error_obs**2 + jnp.exp(2*lnsigma)

    mask_all = self.mask_obs + (self.mask_fit > 0)
    idx = ~mask_all
    gp = celerite2.jax.GaussianProcess(kernel, mean=0.0)
    gp.compute(self.wav_obs[idx].ravel(), diag=diags[idx].ravel())
    fres = self.flux_obs[idx].ravel() - fluxmodel[idx].ravel()
    numpyro.sample("obs", gp.numpyro_dist(), obs=fres)

    numpyro.deterministic("flux_residual", fres)
    mu, variance = gp.predict(fres, return_var=True) # t=sf.wavgrid doesn't work
    """
    numpyro.deterministic("gpfluxmodel", mu + fluxmodel[idx].ravel())
    numpyro.deterministic("gpfluxvar", variance)
    """
    #numpyro.deterministic("fluxmodeldense", fluxmodel_orders(self, self.wavgrid, c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2))


def initialize_HMC(sf, keys=None, vals=None):
    params_center = 0.5*(sf.bounds[0]+sf.bounds[1])
    params_opt_shift = 0.99*sf.params_opt + 0.01*params_center
    pdict_init = dict(zip(sf.pnames, params_opt_shift))
    pdict_init['rv'] = jnp.array([np.mean(sf.rvbounds)]*len(sf.ccfrvlist))
    del pdict_init['u1'], pdict_init['u2']
    if keys is not None:
        for k, v in zip(keys, vals):
            pdict_init[k] = v
    print ("# initial parameters for HMC:")
    for key in pdict_init.keys():
        print (key, pdict_init[key])
    init_strategy = init_to_value(values=pdict_init)
    return init_strategy


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


def pairmodel(sf1, sf2, empirical_vmacro=False, lnsigma_max=-3, single_wavres=False):
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
