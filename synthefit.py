__all__ = ["SpecGrid", "SpecFit"]

#%%
import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial
from jax import (jit, random)
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from utils import *

#%%
#"""
class SpecGrid:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synthe_5200-5280_normed.npz')
        self.dgrid = np.load(path)
        self.t0, self.dt = self.dgrid['tgrid'][0], np.diff(self.dgrid['tgrid'])[0]
        self.g0, self.dg = self.dgrid['ggrid'][0], np.diff(self.dgrid['ggrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.wavgrid = self.dgrid['wavgrid']
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0, self.dwav = self.wavgrid[0], np.diff(self.wavgrid)[0]
        self.wavmin, self.wavmax = np.min(self.wavgrid), np.max(self.wavgrid)
        #self.keys = ['flux']

    #def set_keys(self, keys):
    #    self.keys = keys

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, feh, wav):
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        fidx = (feh - self.f0) / self.df
        wavidx = (wav - self.wav0) / self.dwav
        idxs = [tidx, gidx, fidx, wavidx]
        #return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]
        return mapc(self.dgrid['flux'], idxs, order=1, cval=-jnp.inf)

import celerite2.jax
from celerite2.jax import terms as jax_terms
from numpyro.infer import init_to_value
class SpecFit:
    def __init__(self, sg, wav_obs, flux_obs, error_obs):
        self.sg = sg
        self.Nwav = len(sg.wavgrid)
        self.wav = np.logspace(np.log10(sg.wavmin), np.log10(sg.wavmax), self.Nwav)[1:-1]
        self.dlogwav = np.median(np.diff(np.log(self.wav)))
        self.varr = varr_for_kernels(self.dlogwav)
        self.beta_ip = get_beta()
        self.wav_obs = wav_obs
        self.wavrange = wav_obs.max() - wav_obs.min()
        self.flux_obs = flux_obs
        self.error_obs = error_obs

    def model(self, params):
        c0, c1, teff, logg, feh, vsini, zeta, rv = params[:8]
        flux_phys = self.sg.values(teff, logg, feh, self.wav)
        flux_base = c0 + c1 * (self.wav_obs - jnp.mean(self.wav_obs)) / self.wavrange
        return flux_base * broaden_and_shift(self.wav_obs, self.wav, flux_phys, vsini, zeta, self.beta_ip, rv, self.varr)

    def gpmodel(self, params):
        flux_model = self.model(params)
        lna, lnc, lnsigma = params[8:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))
        return gp.predict(self.flux_obs)

    @partial(jit, static_argnums=(0,))
    def objective_gp(self, params):
        flux_model = self.model(params)

        lna, lnc, lnsigma = params[8:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))

        return -2 * gp.log_likelihood(self.flux_obs)

    def init_optim(self, vsinibounds, zetabounds, rvbounds):
        vsinimin, vsinimax = vsinibounds
        zetamin, zetamax = zetabounds
        rvmin, rvmax = rvbounds
        pnames = ['c0', 'c1', 'teff', 'logg', 'feh', 'vsini', 'zeta', 'rv', 'lna', 'lnc', 'lnsigma']
        init_params = jnp.array([1, 0, 5000, 1.5, -0.4, 0.5*vsinimax, 0.5*zetamax, rvmin]+[-3., 0., -5.])
        params_lower = jnp.array([0.5, -0.2, 3500., 1, -1., vsinimin, zetamin, 0.5*(rvmin+rvmax)]+[-5, -5, -15.])
        params_upper = jnp.array([2., 0.2, 7000, 3, 0.5, vsinimax, zetamax, rvmax]+[0, 5, -5.])
        return pnames, init_params, (params_lower, params_upper)

    def optim(self, vsinibounds, zetabounds, rvbounds, method='TNC', set_init_params=None):
        import jaxopt
        pnames, init_params, bounds = self.init_optim(vsinibounds, zetabounds, rvbounds)
        if set_init_params is not None:
            init_params = set_init_params
        self.pnames = pnames
        self.init_params = init_params
        self.bounds = bounds
        self.vsinibounds = vsinibounds
        self.zetabounds = zetabounds
        self.rvbounds = rvbounds

        print ("# initial objective function:", self.objective_gp(init_params))
        solver = jaxopt.ScipyBoundedMinimize(fun=self.objective_gp, method=method)
        res = solver.run(init_params, bounds=bounds)

        params, state = res
        print (state)
        for n,v in zip(pnames, params):
            print ("%s\t%f"%(n,v))

        return params

    def qlplots(self, params):
        fullmodel = self.gpmodel(params)
        physmodel = self.model(params)
        gpmodel = fullmodel - physmodel

        plt.figure(figsize=(12,5))
        plt.xlabel("wavelength ($\AA$)")
        plt.ylabel("normalized flux")
        plt.xlim(self.wav[0], self.wav[-1])
        plt.plot(self.wav_obs, self.flux_obs, '.', markersize=1, color='gray')
        plt.plot(self.wav_obs, physmodel, '-', color='C1', lw=1)

        plt.plot(self.wav_obs, self.flux_obs-physmodel, '.', markersize=1, color='gray')
        plt.plot(self.wav_obs, gpmodel, '-', color='C1', lw=1)
        plt.show()

        res = np.array(self.flux_obs - physmodel)
        resg = np.random.randn(int(1e6))*np.std(res)+np.mean(res)
        plt.yscale("log")
        plt.xlabel("data minus physical model")
        plt.hist(res, bins=100, density=True)
        plt.hist(resg, bins=100, histtype='step', lw=1, density=True, label='Gaussian')
        plt.legend(loc='best')
        plt.show()

    def npmodel(self, fit_zeta=False):
        teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
        logg = numpyro.sample("logg", dist.Uniform(1, 5.))
        feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))
        vsini = numpyro.sample("vsini", dist.Uniform(self.vsinibounds[0], self.vsinibounds[1]))
        if fit_zeta:
            zeta = numpyro.sample("zeta", dist.Uniform(self.zetabounds[0], self.zetabounds[1]))
        else:
            zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
        rv = numpyro.sample("rv", dist.Uniform(self.rvbounds[0], self.rvbounds[1]))

        c0 = numpyro.sample("c0", dist.Uniform(0.5, 2))
        c1 = numpyro.sample("c1", dist.Uniform(-0.2, 0.2))
        flux_model = numpyro.deterministic("flux_model", self.model(jnp.array([c0, c1, teff, logg, feh, vsini, zeta, rv])))

        lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
        lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=5))
        lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-15, high=-5))
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))

        numpyro.sample("obs", gp.numpyro_dist(), obs=self.flux_obs)
        numpyro.deterministic("flux_gpmodel", gp.predict(self.flux_obs))

    def run_hmc(self, pinit=None, fit_zeta=False, target_accept_prob=0.9, nw=100, ns=100):
        if pinit is None:
            pinit = self.init_params
        params_mid = 0.5 * (self.bounds[0] + self.bounds[1])
        pinit = pinit * 0.99 + params_mid * 0.01 # avoid initialization at the edge
        init_strategy = init_to_value(values=dict(zip(self.pnames, pinit)))
        kernel = numpyro.infer.NUTS(self.npmodel, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, fit_zeta=fit_zeta)
        mcmc.print_summary()
        return mcmc

    def mcmcplots(self, mcmc, skip_trace=True):
        if not skip_trace:
            import arviz
            arviz.plot_trace(mcmc, var_names=self.pnames);

        samples = mcmc.get_samples()
        models = samples['flux_gpmodel']
        model_mean = np.mean(models, axis=0)
        model_5, model_95 = np.percentile(models, [5, 95], axis=0)

        plt.figure(figsize=(12,5))
        plt.xlabel("wavelength ($\AA$)")
        plt.ylabel("normalized flux")
        plt.xlim(self.wav_obs[0], self.wav_obs[-1])
        plt.plot(self.wav_obs, self.flux_obs, '.', markersize=1, color='gray')
        plt.plot(self.wav_obs, model_mean, color='C1', lw=1)
        plt.fill_between(self.wav_obs, model_5, model_95, color='C1', lw=1, alpha=0.2)
        plt.show()

        import corner
        import pandas as pd
        hyper = pd.DataFrame(data=dict(zip(self.pnames, [samples[k] for k in self.pnames])))
        fig = corner.corner(hyper, labels=self.pnames, show_titles="%.2f")
