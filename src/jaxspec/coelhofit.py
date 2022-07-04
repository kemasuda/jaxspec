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
from .utils import *
from jax.config import config
config.update('jax_enable_x64', True)

#%%
class SpecGrid:
    def __init__(self, path):
        self.dgrid = np.load(path)
        self.t0, self.dt = self.dgrid['tgrid'][0], np.diff(self.dgrid['tgrid'])[0]
        self.g0, self.dg = self.dgrid['ggrid'][0], np.diff(self.dgrid['ggrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.a0, self.da = self.dgrid['agrid'][0], np.diff(self.dgrid['agrid'])[0]
        self.wavgrid = self.dgrid['wavgrid']
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0, self.dwav = self.wavgrid[0], np.diff(self.wavgrid)[0]
        self.wavmin, self.wavmax = np.min(self.wavgrid), np.max(self.wavgrid)

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, feh, alpha, wav):
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        fidx = (feh - self.f0) / self.df
        aidx = (alpha - self.a0) / self.da
        wavidx = (wav - self.wav0) / self.dwav
        idxs = [tidx, gidx, fidx, aidx, wavidx]
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
        self.beta_ip = get_beta(80000.)
        self.wav_obs = wav_obs
        self.wavrange = wav_obs.max() - wav_obs.min()
        self.flux_obs = flux_obs
        self.error_obs = error_obs

    def reset_resolution(self, resolution):
        self.beta_ip = get_beta(resolution)

    def model(self, params, dense=False):
        c0, c1, teff, logg, feh, alpha, vsini, zeta, rv = params[:9]
        flux_phys = self.sg.values(teff, logg, feh, alpha, self.wav)
        if dense:
            wav = self.wav
        else:
            wav = self.wav_obs
        flux_base = c0 + c1 * (wav - jnp.mean(self.wav_obs)) / self.wavrange
        return flux_base * broaden_and_shift(wav, self.wav, flux_phys, vsini, zeta, self.beta_ip, rv, self.varr)

    def ldmodel(self, params, dense=False, beta_ip=None):
        c0, c1, teff, logg, feh, alpha, vsini, zeta, rv, u1, u2 = params
        flux_phys = self.sg.values(teff, logg, feh, alpha, self.wav)
        if dense:
            wav = self.wav
        else:
            wav = self.wav_obs
        flux_base = c0 + c1 * (wav - jnp.mean(self.wav_obs)) / self.wavrange
        if beta_ip is None:
            beta_ip = self.beta_ip
        return flux_base * broaden_and_shift(wav, self.wav, flux_phys, vsini, zeta, beta_ip, rv, self.varr, u1=u1, u2=u2)

    def gpmodel(self, params):
        flux_model = self.model(params)
        lna, lnc, lnsigma = params[9:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))
        return gp.predict(self.flux_obs)

    @partial(jit, static_argnums=(0,))
    def objective_gp(self, params):
        flux_model = self.model(params)

        lna, lnc, lnsigma = params[9:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))

        return -2 * gp.log_likelihood(self.flux_obs)

    def init_optim(self, vsinibounds, zetabounds, rvbounds):
        vsinimin, vsinimax = vsinibounds
        zetamin, zetamax = zetabounds
        rvmin, rvmax = rvbounds
        pnames = ['c0', 'c1', 'teff', 'logg', 'feh', 'alpha', 'vsini', 'zeta', 'rv', 'lna', 'lnc', 'lnsigma']
        init_params = jnp.array([1, 0, 5000, 4., -0.2, 0.1, 0.5*vsinimax, 0.5*zetamax, rvmin]+[-3., 0., -5.])
        params_lower = jnp.array([0.8, -0.1, 3500., 1, -1., 0., vsinimin, zetamin, 0.5*(rvmin+rvmax)]+[-5, -5, -10.])
        params_upper = jnp.array([1.2, 0.1, 7000, 5, 0.5, 0.4, vsinimax, zetamax, rvmax]+[0, 1, -5.])
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

        plt.figure()
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

    def npmodel(self, fit_zeta=False, lnsigma_max=-3, loggprior=None, fit_resolution=False):
        teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
        logg = numpyro.sample("logg", dist.Uniform(1, 5.))
        feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))
        alpha = numpyro.sample("alpha", dist.Uniform(0, 0.4))
        vsini = numpyro.sample("vsini", dist.Uniform(self.vsinibounds[0], self.vsinibounds[1]))
        if fit_zeta:
            zeta = numpyro.sample("zeta", dist.Uniform(self.zetabounds[0], self.zetabounds[1]))
        else:
            zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
        rv = numpyro.sample("rv", dist.Uniform(self.rvbounds[0], self.rvbounds[1]))

        c0 = numpyro.sample("c0", dist.Uniform(0.8, 1.2))
        c1 = numpyro.sample("c1", dist.Uniform(-0.1, 0.1))
        q1 = numpyro.sample("q1", dist.Uniform(0, 1))
        q2 = numpyro.sample("q2", dist.Uniform(0, 1))
        u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
        u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

        # could choose fit resolution (not sure if it's useful)
        if fit_resolution:
            resolution = numpyro.sample("resolution", dist.Uniform(40000., 100000.))
            beta_ip = get_beta(resolution)
        else:
            beta_ip = None

        flux_model = numpyro.deterministic("flux_model", self.ldmodel(jnp.array([c0, c1, teff, logg, feh, alpha, vsini, zeta, rv, u1, u2]), beta_ip=beta_ip))

        lna = numpyro.sample("lna", dist.Uniform(low=-5, high=0))
        lnc = numpyro.sample("lnc", dist.Uniform(low=-5, high=1))
        lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma))

        numpyro.sample("obs", gp.numpyro_dist(), obs=self.flux_obs)
        numpyro.deterministic("flux_gpmodel", gp.predict(self.flux_obs))

        numpyro.deterministic("flux_model_dense", self.ldmodel(jnp.array([c0, c1, teff, logg, feh, alpha, vsini, zeta, rv, u1, u2]), dense=True))

        if loggprior is not None:
            mu, sigma = loggprior
            numpyro.factor("loggprior", -0.5 * (logg - mu)**2 / sigma**2)

    def sinemodel(self, fit_zeta=False, lnsigma_max=-3, loggprior=None, fit_resolution=False):
        teff = numpyro.sample("teff", dist.Uniform(3500, 7000))
        logg = numpyro.sample("logg", dist.Uniform(1, 5.))
        feh = numpyro.sample("feh", dist.Uniform(-1, 0.5))
        alpha = numpyro.sample("alpha", dist.Uniform(0, 0.4))
        vsini = numpyro.sample("vsini", dist.Uniform(self.vsinibounds[0], self.vsinibounds[1]))
        if fit_zeta:
            zeta = numpyro.sample("zeta", dist.Uniform(self.zetabounds[0], self.zetabounds[1]))
        else:
            zeta = numpyro.deterministic("zeta", 3.98 + (teff - 5770.) / 650.)
        rv = numpyro.sample("rv", dist.Uniform(self.rvbounds[0], self.rvbounds[1]))

        q1 = numpyro.sample("q1", dist.Uniform(0, 1))
        q2 = numpyro.sample("q2", dist.Uniform(0, 1))
        u1 = numpyro.deterministic("u1", 2*jnp.sqrt(q1)*q2)
        u2 = numpyro.deterministic("u2", jnp.sqrt(q1)-u1)

        # could choose fit resolution (not sure if it's useful)
        if fit_resolution:
            resolution = numpyro.sample("resolution", dist.Uniform(40000., 100000.))
            beta_ip = get_beta(resolution)
        else:
            beta_ip = None

        flux_model = numpyro.deterministic("flux_model", self.ldmodel(jnp.array([1., 0., teff, logg, feh, alpha, vsini, zeta, rv, u1, u2]), beta_ip=beta_ip))

        wav = self.wav_obs
        freqorder = 4
        x = (wav - 0.5*(wav.max() + wav.min())) / (wav.max() - wav.min())
        ns = jnp.arange(freqorder+1)
        X = jnp.hstack([jnp.cos(2*jnp.pi*ns[:,None]*x).T, jnp.sin(2*jnp.pi*ns[1:,None]*x).T])
        coeffs = numpyro.sample("coeffs", dist.Normal(loc=jnp.zeros(2*freqorder+1), scale=0.2*jnp.ones(2*freqorder+1)))
        flux_base = X@coeffs + 1.

        flux_gpmodel = numpyro.deterministic("flux_gpmodel", flux_model * flux_base)
        numpyro.deterministic("flux_model_dense", self.ldmodel(jnp.array([1., 0., teff, logg, feh, alpha, vsini, zeta, rv, u1, u2]), dense=True))

        lnsigma = numpyro.sample("lnsigma", dist.Uniform(low=-10, high=lnsigma_max))
        error_scale = jnp.sqrt(self.error_obs**2 + jnp.exp(2*lnsigma))
        numpyro.sample("obs", dist.Normal(loc=flux_gpmodel, scale=error_scale), obs=self.flux_obs)

        if loggprior is not None:
            mu, sigma = loggprior
            numpyro.factor("loggprior", -0.5 * (logg - mu)**2 / sigma**2)

    def run_hmc(self, pinit=None, target_accept_prob=0.9, nw=100, ns=100, continuum_model='gp', **kwargs):#fit_zeta=False,
        if pinit is None:
            pinit = self.init_params
        params_mid = 0.5 * (self.bounds[0] + self.bounds[1])
        pinit = pinit * 0.99 + params_mid * 0.01 # avoid initialization at the edge
        pinit_dict = dict(zip(self.pnames, pinit))
        del pinit_dict['zeta']
        init_strategy = init_to_value(values=pinit_dict)
        if continuum_model=='gp':
            kernel = numpyro.infer.NUTS(self.npmodel, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
        else:
            kernel = numpyro.infer.NUTS(self.sinemodel, target_accept_prob=target_accept_prob, init_strategy=init_strategy)
            self.pnames = ['teff', 'logg', 'feh', 'alpha', 'vsini', 'zeta', 'rv', 'lnsigma']
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=nw, num_samples=ns)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, **kwargs)#fit_zeta=fit_zeta)
        mcmc.print_summary()
        return mcmc

    def mcmcplots(self, mcmc, skip_trace=True, save=None):
        if not skip_trace:
            import arviz
            arviz.plot_trace(mcmc, var_names=self.pnames);

        samples = mcmc.get_samples()
        models = samples['flux_gpmodel']
        models_phys = samples['flux_model_dense']
        model_mean = np.mean(models, axis=0)
        model_5, model_95 = np.percentile(models, [5, 95], axis=0)

        plt.figure()
        plt.xlabel("wavelength ($\AA$)")
        plt.ylabel("normalized flux")
        plt.xlim(self.wav_obs[0], self.wav_obs[-1])
        plt.plot(self.wav_obs, self.flux_obs, 'o', markersize=2, color='gray', mfc='none', mew=0.3, label='data')
        #plt.plot(self.wav_obs, np.mean(models_phys, axis=0), color='C1', lw=0.8, ls='dashed')
        plt.plot(self.wav, np.mean(models_phys, axis=0), color='C0', lw=0.8, ls='dashed', zorder=-1000, label='physical model')
        plt.plot(self.wav_obs, model_mean, color='C1', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')
        plt.fill_between(self.wav_obs, model_5, model_95, color='C1', lw=1, alpha=0.2, zorder=-1000)
        plt.legend(loc='lower right')
        if save is not None:
            plt.savefig(save+"_models.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        import corner
        import pandas as pd
        pnames = self.pnames + ["q1", "q2"]
        hyper = pd.DataFrame(data=dict(zip(pnames, [samples[k] for k in pnames])))
        fig = corner.corner(hyper, labels=pnames, show_titles="%.2f")
        if save is not None:
            plt.savefig(save+"_corner.png", dpi=200, bbox_inches="tight")
            plt.close()
