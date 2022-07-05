__all__ = ["SpecGridCoelho", "SpecGridSynthe", "SpecModel", "SpecFit"]

#%%
import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial
from jax import (jit, random)
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import glob, re
from scipy.interpolate import interp1d
from jax.config import config
config.update('jax_enable_x64', True)

#%%
from jaxspec.utils import *
import celerite2
from celerite2.jax import terms as jax_terms
import jaxopt

#%%
class SpecGridCoelho:
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

class SpecGridSynthe:
    def __init__(self, path):
        self.dgrid = np.load(path)
        self.t0, self.dt = self.dgrid['tgrid'][0], np.diff(self.dgrid['tgrid'])[0]
        self.g0, self.dg = self.dgrid['ggrid'][0], np.diff(self.dgrid['ggrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.wavgrid = self.dgrid['wavgrid']
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0, self.dwav = self.wavgrid[0], np.diff(self.wavgrid)[0]
        self.wavmin, self.wavmax = np.min(self.wavgrid), np.max(self.wavgrid)

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, feh, alpha, wav): # alpha is not in the grid currently
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        fidx = (feh - self.f0) / self.df
        wavidx = (wav - self.wav0) / self.dwav
        idxs = [tidx, gidx, fidx, wavidx]
        return mapc(self.dgrid['flux'], idxs, order=1, cval=-jnp.inf)

class SpecModel:
    def __init__(self, sg, wav_obs, flux_obs, error_obs, vmax=50.):
        self.sg = sg
        self.Nwav = len(sg.wavgrid)
        self.wavgrid = np.logspace(np.log10(sg.wavmin), np.log10(sg.wavmax), self.Nwav)[1:-1]
        self.dlogwav = np.median(np.diff(np.log(self.wavgrid)))
        self.varr = varr_for_kernels(self.dlogwav, vmax=vmax)
        self.wav_obs = wav_obs
        self.wav_obs_range = wav_obs.max() - wav_obs.min()
        self.flux_obs = flux_obs
        self.error_obs = error_obs
        self.mask = np.zeros_like(flux_obs)

    def sgvalues(self, teff, logg, feh, alpha):
        return self.sg.values(teff, logg, feh, alpha, self.wavgrid)

    def fluxmodel(self, params_phys, observed=True, original=False):
        if observed and (~original):
            wav_out = self.wav_obs
        else:
            wav_out = self.wavgrid

        c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2 = params_phys
        flux_rest = self.sg.values(teff, logg, feh, alpha, self.wavgrid)
        if original:
            return flux_rest

        flux_base = c0 + c1 * (wav_out - jnp.mean(self.wav_obs)) / self.wav_obs_range
        flux_phys = flux_base * broaden_and_shift(wav_out, self.wavgrid, flux_rest, vsini, zeta, get_beta(wavres), rv, self.varr, u1, u2)

        return flux_phys

    #@partial(jit, static_argnums=(0,))
    def gpfluxmodel(self, params):
        flux_model = self.fluxmodel(params[:-3])
        lna, lnc, lnsigma = params[-3:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model)
        gp.compute(self.wav_obs, diag=self.error_obs**2+jnp.exp(2*lnsigma)+self.mask*1e10)
        return gp.predict(self.flux_obs), gp.log_likelihood(self.flux_obs)


def grid_wavranges_paths(gridpath):
    wavranges_grid, paths = [], []
    for gridfile in glob.glob(gridpath+"*.npz"):
        try:
            pattern = ".*/(\d+-\d+).*"
            result = re.match(pattern, gridfile)
            wavranges_grid.append(result.group(1).split("-"))
        except:
            pattern = ".*_(\d+-\d+).*"
            result = re.match(pattern, gridfile)
            wavranges_grid.append(result.group(1).split("-"))
        paths.append(gridfile)
    wavranges_grid = np.array(wavranges_grid).astype(float)
    return wavranges_grid, paths

class SpecFit:
    def __init__(self, data, orders, gridpath, vmax=50.):
        wavranges, paths = grid_wavranges_paths(gridpath)

        smlist, wobslist, fobslist, eobslist = [], [], [], []
        for order in orders:
            idxo = (data.order == order) & (~data.all_mask)
            wobs, fobs, eobs = np.array(data.wave[idxo]), np.array(data.normed_flux[idxo]), np.array(data.flux_error[idxo])
            wobsmin, wobsmax = wobs.min(), wobs.max()

            grididx = int(np.where((wavranges[:,0] < wobsmin)&(wavranges[:,1] > wobsmax))[0])
            gridpath = paths[grididx]
            print (gridpath, "found.")
            assert np.min(wobs - wavranges[grididx,0]) > 5
            assert np.min(wavranges[grididx,1] - wobs) > 5
            if "synthe" in gridpath:
                sg = SpecGridSynthe(gridpath)
            elif "coelho" in gridpath:
                sg = SpecGridCoelho(gridpath)

            smlist.append(SpecModel(sg, wobs, fobs, eobs, vmax=vmax))
            wobslist.append(wobs)
            fobslist.append(fobs)
            eobslist.append(eobs)

        self.smlist = smlist
        self.wobslist = wobslist
        self.fobslist = fobslist
        self.eobslist = eobslist
        self.orders = orders
        self.wavresmin = [70000.]*len(smlist)
        self.wavresmax = [70000.]*len(smlist)
        self.ccfrvlist = None
        self.ccfvbroad = None
        self.rvbounds = None
        self.params_opt = None
        self.pnames = None
        self.bounds = None


    def ziplist(self):
        return zip(self.wobslist, self.fobslist, self.eobslist, self.smlist, self.orders)

    def ziplistres(self):
        return zip(self.wobslist, self.fobslist, self.eobslist, self.smlist, self.orders, self.wavresmin, self.wavresmax)

    def check_ccf(self, output_dir=None):
        vgrids, ccfs, ccffuncs = [], [], []
        for wobs, fobs, eobs, sm, order in self.ziplist():
            print ("# order %d"%order)
            teff, logg, feh, alpha = 5800, 4.4, 0., 0.
            wmodel, fmodel = sm.wavgrid, sm.sgvalues(teff, logg, feh, alpha)
            vgrid, ccf = compute_ccf(wobs, fobs, wmodel, fmodel)
            vgrids.append(vgrid)
            ccfs.append(ccf)
            ccffuncs.append(interp1d(vgrid, ccf))

        ccfrvs = np.array([vg[np.argmax(ccf)] for vg, ccf in zip(vgrids, ccfs)])
        ccfrv = np.median(ccfrvs)
        rvgrid = np.linspace(ccfrv-100, ccfrv+100, 10000)
        medccf = np.median(np.array([cf(rvgrid) for cf in ccffuncs]), axis=0)
        plt.figure(figsize=(8,4))
        plt.xlim(ccfrv-100, ccfrv+100)
        plt.xlabel("radial velocity (km/s)")
        plt.ylabel("normalized CCF")
        plt.axvline(x=ccfrv, label='median: %.1fkm/s'%ccfrv, color='gray', lw=2, alpha=0.4)
        for i, (vg, ccf) in enumerate(zip(vgrids, ccfs)):
            plt.plot(vg, ccf/np.max(ccf), '-', lw=0.5, label='order %d'%self.orders[i])
        plt.plot(rvgrid, medccf/np.max(medccf), color='gray', lw=2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35,1))
        if output_dir is not None:
            plt.savefig(output_dir+"ccfs.png", dpi=200, bbox_inches='tight')
            plt.close()

        dccf = medccf/np.max(medccf) - 0.5
        dccfderiv = dccf[1:] * dccf[:-1]
        v50 = rvgrid[1:][dccfderiv<0]
        vbroad = np.max(v50) - np.min(v50)

        self.ccfrvlist = ccfrvs
        self.ccfvbroad = vbroad

        return ccfrv, vbroad

    def add_wavresinfo(self, resfile):
        dres = pd.read_csv(resfile)
        rmin, rmax = [], []
        for order in self.orders:
            res = dres[dres.order==order].resolution
            rmin.append(res.min())
            rmax.append(res.max())
        self.wavresmin = rmin
        self.wavresmax = rmax

    @partial(jit, static_argnums=(0,))
    def gpobjective(self, p):
        return jnp.sum(jnp.array([-2*sm.gpfluxmodel(p)[1] for sm in self.smlist]))

    #def optim(self, vsinimin=0., zetamin=0., zetamax=10., method='TNC', set_init_params=None):
    def optim(self, solver=None, vsinimin=0., zetamin=0., zetamax=10., method='TNC', set_init_params=None):
        from astropy.stats import sigma_clipped_stats
        vsinimax = self.ccfvbroad
        #rvmean = np.mean(self.ccfrvlist)
        rvmean = sigma_clipped_stats(self.ccfrvlist)[0]
        if vsinimax < 20:
            rvmin, rvmax = rvmean - 5., rvmean + 5.
        else:
            rvmin, rvmax = rvmean - 0.5*vsinimax, rvmean + 0.5*vsinimax
        self.rvbounds = (rvmin, rvmax)
        zetamin, zetamax = 0, 10.
        resmin, resmax = np.max(self.wavresmin), np.min(self.wavresmax)

        pnames = ['c0', 'c1', 'teff', 'logg', 'feh', 'alpha', 'vsini', 'zeta', 'wavres', 'rv', 'u1', 'u2', 'lna', 'lnc', 'lnsigma']
        if set_init_params is None:
            init_params = jnp.array([1, 0, 6000, 4., -0.2, 0.1, 0.5*vsinimax, 0.5*zetamax, 0.5*(resmin+resmax), rvmean, 0, 0]+[-3., 0., -5.])
        else:
            init_params = set_init_params
        params_lower = jnp.array([0.8, -0.1, 3500., 3., -1., 0., vsinimin, zetamin, resmin, rvmin, 0, 0]+[-5, -5, -10.])
        params_upper = jnp.array([1.2, 0.1, 7000, 5., 0.5, 0.4, vsinimax, zetamax, resmax, rvmax, 0, 0]+[0, 1, -5.])
        bounds = (params_lower, params_upper)
        self.pnames = pnames
        self.bounds = bounds

        #objective = lambda p: jnp.sum(jnp.array([-2*sm.gpfluxmodel(p)[1] for sm in self.smlist]))

        print ("# initial objective function:", self.gpobjective(init_params))
        if solver is None:
            solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)
        res = solver.run(init_params, bounds=bounds)

        params, state = res
        print (state)
        for n,v in zip(pnames, params):
            print ("%s\t%f"%(n,v))

        self.params_opt = params
        return params

    def qlplots(self, params, output_dir=None):
        for wobs, fobs, eobs, sm, order in self.ziplist():
            m = sm.fluxmodel(params[:-3], observed=True)
            md = sm.fluxmodel(params[:-3], observed=False)
            mgp = sm.gpfluxmodel(params)[0]
            omitted = sm.mask > 0

            plt.figure(figsize=(10,4))
            plt.title("order %d"%order)
            plt.xlabel("wavelength ($\AA$)")
            plt.ylabel("normalized flux")
            plt.xlim(wobs.min(), wobs.max())
            plt.plot(wobs, fobs, 'o', mfc='none', markersize=4, color='gray')
            plt.plot(wobs[omitted], fobs[omitted], 'o', markersize=4, color='salmon')
            plt.plot(sm.wavgrid, md, lw=1, ls='dashed', label='physical model')
            plt.plot(wobs, mgp, lw=1, label='gp model')
            plt.legend(loc='lower right')
            if output_dir is not None:
                plt.savefig(output_dir+"ql_order%02d.png"%order, dpi=200, bbox_inches="tight")
                plt.close()

    def optim_iterate(self, maxitr=5, cut0=5., cut1=3., method="TNC", output_dir=None):
        params_opt = None
        num_outliers = 0
        solver = jaxopt.ScipyBoundedMinimize(fun=self.gpobjective, method=method)

        for i in range(maxitr):
            print ("# iteration %d..."%i)

            params_opt = self.optim(solver, set_init_params=params_opt)
            #params_opt = self.optim(set_init_params=params_opt)
            #params_opt = optim(set_init_params=params_opt)

            if i==0:
                cut = cut0
            else:
                cut = cut1

            num_outliers_new = 0
            for wobs, fobs, eobs, sm, order in self.ziplist():
                m = sm.fluxmodel(params_opt[:-3], observed=True)
                md = sm.fluxmodel(params_opt[:-3], observed=False)
                mgp = sm.gpfluxmodel(params_opt)[0]
                res = sm.flux_obs - m
                sigma = np.std(res)
                #idxo = np.abs(res) > cut * sigma
                idxo = np.abs(res - np.mean(res)) > cut * sigma

                sm.mask = sm.mask + idxo
                omitted = sm.mask > 0.
                num_outliers_new += np.sum(omitted)

            if num_outliers_new == num_outliers:
                print ("# no new outliers (%d sigma cut)."%cut)
                print ()
                break
            elif i == maxitr - 1:
                print ("# %d outliers (%d sigma cut)."%(num_outliers_new, cut))
                print ("# max iteration reached.")
                print ()
                break
            else:
                print ("# %d outliers (%d sigma cut)."%(num_outliers_new, cut))
                print ()
                num_outliers = num_outliers_new

        if output_dir is not None:
            self.qlplots(params_opt, output_dir=output_dir)
        self.params_opt = params_opt
        return params_opt

    def check_residuals(self, output_dir=None):
        if self.params_opt is None:
            print ("run optim() or optim_iterate() first.")
            return None
        for wobs, fobs, eobs, sm, order in self.ziplist():
            m = sm.fluxmodel(self.params_opt[:-3], observed=True)
            #res = np.array((fobs-m)/eobs)
            res = np.array(fobs-m)
            plt.figure()
            plt.title("order %d"%order)
            plt.yscale("log")
            plt.xlabel("normalized residual")
            plt.ylabel("frequency")
            rnds = np.random.randn(np.sum(sm.mask==0)*1000)*np.std(res) + np.mean(res)
            bins = np.linspace(res.min(), res.max(), 100)
            plt.hist(rnds, bins=bins, histtype='step', lw=1, color='gray', weights=0.001*np.ones_like(rnds))
            plt.hist(res, bins=bins, alpha=0.4);
            plt.hist(res[sm.mask>0], bins=bins, alpha=0.4, label='clipped')
            plt.ylim(0.1, None)
            plt.legend(loc='upper right')
            if output_dir is not None:
                plt.savefig(output_dir+"residual_order%02d.png"%order, dpi=200, bbox_inches="tight")
                plt.close()

    def remove_masked_data(self):
        for sm in self.smlist:
            idx = sm.mask == 0.
            sm.wav_obs = sm.wav_obs[idx]
            sm.flux_obs = sm.flux_obs[idx]
            sm.error_obs = sm.error_obs[idx]
            sm.mask = sm.mask[idx]

    """
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
    """
