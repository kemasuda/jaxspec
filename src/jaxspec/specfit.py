__all__ = ["SpecFit"] #"grid_wavranges_paths"


import numpy as np
import jax.numpy as jnp
import glob, re
from jax.config import config
config.update('jax_enable_x64', True)

from .utils import *
from .specgrid import SpecGrid
from .specmodel import SpecModel
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import jaxopt


def get_grid_wavranges_and_paths(gridpath):
    """ get wavelength ranges and paths for grid files in the path

        Args:
            gridpath: path for the directory that contains grid files

        Returns:
            wavranges_grid: min and max wavelengths for grid files
            paths: paths for grid files

    """
    if gridpath[-1] != "/":
        gridpath += "/"
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
    def __init__(self, gridpath, data, orders, vmax=50., wav_margin=4.):
        wav_obs, flux_obs, error_obs, mask_obs = data
        assert np.shape(wav_obs)[0] == len(orders)

        wavranges, paths = get_grid_wavranges_and_paths(gridpath)
        paths_order = []
        for i, wobs in enumerate(wav_obs):
            wobsmin, wobsmax = wobs.min(), wobs.max()
            grididx = np.where((wavranges[:,0] < wobsmin) & (wavranges[:,1] > wobsmax))[0]
            assert len(grididx) == 1, "grid data for order %d not found."%orders[i]
            grididx = int(grididx)
            assert np.min(wobs - wavranges[grididx,0]) > wav_margin, "observed wavelengths outside of margin."
            assert np.min(wavranges[grididx,1] - wobs) > wav_margin, "observed wavelengths of margin."
            paths_order.append(paths[grididx])

        self.sm = SpecModel(SpecGrid(paths_order), wav_obs, flux_obs, error_obs, mask_obs)
        self.orders = orders
        self.wavresmin = [70000.]*len(orders)
        self.wavresmax = [70000.]*len(orders)
        self.ccfrvlist = None
        self.ccfvbroad = None
        self.rvbounds = None
        self.params_opt = None
        self.pnames = None
        self.bounds = None

    def add_wavresinfo(self, resfile):
        dres = pd.read_csv(resfile)
        rmin, rmax = [], []
        for order in self.orders:
            res = dres[dres.order==order].resolution
            rmin.append(res.min())
            rmax.append(res.max())
        self.wavresmin = rmin
        self.wavresmax = rmax

    #def ziplist(self):
    #    return zip(self.sm.wav_obs, self.sm.flux_obs, self.sm.error_obs, self.sm.mask_obs, self.orders)

    def check_ccf(self, teff=5800, logg=4.4, feh=0., alpha=0., output_dir=None, tag=''):
        vgrids, ccfs, ccffuncs = [], [], []
        sm = self.sm
        wmodels, fmodels = sm.wavgrid, sm.sgvalues(teff, logg, feh, alpha)
        for wobs, fobs, eobs, mobs, mfit, order, wmodel, fmodel in zip(sm.wav_obs, sm.flux_obs, sm.error_obs, sm.mask_obs, sm.mask_fit, self.orders, wmodels, fmodels):
            print ("# order %d"%order)
            mask = mobs + (mfit > 0.)
            vgrid, ccf = compute_ccf(wobs[~mask], fobs[~mask], wmodel, fmodel)
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
            plt.savefig(output_dir+"ccfs%s.png"%tag, dpi=200, bbox_inches='tight')
            plt.close()

        dccf = medccf/np.max(medccf) - 0.5
        dccfderiv = dccf[1:] * dccf[:-1]
        v50 = rvgrid[1:][dccfderiv<0]
        vbroad = np.max(v50) - np.min(v50)

        self.ccfrvlist = ccfrvs
        self.ccfvbroad = vbroad

        return ccfrv, vbroad

    def optim(self, solver=None, vsinimin=0., zetamin=0., zetamax=10., lnsigmamax=-5, method='TNC', set_init_params=None):
        vsinimax = self.ccfvbroad
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
            init_params = jnp.array([1, 0, 6000, 4., -0.2, 0.1, 0.5*vsinimax, 0.5*zetamax, resmin, rvmin, 0, 0]+[-3., 0., --8])
        else:
            init_params = set_init_params
        params_lower = jnp.array([0.8, -0.1, 3500., 3., -1., 0., vsinimin, zetamin, 0.5*(resmin+resmax), rvmean, 0, 0]+[-5, -5, -10.])
        params_upper = jnp.array([1.2, 0.1, 7000, 5., 0.5, 0.4, vsinimax, zetamax, resmax, rvmax, 0, 0]+[0, 1, lnsigmamax])
        bounds = (params_lower, params_upper)
        self.pnames = pnames
        self.bounds = bounds

        objective = lambda p: -self.sm.gp_loglikelihood(p)

        print ("# initial objective function:", objective(init_params))
        if solver is None:
            solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)
        res = solver.run(init_params, bounds=bounds)

        params, state = res
        print (state)
        for n,v in zip(pnames, params):
            print ("%s\t%f"%(n,v))

        self.params_opt = params
        return params

    def optim_iterate(self, maxiter=10, cut0=5., cut1=3., method="TNC", plot=True, lnsigmamax=-5, **kwargs):
        params_opt = None
        sm = self.sm
        mask_all = sm.mask_obs + (sm.mask_fit > 0.)
        num_masked = np.sum(mask_all)
        objective = lambda p: -sm.gp_loglikelihood(p)
        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method=method)

        for i in range(maxiter):
            print ("# iteration %d..."%i)

            params_opt = self.optim(solver, set_init_params=params_opt, lnsigmamax=lnsigmamax)

            if i==0:
                cut = cut0
            else:
                cut = cut1

            """
            ms = sm.fluxmodel(params_opt[:-3], observed=True)
            mds = sm.fluxmodel(params_opt[:-3], observed=False)
            _, (gp, res) = sm.gpfluxmodel(params_opt, predict=True)
            """
            ms = sm.fluxmodel(sm.wav_obs, params_opt[:-3])
            mds = sm.fluxmodel(sm.wavgrid, params_opt[:-3])
            gp, res = sm.gp_predict(params_opt)

            mgps = np.array([gp.predict(res, t=wobs) for wobs in sm.wav_obs]) + ms

            res_phys = np.array(sm.flux_obs - ms)
            res_phys[mask_all] = np.nan
            sigmas = np.nanstd(res_phys, axis=1)
            idxo = np.abs(res_phys - np.nanmean(res_phys, axis=1)[:,np.newaxis]) > cut*sigmas[:,np.newaxis]
            sm.mask_fit += idxo
            mask_all = sm.mask_obs + (sm.mask_fit > 0.)
            num_masked_new = np.sum(mask_all)

            if num_masked_new == num_masked:
                print ("# no new outliers (%dsigma cut)."%cut)
                print ()
                break
            elif i == maxiter - 1:
                print ("# %d new outliers (%dsigma cut)."%(num_masked_new-num_masked, cut))
                print ("# max iteration reached.")
                print ()
                break
            else:
                print ("# %d new outliers (%dsigma cut)."%(num_masked_new-num_masked, cut))
                print ()
                num_masked = num_masked_new

        if plot:
            #self.plot_models(ms, mgps, mds, output_dir=output_dir)
            self.plot_models(ms, mgps, mds, **kwargs)
        #self.qlplots(params_opt, output_dir=output_dir)
        self.params_opt = params_opt

        return params_opt

    def qlplots(self, params, output_dir=None):
        sm = self.sm
        """
        ms = sm.fluxmodel(params[:-3], observed=True)
        mds = sm.fluxmodel(params[:-3], observed=False)
        _, (gp, res) = sm.gpfluxmodel(params, predict=True)
        """
        ms = sm.fluxmodel(sm.wav_obs, params_opt[:-3])
        mds = sm.fluxmodel(sm.wavgrid, params_opt[:-3])
        gp, res = sm.gp_predict(params_opt)
        mgps = np.array([gp.predict(res, t=wobs) for wobs in sm.wav_obs]) + ms

        for wobs, fobs, eobs, mobs, mfit, order, wgrid, m, md, mgp in zip(sm.wav_obs, sm.flux_obs, sm.error_obs, sm.mask_obs, sm.mask_fit, self.orders, sm.wavgrid, ms, mds, mgps):
            masked_obs = mobs
            masked_fit = mfit > 0.
            masked_all = masked_obs + masked_fit
            #omitted = mobs + mfit > 0.

            plt.figure(figsize=(10,4))
            plt.title("order %d"%order)
            plt.xlabel("wavelength ($\AA$)")
            plt.ylabel("normalized flux")
            plt.xlim(wobs.min(), wobs.max())
            plt.ylim(0.4, 1.2)
            plt.plot(wobs[~masked_obs], fobs[~masked_obs], 'o', mfc='none', markersize=4, color='gray')
            plt.plot(wobs[masked_obs], fobs[masked_obs], 'o', mfc='none', markersize=4, color='gray', alpha=0.2)
            plt.plot(wobs[masked_fit], fobs[masked_fit], 'o', markersize=4, color='salmon')
            plt.plot(wgrid, md, lw=1, ls='dashed', label='physical model')
            plt.plot(wobs, mgp, lw=1, label='gp model')
            #plt.plot(wobs[~masked_obs], (fobs-m)[~masked_obs], 'o', mfc='none', markersize=4, color='gray')
            #plt.plot(wobs[masked_obs], (fobs-m)[masked_obs], 'o', mfc='none', markersize=4, color='gray', alpha=0.2)
            #plt.plot(wobs[masked_fit], (fobs-m)[masked_fit], 'o', markersize=4, color='salmon')
            plt.legend(loc='lower right')

            if output_dir is not None:
                plt.savefig(output_dir+"ql_order%02d.png"%order, dpi=200, bbox_inches="tight")
                plt.close()

    def plot_models(self, ms, mgps, mds=None, output_dir=None, head=None, res_factor=1.5):
        sm = self.sm
        rmax = np.max(np.abs(sm.flux_obs-ms)[~(sm.mask_obs+sm.mask_fit>0)])*res_factor

        for i in range(sm.Norder):
            wobs, fobs, eobs, mobs, mfit  = sm.wav_obs[i], sm.flux_obs[i], sm.error_obs[i], sm.mask_obs[i], sm.mask_fit[i]
            order = self.orders[i]

            mmean = ms[i]
            mgpmean = mgps[i]

            masked_obs = mobs
            masked_fit = mfit > 0.
            masked_all = masked_obs + masked_fit

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[3,1], hspace=0.1))
            plt.xlim(wobs.min(), wobs.max())
            ax1.set_title("order %d"%order)
            ax2.set_xlabel("wavelength ($\mathrm{\AA}$)")
            ax1.set_ylabel("normalized flux")
            ax2.set_ylabel("residual")
            msize = 2
            ax1.plot(wobs[~masked_obs], fobs[~masked_obs], 'o', markersize=msize, color='gray', mfc='none', mew=0.3, label='data')
            #ylim1 = ax1.get_ylim()
            ax1.plot(wobs[masked_obs], fobs[masked_obs], 'o', markersize=msize, color='gray', mfc='none', mew=0.3, label='', alpha=0.2)
            #oidx = sm.mask > 0
            ax1.plot(wobs[masked_fit], fobs[masked_fit], 'o', markersize=msize, color='salmon', mew=0.3, label='')
            if mds is not None:
                ax1.plot(sm.wavgrid[i], mds[i], color='C0', lw=0.8, ls='dashed', zorder=-1000, label='physical model')
            else:
                ax1.plot(wobs, mmean, color='C0', lw=0.8, ls='dashed', zorder=-1000, label='physical model')
            #ax1.plot(wobs, mgpmean, color='C1', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')
            #ax1.plot(sm.wavgrid[i], mdmean, color='C0', lw=0.8, ls='dashed', zorder=-1000, label='physical model')
            ax1.plot(wobs, mgpmean+mmean*0, color='C1', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')
            #ax1.plot(_wgps, _mgps, color='k', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')
            ax1.legend(loc='lower right')

            #ax1.set_ylim(0.45, 1.15)
            #ax1.set_ylim(np.min(ms.ravel())-0.15, 1.15)
            ax1.set_ylim(np.min(ms.ravel())-0.15, 1+rmax)

            #oidx = sm.mask > 0
            #rmax = np.max(np.abs(fobs-mmean))
            #rmax = np.max(np.abs(fobs-mmean)[~masked_all])*2
            ax2.set_ylim(-rmax, rmax)
            #ax2.set_ylim(-0.12, 0.12)
            ax2.axhline(y=0, color='C0', lw=0.8, ls='dashed', zorder=-1000)
            ax2.plot(wobs[~masked_obs], (fobs-mmean)[~masked_obs], 'o', markersize=msize, color='gray', mfc='none', mew=0.3, label='data')
            ax2.plot(wobs[masked_fit], (fobs-mmean)[masked_fit], 'o', markersize=msize, color='salmon', mew=0.3, label='data')
            ax2.plot(wobs, mgpmean-mmean*1, color='C1', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')
            #ax2.plot(_wgps, _mgps-ms[idx].ravel(), color='k', lw=0.8, zorder=-1000, alpha=0.8, label='GP model')

            ax1.yaxis.set_label_coords(-0.055, .5)
            ax2.yaxis.set_label_coords(-0.055, .5)
            if output_dir is not None:
                name = "order%02d.png"%order
                if head is not None:
                    name = head + name
                plt.savefig(output_dir+name, dpi=200, bbox_inches="tight")
                plt.close()

    def check_residuals(self, output_dir=None, tag=''):
        if self.params_opt is None:
            print ("run optim() or optim_iterate() first.")
            return None
        sm = self.sm
        #ms = sm.fluxmodel(self.params_opt[:-3], observed=True)
        ms = sm.fluxmodel(sm.wav_obs, self.params_opt[:-3])
        residuals = np.array(sm.flux_obs - ms)

        for res, mobs, mfit, order in zip(residuals, sm.mask_obs, sm.mask_fit, self.orders):
            idx = (~mobs) & (mfit==0.)
            idx2 = (~mobs) & (mfit>0.)
            plt.figure()
            plt.title("order %d"%order)
            plt.yscale("log")
            plt.xlabel("normalized residual")
            plt.ylabel("frequency")
            rnds = np.random.randn(np.sum(idx)*1000)*np.std(res[idx]) + np.mean(res[idx])
            bins = np.linspace(res[~mobs].min(), res[~mobs].max(), 100)
            plt.hist(rnds, bins=bins, histtype='step', lw=1, color='gray', weights=0.001*np.ones_like(rnds))
            plt.hist(res[idx], bins=bins, alpha=0.4);
            plt.hist(res[idx2], bins=bins, alpha=0.4, label='clipped')
            plt.ylim(0.1, None)
            plt.legend(loc='upper right')
            if output_dir is not None:
                plt.savefig(output_dir+"residual%s_order%02d.png"%(tag,order), dpi=200, bbox_inches="tight")
                plt.close()