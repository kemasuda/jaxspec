from scipy.stats import median_abs_deviation as mad
from scipy.signal import medfilt
import jaxopt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .specmodel import SpecModel, SpecModel2
from .specgrid import SpecGrid, SpecGridBosz
from .utils import *
__all__ = ["SpecFit", "SpecFit2"]


import numpy as np
import jax.numpy as jnp
import glob
import re
from jax import config
config.update('jax_enable_x64', True)


def get_grid_wavranges_and_paths(gridpath):
    """get wavelength ranges and paths for grid files in the path

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
    assert len(wavranges_grid), "Spectrum grid files not found."
    return wavranges_grid, paths


class SpecFit:
    """class for spectrum fitting"""

    def __init__(self, gridpath, data, orders, vmax=50., wav_margin=4., gpu=False, model='coelho', wavres_default=70000.):
        """initialization

            Args:
                gridpath: path for model grid files
                data: list of (wavelength, flux, error, mask)
                orders: list of int specifying orders
                vmax: wdith of the velocity grid for line profile calcuation
                wav_margin: wavelength margin required for the model grid
                gpu: Gaussian Process for GPU (not tested)
                model: grid model, 'bosz' for BOSZ grid, 'coelho' for others
                wavres_default: default wavelength resolution

        """
        wav_obs, flux_obs, error_obs, mask_obs = data
        assert np.shape(wav_obs)[0] == len(orders)

        wavranges, paths = get_grid_wavranges_and_paths(gridpath)
        paths_order = []
        for i, wobs in enumerate(wav_obs):
            wobsmin, wobsmax = wobs.min(), wobs.max()
            grididx = np.where((wavranges[:, 0] < wobsmin) & (
                wavranges[:, 1] > wobsmax))[0]
            assert len(
                grididx) == 1, "grid data for order %d not found." % orders[i]
            grididx = int(grididx)
            assert np.min(
                wobs - wavranges[grididx, 0]) > wav_margin, "observed wavelengths outside of margin."
            assert np.min(
                wavranges[grididx, 1] - wobs) > wav_margin, "observed wavelengths outside of margin."
            paths_order.append(paths[grididx])

        if model == 'bosz':
            _sg = SpecGridBosz(paths_order)
        else:
            _sg = SpecGrid(paths_order)
        self.sm = SpecModel(_sg, wav_obs, flux_obs,
                            error_obs, mask_obs, vmax=vmax, gpu=gpu)
        self.orders = orders
        self.wavresmin = [wavres_default]*len(orders)
        self.wavresmax = [wavres_default]*len(orders)
        self.ccfrvlist = None
        self.ccfvbroad = None
        self.rvbounds = None
        self.params_opt = None
        self.pnames = None
        self.bounds = None
        self.vmax = vmax

    def add_wavresinfo(self, res_min, res_max):
        """set wavelength information

            Args:
                res_min, res_max: array of minimum and maximum wavelength resolutions

        """
        assert len(res_min) == len(self.orders)
        assert len(res_max) == len(self.orders)
        assert np.min(res_max - res_min) >= 0.
        self.wavresmin = np.array(res_min).astype(float)
        self.wavresmax = np.array(res_max).astype(float)

    def check_ccf(self, teff=5800, logg=4.4, feh=0., alpha=0., ccfvmax=100., output_dir=None, tag=''):
        """compute CCF with a theoretical template

            Args:
                teff, logg, feh, alpha: parameters for a template
                ccfvmax: CCF is computed for velocities = (RV-ccfvmax, RV+ccfvmax)
                output_dir: save plot if this is specified
                tag: added at the end of plot names

            Returns:
                RV corresponding to CCF peak (km/s)
                FWHM of CCF (km/s)

        """
        vgrids, ccfs, ccffuncs = [], [], []
        sm = self.sm
        if sm.sg.model == 'bosz':
            wmodels, fmodels = sm.wavgrid, sm.sg.values(
                teff, logg, feh, alpha, 0., 1., sm.wavgrid)
        else:
            wmodels, fmodels = sm.wavgrid, sm.sg.values(
                teff, logg, feh, alpha, sm.wavgrid)
        for wobs, fobs, eobs, mobs, mfit, order, wmodel, fmodel in zip(sm.wav_obs, sm.flux_obs, sm.error_obs, sm.mask_obs, sm.mask_fit, self.orders, wmodels, fmodels):
            print("# order %d" % order)
            mask = mobs + (mfit > 0.)
            vgrid, ccf = compute_ccf(wobs[~mask], fobs[~mask], wmodel, fmodel)
            vgrids.append(vgrid)
            ccfs.append(ccf)
            ccffuncs.append(interp1d(vgrid, ccf))

        ccfrvs = np.array([vg[np.argmax(ccf)]
                          for vg, ccf in zip(vgrids, ccfs)])
        ccfrv = np.median(ccfrvs)
        rvgrid = np.linspace(ccfrv-ccfvmax, ccfrv+ccfvmax, 10000)
        medccf = np.median(np.array([cf(rvgrid) for cf in ccffuncs]), axis=0)

        plt.figure(figsize=(8, 4))
        plt.xlim(ccfrv-ccfvmax, ccfrv+ccfvmax)
        plt.xlabel("radial velocity (km/s)")
        plt.ylabel("normalized CCF")
        plt.axvline(x=ccfrv, label='median: %.1fkm/s' %
                    ccfrv, color='gray', lw=2, alpha=0.4)
        for i, (vg, ccf) in enumerate(zip(vgrids, ccfs)):
            plt.plot(vg, ccf/np.max(ccf), '-', lw=0.5,
                     label='order %d' % self.orders[i])
        plt.plot(rvgrid, medccf/np.max(medccf), color='gray', lw=2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        if output_dir is not None:
            plt.savefig(output_dir+"ccfs%s.png" %
                        tag, dpi=200, bbox_inches='tight')
            plt.close()

        dccf = medccf/np.max(medccf) - 0.5
        dccfderiv = dccf[1:] * dccf[:-1]
        v50 = rvgrid[1:][dccfderiv < 0]
        vbroad = np.max(v50) - np.min(v50)

        self.ccfrvlist = ccfrvs
        self.ccfvbroad = vbroad

        assert vbroad < self.vmax, f"vmax {self.vmax} should be sufficiently larger than line width {vbroad}; instantiate the SpecFit class again."

        return ccfrv, vbroad

    def plot_models(self, ms, mgps=None, mds=None, output_dir=None, head=None, res_factor=1.5):
        sm = self.sm
        rmax = np.max(np.abs(sm.flux_obs-ms)
                      [~(sm.mask_obs+sm.mask_fit > 0)])*res_factor

        for i in range(sm.Norder):
            wobs, fobs, eobs, mobs, mfit = sm.wav_obs[i], sm.flux_obs[
                i], sm.error_obs[i], sm.mask_obs[i], sm.mask_fit[i]
            order = self.orders[i]

            mmean = ms[i]
            if mgps is not None:
                mgpmean = mgps[i]

            masked_obs = mobs
            masked_fit = mfit > 0.
            masked_all = masked_obs + masked_fit

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                           gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1))
            plt.xlim(wobs.min(), wobs.max())
            ax1.set_title("order %d" % order)
            ax2.set_xlabel("wavelength ($\mathrm{\AA}$)")
            ax1.set_ylabel("normalized flux")
            ax2.set_ylabel("residual")
            msize = 2
            ax1.plot(wobs[~masked_obs], fobs[~masked_obs], 'o',
                     markersize=msize, color='gray', mfc='none', mew=0.3, label='data')
            ax1.plot(wobs[masked_obs], fobs[masked_obs], 'o', markersize=msize,
                     color='gray', mfc='none', mew=0.3, label='', alpha=0.2)
            ax1.plot(wobs[masked_fit], fobs[masked_fit], 'o',
                     markersize=msize, color='salmon', mew=0.3, label='')
            if mds is not None:
                ax1.plot(sm.wavgrid[i], mds[i], color='C0', lw=0.8,
                         ls='dashed', zorder=-1000, label='physical model')
            else:
                ax1.plot(wobs, mmean, color='C0', lw=0.8, ls='dashed',
                         zorder=-1000, label='physical model')
            if mgps is not None:
                ax1.plot(wobs, mgpmean+mmean*0, color='C1', lw=0.8,
                         zorder=-1000, alpha=0.8, label='GP model')
            ax1.legend(loc='lower right')

            ax1.set_ylim(np.min(ms.ravel())-0.15, 1+rmax)
            ax2.set_ylim(-rmax, rmax)
            ax2.axhline(y=0, color='C0', lw=0.8, ls='dashed', zorder=-1000)
            ax2.plot(wobs[~masked_obs], (fobs-mmean)[~masked_obs], 'o',
                     markersize=msize, color='gray', mfc='none', mew=0.3, label='data')
            ax2.plot(wobs[masked_fit], (fobs-mmean)[masked_fit], 'o',
                     markersize=msize, color='salmon', mew=0.3, label='data')
            if mgps is not None:
                ax2.plot(wobs, mgpmean-mmean*1, color='C1', lw=0.8,
                         zorder=-1000, alpha=0.8, label='GP model')

            ax1.yaxis.set_label_coords(-0.055, .5)
            ax2.yaxis.set_label_coords(-0.055, .5)
            if output_dir is not None:
                name = "order%02d.png" % order
                if head is not None:
                    name = head + name
                plt.savefig(output_dir+name, dpi=200, bbox_inches="tight")
                plt.close()

    def check_residuals(self, par_dict, output_dir=None, tag=''):
        sm = self.sm
        ms = sm.fluxmodel_multiorder(par_dict)
        residuals = np.array(sm.flux_obs - ms) / self.sm.error_obs

        for res, mobs, mfit, order in zip(residuals, sm.mask_obs, sm.mask_fit, self.orders):
            idx = (~mobs) & (mfit == 0.)
            idx2 = (~mobs) & (mfit > 0.)
            plt.figure()
            plt.title("order %d" % order)
            plt.yscale("log")
            plt.xlabel("normalized residual")
            plt.ylabel("frequency")

            bins = np.linspace(res[~mobs].min(), res[~mobs].max(), 100)
            mu, sd = np.mean(res[idx]), np.std(res[idx])
            plt.hist(res[idx], bins=bins, alpha=0.4)
            plt.hist(res[idx2], bins=bins, alpha=0.4, label='clipped')
            plt.plot(bins, np.exp(-0.5*(bins-mu)**2/sd**2)/np.sqrt(2 *
                     np.pi)/sd*np.sum(idx)*np.diff(bins)[0], lw=1, color='gray')
            plt.ylim(0.1, None)
            plt.legend(loc='upper right')
            if output_dir is not None:
                plt.savefig(output_dir+"residual%s_order%02d.png" %
                            (tag, order), dpi=200, bbox_inches="tight")
                plt.close()

    def mask_outliers(self, p_fit, sigma_threshold=3., output_dir=None, extend_outlier_mask=True):
        for i in range(self.sm.Norder):
            x, y, err = self.sm.wav_obs[i], self.sm.flux_obs[i], self.sm.error_obs[i]
            clip = self.sm.mask_obs[i]
            yres_phys = y - p_fit['fluxmodel'][i]
            # for median filtering
            if 'vsini' in p_fit.keys():
                vsini = p_fit['vsini']
            else:
                vsini = max(p_fit['vsini1'], p_fit['vsini2'])
            npix_vsini = int(
                np.median(x) * vsini * 2 / 3e5 / np.median(np.diff(x))) * 4 + 1
            yres_phys_smoothed = medfilt(yres_phys, kernel_size=npix_vsini)
            yres_res = (yres_phys - yres_phys_smoothed) / err
            sigma_cut = 1.4826 * mad(yres_res[~clip])
            # mask_obs and mask_fit are exclusive
            flag_outlier = (np.abs(yres_res) >
                            sigma_threshold * sigma_cut) & (~clip)

            if extend_outlier_mask:
                flag_outlier = extend_mask(flag_outlier) > 0
            self.sm.mask_fit[i] = np.array(flag_outlier).astype(float)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax1.plot(x[~clip], y[~clip], 'o', color='gray',
                     mfc='none', mew=0.5, markersize=3, alpha=0.6)
            ax1.plot(x, p_fit['fluxmodel'][i], color='C0', zorder=-1000, lw=1)
            ax1.plot(x[~clip & (flag_outlier)], y[~clip & (
                flag_outlier)], 'o', color='salmon', lw=1, markersize=3)
            ax1.set_ylabel("normalized flux")

            ax2.plot(x[~clip], yres_phys[~clip], 'o',
                     color='gray', mfc='none', mew=0.5, markersize=3, alpha=0.6)
            ax2.plot(x[~clip], yres_phys_smoothed[~clip], '-', color='k', lw=1)
            ax2.plot(x[~clip & (flag_outlier)], yres_phys[~clip & (
                flag_outlier)], 'o', color='salmon', lw=1, markersize=3)
            ax2.set_ylabel("residual")
            # ax2.set_ylim(ax2.get_ylim())
            # ax2.plot(x[clip], yres_phys[clip], 'x', lw=1, markersize=3)

            ax3.plot(x[~clip], yres_res[~clip], 'o',
                     color='gray', mfc='none', mew=0.5, markersize=3, alpha=0.6)
            ax3.plot(x[~clip & (flag_outlier)], yres_res[~clip & (
                flag_outlier)], 'o', color='salmon', lw=1, markersize=3)
            ax3.set_ylabel("residual / error")
            ax3.set_xlabel("wavelength (A)")

            ax3.set_xlim(x[0], x[-1])
            fig.tight_layout(pad=0.2)

            if output_dir is not None:
                plt.savefig(output_dir+"outlier_order%02d.png" %
                            self.orders[i], dpi=200, bbox_inches="tight")
                plt.close()

        return None


def extend_mask(flag):
    """extend boolean mask when True repeats

        Args:
            flag: boolean array

        Returns:
            extended float array

    """
    n = len(flag)
    extended = flag.copy()

    start = 0
    while start < n:
        if flag[start]:  # Found a sequence of True
            # Count the length of consecutive True's
            end = start
            while end < n and flag[end]:
                end += 1
            length = (end - start) // 2

            # Extend by the length of the sequence on each side
            left = max(start - length, 0)
            right = min(end + length, n)
            extended[left:right] = True

            # Move start to the end of the current True sequence
            start = end
        else:
            start += 1

    return extended.astype(float)


class SpecFit2(SpecFit):
    """ SB2 """

    def __init__(self, gridpath, data, orders, vmax=50., wav_margin=4.):
        wav_obs, flux_obs, error_obs, mask_obs = data
        assert np.shape(wav_obs)[0] == len(orders)

        wavranges, paths = get_grid_wavranges_and_paths(gridpath)
        paths_order = []
        for i, wobs in enumerate(wav_obs):
            wobsmin, wobsmax = wobs.min(), wobs.max()
            grididx = np.where((wavranges[:, 0] < wobsmin) & (
                wavranges[:, 1] > wobsmax))[0]
            assert len(
                grididx) == 1, "grid data for order %d not found." % orders[i]
            grididx = int(grididx)
            assert np.min(
                wobs - wavranges[grididx, 0]) > wav_margin, "observed wavelengths outside of margin."
            assert np.min(
                wavranges[grididx, 1] - wobs) > wav_margin, "observed wavelengths of margin."
            paths_order.append(paths[grididx])

        self.sm = SpecModel2(SpecGrid(paths_order), wav_obs,
                             flux_obs, error_obs, mask_obs)
        self.orders = orders
        self.wavresmin = [70000.]*len(orders)
        self.wavresmax = [70000.]*len(orders)
        self.ccfrvlist = None
        self.ccfvbroad = None
        self.rvbounds = None
        self.params_opt = None
        self.pnames = None
        self.bounds = None
        self.v1 = None
        self.v2 = None

    def check_ccf(self, teff=5800, logg=4.4, feh=0., alpha=0., output_dir=None, ccfvmax=100., tag=''):
        vgrids, ccfs, ccffuncs = [], [], []
        sm = self.sm
        if sm.sg.model == 'bosz':
            wmodels, fmodels = sm.wavgrid, sm.sg.values(
                teff, logg, feh, alpha, 0., 1., sm.wavgrid)
        else:
            wmodels, fmodels = sm.wavgrid, sm.sg.values(
                teff, logg, feh, alpha, sm.wavgrid)
        for wobs, fobs, eobs, mobs, mfit, order, wmodel, fmodel in zip(sm.wav_obs, sm.flux_obs, sm.error_obs, sm.mask_obs, sm.mask_fit, self.orders, wmodels, fmodels):
            print("# order %d" % order)
            mask = mobs + (mfit > 0.)
            vgrid, ccf = compute_ccf(wobs[~mask], fobs[~mask], wmodel, fmodel)
            vgrids.append(vgrid)
            ccfs.append(ccf)
            ccffuncs.append(interp1d(vgrid, ccf))

        ccfrvs = np.array([vg[np.argmax(ccf)]
                          for vg, ccf in zip(vgrids, ccfs)])
        ccfrv = np.median(ccfrvs)
        rvgrid = np.linspace(ccfrv-ccfvmax, ccfrv+ccfvmax, 10000)
        medccf = np.median(np.array([cf(rvgrid) for cf in ccffuncs]), axis=0)

        # CCF velocities for two stars (we choose v1 < v2)
        v_center = np.average(rvgrid, weights=np.abs(medccf))
        ccf1 = np.where(rvgrid < v_center, medccf, 0)
        ccf2 = np.where(rvgrid < v_center, 0, medccf)
        v1, v2 = rvgrid[np.argmax(ccf1)], rvgrid[np.argmax(ccf2)]

        x = rvgrid
        y = np.array(medccf / np.max(medccf))

        def objective(p):
            mu1, dmu, sig, a, b = p
            mu2 = mu1 + dmu
            model = a * jnp.exp(-0.5*(x-mu1)**2/sig**2) + \
                b * jnp.exp(-0.5*(x-mu2)**2/sig**2)
            return jnp.sum((y-model)**2)

        solver = jaxopt.ScipyBoundedMinimize(fun=objective, method="TNC")
        res = solver.run([v1, v2-v1, 5., 1., 1.], bounds=([v1-10,
                         0, 5., 0.5, 0.5], [v1+10, 300, 5., 1., 1.]))
        v1opt, dvopt, sigopt, aopt, bopt = res.params

        self.v1 = v1opt
        self.v2 = v1opt + dvopt
        ccf1, ccf2 = aopt * \
            jnp.exp(-0.5*(x-self.v1)**2/sigopt**2), bopt * \
            jnp.exp(-0.5*(x-self.v2)**2/sigopt**2)

        plt.figure(figsize=(8, 4))
        plt.xlim(ccfrv-ccfvmax, ccfrv+ccfvmax)
        plt.xlabel("radial velocity (km/s)")
        plt.ylabel("normalized CCF")
        # plt.axvline(x=ccfrv, label='median: %.1fkm/s'%ccfrv, color='gray', lw=2, alpha=0.4)
        plt.axvline(x=self.v1, label='star1: %.1fkm/s' %
                    self.v1, color='C0', lw=1.5, alpha=0.4, ls='dashed')
        plt.plot(x, ccf1, color='C0', lw=1., alpha=0.6, ls='solid')
        plt.axvline(x=self.v2, label='star2: %.1fkm/s' %
                    self.v2, color='C1', lw=1.5, alpha=0.4, ls='dashed')
        plt.plot(x, ccf2, color='C1', lw=1., alpha=0.6, ls='solid')
        # plt.plot(x, ccf1+ccf2, color='gray', lw=1., alpha=0.6, ls='solid')
        for i, (vg, ccf) in enumerate(zip(vgrids, ccfs)):
            plt.plot(vg, ccf/np.max(ccf), '-', lw=0.5,
                     label='order %d' % self.orders[i])
        plt.plot(rvgrid, medccf/np.max(medccf), color='gray', lw=2)
        plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        if output_dir is not None:
            plt.savefig(output_dir+"ccfs%s.png" %
                        tag, dpi=200, bbox_inches='tight')
            plt.close()

        dccf = medccf/np.max(medccf) - 0.5
        dccfderiv = dccf[1:] * dccf[:-1]
        v50 = rvgrid[1:][dccfderiv < 0]
        vbroad = np.max(v50) - np.min(v50) - dvopt  # sum of HWHMs

        self.ccfrvlist = ccfrvs
        self.ccfvbroad = vbroad

        return rvgrid, medccf
