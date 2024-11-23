__all__ = ["SpecModel", "SpecModel2"]

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from .utils import *
from celerite2.jax import terms as jax_terms
from celerite2.jax import GaussianProcess
import tinygp


class SpecModel:
    """ class to compute spectrum model
    """

    def __init__(self, sg, wav_obs, flux_obs, error_obs, mask_obs, vmax=50., gpu=False):
        """ initialization

            Args:
                sg: SpecGrid instance
                wav_obs: observed wavelengths (Norder, Npix)
                flux_obs: observed flux (Norder, Npix)
                error_obs: error (Norder, Npix)
                mask_obs: if True the data point is omitted from the entire analysis (Norder, Npix)
                        self.mask_fit is similar, but may be changed iteratively during fitting
                vmax: maximum velocity width for the broadening kernel (-vmax to +vmax)
                        defaults to 50; needs to be increased if vsini is large

        """
        self.sg = sg
        self.Norder, self.Nwav = np.shape(sg.wavgrid)
        # log-uniform wavelength grid; note that this is different from uniform grid defined in SpecGrid
        self.wavgrid = np.array([np.logspace(np.log10(sg.wavmin[i]), np.log10(
            sg.wavmax[i]), self.Nwav)[1:-1] for i in range(self.Norder)])
        self.dlogwav = np.median(np.diff(np.log(self.wavgrid)), axis=1)
        npix_half = int(np.round(vmax / self.dlogwav[0] / c_in_kms))
        # vmax is chosen so that len(varr) is the same for all orders
        self.varr = np.array([varr_for_kernels(
            self.dlogwav[i], vmax=self.dlogwav[i]*c_in_kms*npix_half) for i in range(self.Norder)])
        self.wav_obs = np.atleast_2d(wav_obs)
        self.wav_obs_range = np.max(wav_obs, axis=1) - np.min(wav_obs, axis=1)
        self.flux_obs = np.atleast_2d(flux_obs)
        self.error_obs = np.atleast_2d(error_obs)
        self.mask_obs = np.atleast_2d(mask_obs)
        self.mask_fit = np.zeros_like(self.wav_obs)
        self.gpu = gpu

    @partial(jit, static_argnums=(0,))
    def fluxmodel_multiorder(self, par):
        """ broadened & shifted flux model; including order-dependent linear continua

            Returns:
                flux model (Norder, Npix) at wav_obs

        """
        c0, c1, teff, logg, alpha, vsini, zeta, res, rv, u1, u2, dilution \
            = par["norm"], par["slope"], par["teff"], par["logg"], par["alpha"], par["vsini"], par["zeta"], par['wavres'], par["rv"], par['u1'], par['u2'], par['dilution']
        wav_out = self.wav_obs
        if self.sg.model == 'bosz':
            flux_raw = self.sg.values(
                teff, logg, par['mh'], alpha, par['carbon'], par['vmic'], self.wavgrid)
        else:
            flux_raw = self.sg.values(
                teff, logg, par["feh"], alpha, self.wavgrid)
        flux_base = c0[:, jnp.newaxis] + c1[:, jnp.newaxis] * (wav_out - jnp.mean(
            self.wav_obs, axis=1)[:, jnp.newaxis]) / self.wav_obs_range[:, jnp.newaxis]
        flux_phys = flux_base * ((1 - dilution) * broaden_and_shift_vmap_full(
            wav_out, self.wavgrid, flux_raw, vsini, zeta, get_beta(res), rv, self.varr, u1, u2) + dilution)
        return flux_phys


class SpecModel2(SpecModel):
    """ class to compute spectrum model for SB2
    """

    def __init__(self, sg, wav_obs, flux_obs, error_obs, mask_obs, vmax=50., gpu=False):
        """ initialization

                Args:
                    sg: SpecGrid instance
                    wav_obs: observed wavelengths (Norder, Npix)
                    flux_obs: observed flux (Norder, Npix)
                    error_obs: error (Norder, Npix)
                    mask_obs: if True the data point is omitted from the entire analysis (Norder, Npix)
                            self.mask_fit is similar, but may be changed iteratively during fitting
                    vmax: maximum velocity width for the broadening kernel
                            defaults to 50; needs to be increased if vsini is large

            """
        super().__init__(sg, wav_obs, flux_obs, error_obs, mask_obs, vmax=vmax, gpu=gpu)

    @partial(jit, static_argnums=(0,))
    def fluxmodel_multiorder(self, par):
        """ broadened & shifted flux model; including order-dependent linear continua

            Returns:
                flux model (Norder, Npix) at wav_obs

        """
        c0, c1, teff1, teff2, logg1, logg2, alpha1, alpha2, vsini1, vsini2, zeta1, zeta2, res, rv1, rv2, u11, u12, u21, u22, f2_f1 \
            = par["norm"], par["slope"], par["teff1"], par["teff2"], par["logg1"], par["logg2"], par["alpha1"], par["alpha2"], par["vsini1"], par["vsini2"], par["zeta1"], par["zeta2"], par['wavres'], par["rv1"], par["rv2"], par['u11'], par['u12'], par['u21'], par['u22'], par['f2_f1']
        wav_out = self.wav_obs
        if self.sg.model == 'bosz':
            flux_raw1 = self.sg.values(
                teff1, logg1, par["mh1"], alpha1, par['carbon1'], par['vmic1'], self.wavgrid)
            flux_raw2 = self.sg.values(
                teff2, logg2, par["mh2"], alpha2, par['carbon2'], par['vmic2'], self.wavgrid)
        else:
            flux_raw1 = self.sg.values(
                teff1, logg1, par["feh1"], alpha1, self.wavgrid)
            flux_raw2 = self.sg.values(
                teff2, logg2, par["feh2"], alpha2, self.wavgrid)

        flux_base = c0[:, jnp.newaxis] + c1[:, jnp.newaxis] * (wav_out - jnp.mean(
            self.wav_obs, axis=1)[:, jnp.newaxis]) / self.wav_obs_range[:, jnp.newaxis]
        flux_sum = broaden_and_shift_vmap_full(wav_out, self.wavgrid, flux_raw1, vsini1, zeta1, get_beta(
            res), rv1, self.varr, u11, u21) + f2_f1 * broaden_and_shift_vmap_full(wav_out, self.wavgrid, flux_raw2, vsini2, zeta2, get_beta(res), rv2, self.varr, u12, u22)
        flux_phys = flux_base * flux_sum / (1. + f2_f1)
        return flux_phys
