__all__ = ["SpecModel"]

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
from .utils import *
from celerite2.jax import terms as jax_terms
from celerite2.jax import GaussianProcess


class SpecModel:
    """ class to compute spectrum model
    """
    def __init__(self, sg, wav_obs, flux_obs, error_obs, mask_obs, vmax=50.):
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
        self.sg = sg
        self.Norder, self.Nwav = np.shape(sg.wavgrid)
        # log-uniform wavelength grid; note that this is different from uniform grid defined in SpecGrid
        self.wavgrid = np.array([np.logspace(np.log10(sg.wavmin[i]), np.log10(sg.wavmax[i]), self.Nwav)[1:-1] for i in range(self.Norder)])
        self.dlogwav = np.median(np.diff(np.log(self.wavgrid)), axis=1)
        npix_half = int(np.round(vmax / self.dlogwav[0] / c_in_kms))
        self.varr = np.array([varr_for_kernels(self.dlogwav[i], vmax=self.dlogwav[i]*c_in_kms*npix_half) for i in range(self.Norder)]) # vmax is chosen so that len(varr) is the same for all orders
        self.wav_obs = np.atleast_2d(wav_obs)
        self.wav_obs_range = np.max(wav_obs, axis=1) - np.min(wav_obs, axis=1)
        self.flux_obs = np.atleast_2d(flux_obs)
        self.error_obs = np.atleast_2d(error_obs)
        self.mask_obs = np.atleast_2d(mask_obs)
        self.mask_fit = np.zeros_like(self.wav_obs)

    def sgvalues(self, teff, logg, feh, alpha):
        """ fetch flux values at the native wavelength grids
        """
        return self.sg.values(teff, logg, feh, alpha, self.wavgrid)

    def rawflux(self, params_phys):
        c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2 = params_phys
        return self.sgvalues(teff, logg, feh, alpha)

    @partial(jit, static_argnums=(0,))
    def fluxmodel(self, wav_out, params_phys):
        """ broadened & shifted flux model; including a common linear continuum

            Args:
                wav_out: wavelengths where flux values are evaluated
                params_phys: set of physical parameters
                    continuum normalization (unitless), continuum slope (unitless), teff, logg, feh, alpha,
                    vsini (km/s), macroturbulence (km/s), wavelength resolution, radial velocity (km/s),
                    limb darkening coefficients for the quadratic law (u1, u2)

            Returns:
                flux values (Norder, Npix) at wav_out

        """
        c0, c1, teff, logg, feh, alpha, vsini, zeta, wavres, rv, u1, u2 = params_phys
        flux_raw = self.sg.values(teff, logg, feh, alpha, self.wavgrid)
        flux_base = c0 + c1 * (wav_out - jnp.mean(self.wav_obs, axis=1)[:,jnp.newaxis]) / self.wav_obs_range[:,jnp.newaxis]
        flux_phys = flux_base * broaden_and_shift_vmap(wav_out, self.wavgrid, flux_raw, vsini, zeta, get_beta(wavres), rv, self.varr, u1, u2)
        return flux_phys

    @partial(jit, static_argnums=(0,))
    def fluxmodel_multiorder(self, c0, c1, teff, logg, feh, alpha, vsini, zeta, res, rv, u1, u2):
        """ broadened & shifted flux model; including order-dependent linear continua

            Returns:
                flux model (Norder, Npix) at wav_obs

        """
        wav_out = self.wav_obs
        flux_raw = self.sg.values(teff, logg, feh, alpha, self.wavgrid)
        flux_base = c0[:,jnp.newaxis] + c1[:,jnp.newaxis] * (wav_out - jnp.mean(self.wav_obs, axis=1)[:,jnp.newaxis]) / self.wav_obs_range[:,jnp.newaxis]
        flux_phys = flux_base * broaden_and_shift_vmap_full(wav_out, self.wavgrid, flux_raw, vsini, zeta, get_beta(res), rv, self.varr, u1, u2)
        return flux_phys

    @partial(jit, static_argnums=(0,))
    def gp_loglikelihood(self, params):
        """ compute model likelihood using GP

            Args:
                params: physical parameters + log(amplitude), log(timescale), log(sigma) for the  Matern-3/2 kernel

            Returns:
                GP log-likelihood

        """
        lna, lnc, lnsigma = params[-3:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        diags = self.error_obs**2 + jnp.exp(2*lnsigma)

        mask_obs = self.mask_obs
        mask_all = mask_obs + (self.mask_fit > 0)
        idx = ~mask_all
        flux_model = self.fluxmodel(self.wav_obs, params[:-3])

        gp = GaussianProcess(kernel, mean=0.0)
        gp.compute(self.wav_obs[idx].ravel(), diag=diags[idx].ravel())
        res = self.flux_obs[idx].ravel() - flux_model[idx].ravel()

        return gp.log_likelihood(res)

    def gp_predict(self, params):
        """ compute model likelihood using GP

            Args:
                params: physical parameters + log(amplitude), log(timescale), log(sigma) for the  Matern-3/2 kernel

            Returns:
                GP likelihood
                GP instance and residual (if predict is True)

        """
        lna, lnc, lnsigma = params[-3:]
        kernel = jax_terms.Matern32Term(sigma=jnp.exp(lna), rho=jnp.exp(lnc))
        diags = self.error_obs**2 + jnp.exp(2*lnsigma)

        mask_obs = self.mask_obs
        mask_all = mask_obs + (self.mask_fit > 0)
        idx = ~mask_all
        flux_model = self.fluxmodel(self.wav_obs, params[:-3])

        #gp = celerite2.jax.GaussianProcess(kernel, mean=flux_model[idx].ravel())
        #gp.compute(wav_obs[idx].ravel(), diag=diags[idx].ravel())
        #return gp.predict(self.flux_obs[idx].ravel()), gp.log_likelihood(self.flux_obs[idx].ravel())

        gp = GaussianProcess(kernel, mean=0.0)
        gp.compute(self.wav_obs[idx].ravel(), diag=diags[idx].ravel())
        res = self.flux_obs[idx].ravel() - flux_model[idx].ravel()

        return gp, res
        #return gp.predict(res, t=self.wav_obs.ravel())+flux_model.ravel()
        #return gp.log_likelihood(res), (gp, res)
