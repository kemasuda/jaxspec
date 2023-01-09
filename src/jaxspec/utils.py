__all__ = ["get_beta", "varr_for_kernels", "doppler_shift", "broaden_and_shift", "broaden_and_shift_vmap", "broaden_and_shift_vmap_full", "compute_ccf", "c_in_kms"]

import numpy as np
import jax.numpy as jnp
from jax import (jit, vmap)
from .kernels import rotmacrokernel
from scipy.interpolate import interp1d
from scipy.signal import correlate
import psutil, gc

c_in_kms = 2.99792458e5

def get_beta(resolution):
    """ gaussian width (km/s) of the instrumental profile

        Args:
            resolution: lambda / delta(lambda)

        Returns:
            gaussian width of the instrumental profile (km/s)

    """
    return c_in_kms / resolution / 2.354820


def varr_for_kernels(dlogwav, vmax=50):
    """ velocity grid for setting up broadening kernel

        Args:
            dlogwav: wavelength grid in log space
            vmax: velocity width (km/s)

        Returns:
            velocity grid (km/s)

    """
    v_pix = dlogwav * c_in_kms
    #pix_max = int(vmax / v_pix)
    pix_max = int(np.round(vmax / v_pix)) # make sure that pix_max is exactly the same for all orders
    #pixs = np.arange(-pix_max, pix_max*1.01, 1)
    pixs = np.arange(-pix_max, pix_max+0.1, 1)
    varr = pixs * v_pix
    return varr


def doppler_shift(xout, xin, yin, v):
    """ compute Doppler-shifted spectrum

        Args:
            xout: output wavelength
            xin: input wavelength
            yin: input flux
            v: Doppler shift (km/s)

        Returns:
            Doppler-shifted flux at xout

    """
    x_shifted = xin * (1. + v / c_in_kms)
    return jnp.interp(xout, x_shifted, yin)


def broaden_and_shift(wavout, wav, flux, vsini, zeta, beta, rv, varr, u1=0.5, u2=0.2):
    """ computed broadened and Doppler-shifted spectrum

        Args:
            wavout: output wavelength
            wav: input wavelength
            flux: output wavelength
            vsini: projected rotation velocity (km/s)
            zeta: macroturbulence velocity (km/s) in the radial-tangential model (zeta_R = zeta_T assumed)
            beta: gaussian sigma for IP (km/s)
            rv: radial velocity (km/s)
            varr: velocity grid for computing kernels
            u1, u2: coefficients for quadratic limb-darkening law

        Returns:
            broadened and Doppler-shifted spectrum evaluated at wavout

    """
    kernel = rotmacrokernel(varr, zeta, vsini, u1, u2, beta, Nt=500)
    bflux = jnp.convolve(flux, kernel, 'same')
    return doppler_shift(wavout, wav, bflux, rv)

# mappable along the 1st axis (wavout, wav, flux, varr)
broaden_and_shift_vmap = vmap(broaden_and_shift, (0,0,0,None,None,None,None,0,None,None), 0)

# mappable along the 1st axis (wavout, wav, flux, beta, rv, varr)
broaden_and_shift_vmap_full = vmap(broaden_and_shift, (0,0,0,None,None,0,0,0,None,None), 0)


def compute_ccf(x, y, xmodel, ymodel, mask=None, resolution_factor=5):
    """ compute cross-correlation function with model

        Args:
            x: data wavelength
            y: data flux
            xmodel: model wavelength
            ymodel: model flux
            mask: data to be masked
            resolution_factor: ovesampling factor for the data

        Returns:
            velgrid: velocity grid (km/s)
            ccf: CCF values as a function of velocity

    """
    if mask is None:
        mask = np.zeros_like(y).astype(bool)
    yy = np.array(y)
    yy[mask] = np.nan

    ndata = len(x)
    xgrid = np.logspace(np.log10(x.min())+1e-4, np.log10(x.max())-1e-4, ndata*resolution_factor)
    ygrid = interp1d(x, yy)(xgrid) - np.nanmean(yy)#1.
    ymgrid = interp1d(xmodel, ymodel)(xgrid) - np.nanmean(ymodel)
    ygrid[ygrid!=ygrid] = 0

    ccf = correlate(ygrid, ymgrid)
    logxgrid = np.log(xgrid)
    dlogx = np.diff(logxgrid)[0]
    velgrid = (np.arange(len(ccf))*dlogx - (logxgrid[-1]-logxgrid[0])) * c_in_kms

    return velgrid, ccf


def clear_caches():
    """ garbage collection for JAX
    """
    process = psutil.Process()
    if process.memory_info().vms > 4 * 2**30:  # >4GB memory usage
        for module_name, module in sys.modules.items():
            if module_name.startswith("jax"):
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if hasattr(obj, "cache_clear"):
                        obj.cache_clear()
        gc.collect()
