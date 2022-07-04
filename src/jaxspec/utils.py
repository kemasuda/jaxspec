__all__ = ["get_beta", "varr_for_kernels", "doppler_shift", "broaden_and_shift", "compute_ccf"]

import numpy as np
import jax.numpy as jnp
from jax import jit
from .rotkernel_ft import rotkernel
from scipy.interpolate import interp1d
from scipy.signal import correlate

def get_beta(resolution):
    c0 = 2.99792458e5
    return c0 / resolution / 2.354820

def varr_for_kernels(dlogwav, vmax=50, c0=2.99792458e5):
    v_pix = dlogwav * c0
    pix_max = int(vmax / v_pix)
    pixs = np.arange(-pix_max, pix_max*1.01, 1)
    varr = pixs * v_pix
    return varr

def doppler_shift(xout, xin, yin, v, const_c=2.99792458e5):
    x_shifted = xin * (1. + v/const_c)
    return jnp.interp(xout, x_shifted, yin)

# add varr??
@jit
def broaden_and_shift(wavout, wav, flux, vsini, zeta, beta, rv, varr, u1=0.5, u2=0.2):
    kernel = rotkernel(varr, zeta, vsini, u1, u2, beta, Nt=500)
    bflux = jnp.convolve(flux, kernel, 'same')
    return doppler_shift(wavout, wav, bflux, rv)

def compute_ccf(x, y, xmodel, ymodel, mask=None, resolution_factor=5):
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
    velgrid = (np.arange(len(ccf))*dlogx - (logxgrid[-1]-logxgrid[0])) * 299792458e-3

    return velgrid, ccf
