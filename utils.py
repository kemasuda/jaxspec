__all__ = ["get_beta", "varr_for_kernels", "doppler_shift", "broaden_and_shift"]

import numpy as np
import jax.numpy as jnp

def get_beta(resolution=80000., c0=2.99792458e5):
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
from rotkernel_ft import rotkernel
@jit
def broaden_and_shift(wavout, wav, flux, vsini, zeta, beta, rv, u1=0.5, u2=0.2):
    kernel = rotkernel(varr, zeta, vsini, u1, u2, beta, Nt=500)
    bflux = jnp.convolve(flux, kernel, 'same')
    return doppler_shift(wavout, wav, bflux, rv)
