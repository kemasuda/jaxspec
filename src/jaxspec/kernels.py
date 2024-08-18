__all__ = ["rotmacrokernel"]

import jax.numpy as jnp
from jax import jit
from jax.scipy.integrate import trapezoid as trapz

RP = jnp.array([-4.79443220978201773821E9, 1.95617491946556577543E12, -2.49248344360967716204E14, 9.70862251047306323952E15])
RQ = jnp.array([1., 4.99563147152651017219E2, 1.73785401676374683123E5, 4.84409658339962045305E7, 1.11855537045356834862E10, 2.11277520115489217587E12, 3.10518229857422583814E14, 3.18121955943204943306E16, 1.71086294081043136091E18])
DR1 =  5.78318596294678452118E0
DR2 = 3.04712623436620863991E1
PP = jnp.array([7.96936729297347051624E-4, 8.28352392107440799803E-2, 1.23953371646414299388E0, 5.44725003058768775090E0, 8.74716500199817011941E0, 5.30324038235394892183E0, 9.99999999999999997821E-1])
PQ = jnp.array([9.24408810558863637013E-4, 8.56288474354474431428E-2, 1.25352743901058953537E0, 5.47097740330417105182E0, 8.76190883237069594232E0, 5.30605288235394617618E0, 1.00000000000000000218E0])
QP = jnp.array([-1.13663838898469149931E-2, -1.28252718670509318512E0, -1.95539544257735972385E1, -9.32060152123768231369E1, -1.77681167980488050595E2, -1.47077505154951170175E2, -5.14105326766599330220E1, -6.05014350600728481186E0])
QQ = jnp.array([1., 6.43178256118178023184E1, 8.56430025976980587198E2, 3.88240183605401609683E3, 7.24046774195652478189E3, 5.93072701187316984827E3, 2.06209331660327847417E3, 2.42005740240291393179E2])
PIO4 = 0.78539816339744830962
SQ2OPI = 0.79788456080286535588


#%% Bessel function of the 1st kind, order=0
def J0(x):
    x = jnp.where(x > 0., x, -x)

    z = x * x
    ret = 1. - z / 4.

    p = (z - DR1) * (z - DR2)
    p = p * jnp.polyval(RP, z) / jnp.polyval(RQ, z)
    ret = jnp.where(x < 1e-5, ret, p)

    #xinv5 = jnp.where(x <= 5., 0., 1./(x+1e-10)) # required for autograd not to fail when x includes 0
    x_safe = jnp.where(x <= 5., 1., x)
    xinv5 = jnp.where(x <= 5., 0., 1. / x_safe)

    w = 5.0 * xinv5
    z = w * w
    p = jnp.polyval(PP, z) / jnp.polyval(PQ, z)
    q = jnp.polyval(QP, z) / jnp.polyval(QQ, z)
    xn = x - PIO4
    p = p * jnp.cos(xn) - w * q * jnp.sin(xn)
    ret = jnp.where(x <= 5., ret, p * SQ2OPI * jnp.sqrt(xinv5))

    return ret

# Hirano et al. (2011) ApJ 742, 69
# unlike in the paper, beta here is the Gaussian width
def rotmacrokernel(varr, zeta, vsini, u1, u2, beta, Nt=1000):
    tarr = jnp.linspace(0, 1, Nt)

    n, dv = len(varr), jnp.median(jnp.diff(varr))
    sigmas = jnp.fft.fftfreq(n, d=dv)

    t = tarr[:,None]
    piz2 = jnp.pi * jnp.pi * zeta * zeta
    t2 = t * t
    sig2 = sigmas * sigmas
    omint2 = 1. - t2
    projd = jnp.sqrt(omint2)
    ldfactor = (1. - (1. - projd) * (u1 + u2*(1. - projd))) / (1. - u1/3. - u2/6.)
    rt = jnp.exp(-piz2*sig2*omint2) + jnp.exp(-piz2*sig2*t2)
    ip = jnp.exp(-2*jnp.pi*jnp.pi*beta*beta*sig2)
    ys = ldfactor * rt * ip * J0(2*jnp.pi*sigmas*vsini*t) * t
    kernel_ft = trapz(ys.T)

    kernel = jnp.fft.fft(kernel_ft)
    kernel = jnp.real(kernel)
    #kernel = jnp.fft.irfft(kernel_ft[:(n + 1) // 2])
    kernel = jnp.fft.fftshift(kernel)

    return kernel / jnp.sum(kernel)

#%%
"""
def rotmacro_ft(sigma, xi, vsini, u1, u2, intres=100):
    tarr = jnp.outer( jnp.ones(sigma.shape[0]), jnp.linspace(0, 1, intres) )
    tarr = tarr.transpose()

    t1 = (
        (1 - u1 * ( 1 - jnp.sqrt(1 - tarr**2) ) -
         u2 * ( 1 - jnp.sqrt( 1 - tarr**2 ) )**2) / ( 1 - u1 / 3 - u2 / 6 )
    )
    t2 = (
        ( jnp.exp( -jnp.pi**2 * xi**2 * sigma**2 * (1-tarr**2) ) +
          jnp.exp( -jnp.pi**2 * xi**2 * sigma**2 * tarr**2) ) *
        J0(2*jnp.pi*sigma*vsini*tarr) * tarr
    )

    m = t1 * t2
    kernel_ft = jnp.trapz(m,x=tarr,axis=0)
    return kernel_ft

#%%
def rotmacro(n, dv, zeta, vsini, u1, u2, **kwargs):
    #n, dv = 100, 0.1
    nind = (n + 1) // 2
    vmax = dv * nind
    varr = jnp.linspace(-vmax, vmax, n)
#def rotmacro(varr, zeta, vsini, u1, u2, **kwargs):
#    n, dv = len(varr), jnp.median(jnp.diff(varr))
#    nind = (n + 1) // 2

    # Fourier transform frequencies, for given velocity displacements
    sigarr = jnp.fft.fftfreq(n,d=dv)
    sigarr = sigarr[:nind]
    kernel_ft = rotmacro_ft(sigarr, zeta, vsini, u1, u2, **kwargs)
    kernel_ft = jnp.hstack([ kernel_ft, kernel_ft[1:][::-1] ])
    kernel = jnp.fft.ifft(kernel_ft)
    kernel = jnp.fft.fftshift(kernel)
    kernel = kernel.real # Require the real part
    kernel = kernel / jnp.sum(kernel)
    return varr, kernel

#%%
import matplotlib.pyplot as plt
from scipy.special import j0 as j0sci
from jax import grad, jit

#%%
%timeit rk = rotkernel(varr, zeta, vsini, u1, u2, beta)

#%%
zeta, vsini, u1, u2, beta = 10, 0, 0.5, 0.2, 0.
varr = jnp.linspace(-100, 100, 501)
len(varr)
#vsini = 1
%time rk = rotkernel(varr, zeta, vsini, u1, u2, beta)
#%time v2, rk2 = rotmacro(301, 0.66, zeta, vsini, u1, u2, intres=1000)
rkd = rotkerneld(varr, zeta, vsini, u1, u2, beta)
rkd /= jnp.sum(rkd)
plt.figure(figsize=(16,8))
#plt.xlim(-5, 5)
plt.subplot(211)
plt.plot(varr, rk-rkd, 'o')
#plt.plot(v2, rk2)
#plt.plot(varr, rk-rk2, '.-')
#plt.plot(varr, rk2-rkd, 's', markersize=3, lw=1)
plt.subplot(212)
plt.plot(varr, rk, '.')
plt.plot(varr, rkd, '-')

#%%
func0 = lambda vsini: jnp.sum(rotkernel(varr, 0, vsini, 0.5, 0.2, 2))
func = jit(grad(func0))

#%%
func0(4.)
func(4.)

#%%
x = jnp.linspace(-10, 10, 100)
#jnp.sum(J0(x[:,None]))
#func = lambda x: jnp.sum(J0(x[:,None]))
#jnp.sum(func(jnp.array([1.])))
#jit(grad(func))(jnp.array([1.]))

#%%
#plt.plot(x, j0sci(x))
plt.plot(x, J0(x)-j0sci(x))
"""
