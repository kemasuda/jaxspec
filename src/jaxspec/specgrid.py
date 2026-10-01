__all__ = ["SpecGrid", "SpecGridBosz", "SpecGridTlusty"]

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial


def interpolate_flux(fgrid, tidx, gidx, fidx, aidx, wavidx):
    """ interpolate flux grid using map_coordinates

        Args:
            fgrid: flux grid
            tidx: index for Teff
            gidx: index for logg
            fidx: index for FeH
            aidx: index for alpha/Fe
            wavidx: index for wavelength

        Returns:
            linearly interpolated flux at a given set of indices
            -inf returned if the index set is out of the valid range

    """
    return mapc(fgrid, [tidx, gidx, fidx, aidx, wavidx], order=1, cval=-jnp.inf)


# map interpolate_flux along the 1st axis
# return flux arrays for multiple orders
interpolate_flux_vmap = vmap(interpolate_flux, (0, 0, 0, 0, 0, 0), 0)


class SpecGrid:
    """ a minimal class to handle grid spectrum data
    """

    def __init__(self, paths):
        """ initialization

            Args:
                paths: paths for grid files

        """
        grids = [np.load(path) for path in paths]
        self.t0 = np.array([grid['tgrid'][0] for grid in grids])
        self.t1 = np.array([grid['tgrid'][-1] for grid in grids])
        self.dt = np.array([np.diff(grid['tgrid'])[0] for grid in grids])
        self.g0 = np.array([grid['ggrid'][0] for grid in grids])
        self.g1 = np.array([grid['ggrid'][-1] for grid in grids])
        self.dg = np.array([np.diff(grid['ggrid'])[0] for grid in grids])
        self.f0 = np.array([grid['fgrid'][0] for grid in grids])
        self.f1 = np.array([grid['fgrid'][-1] for grid in grids])
        self.df = np.array([np.diff(grid['fgrid'])[0] for grid in grids])
        if 'agrid' in list(np.load(paths[0]).keys()):
            self.a0 = np.array([grid['agrid'][0] for grid in grids])
            self.a1 = np.array([grid['agrid'][-1] for grid in grids])
            self.da = np.array([np.diff(grid['agrid'])[0] for grid in grids])
        else:
            self.a0 = np.array([0. for grid in grids])
            self.a1 = np.array([1. for grid in grids])
            self.da = np.array([1. for grid in grids])
        self.wavgrid = np.array([grid['wavgrid'] for grid in grids])
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0 = np.array([grid['wavgrid'][0] for grid in grids])
        self.dwav = np.diff(self.wavgrid)[:, 0]
        self.wavmin = np.min(self.wavgrid, axis=1)
        self.wavmax = np.max(self.wavgrid, axis=1)
        self.fluxgrid = np.array([grid['flux'] for grid in grids])

        self.model = 'coelho'

        # check that the grids are equally spaced
        # otherwise the current interpolation does not work
        for key in ['tgrid', 'ggrid', 'fgrid', 'wavgrid']:
            step = np.diff(grids[0][key])
            smin, smax = np.min(step), np.max(step)
            assert np.abs(smax/smin - 1.) < 0.01

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, feh, alpha, wav):
        """ compute flux values interpolating the model grids

            Args:
                teff: effective temperature
                logg: surface gravity
                feh: metallicity
                alpha: alpha enhancement
                wav: wavelengths (in angstrom??), (Norder, Npix)

            Returns:
                interpolated flux (Norder, Npix)

        """
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        fidx = (feh - self.f0) / self.df
        aidx = (alpha - self.a0) / self.da
        wavidx = (wav - self.wav0[:, np.newaxis]) / self.dwav[:, np.newaxis]
        return interpolate_flux_vmap(self.fluxgrid, tidx, gidx, fidx, aidx, wavidx)


def interpolate_flux_bosz(fgrid, tidx, gidx, midx, aidx, cidx, vidx, wavidx):
    """ interpolate flux grid using map_coordinates

        Args:
            fgrid: flux grid
            tidx: index for Teff
            gidx: index for logg
            fidx: index for M/H
            aidx: index for alpha/M
            cidx: index for C/M
            vidx: index for vmic
            wavidx: index for wavelength

        Returns:
            linearly interpolated flux at a given set of indices
            -inf returned if the index set is out of the valid range

    """
    return mapc(fgrid, [tidx, gidx, midx, aidx, cidx, vidx, wavidx], order=1, cval=-jnp.inf)


# map interpolate_flux along the 1st axis
# return flux arrays for multiple orders
# interpolate_flux_bosz_vmap = vmap(
#    interpolate_flux_bosz, (0, 0, 0, 0, 0, 0, 0, 0), 0)

def safe_map_coordinates(arr, coords, **kwargs):
    """
    Drop degenerate (length-1) axes before calling map_coordinates.

    arr   : ndarray (can have axes of length 1)
    coords: shape (arr.ndim, ...)

    For axes with length 1, we always take index 0 and do not interpolate
    along that axis.
    """
    shape = arr.shape
    ndim = arr.ndim

    axes_keep = [i for i in range(ndim) if shape[i] > 1]

    # Fast path: no degenerate axes
    if len(axes_keep) == ndim:
        return mapc(arr, coords, **kwargs)

    # Drop length-1 axes
    arr2 = arr
    coords_list = []
    for i in range(ndim):
        if shape[i] > 1:
            coords_list.append(coords[i])
        else:
            # collapse this axis by always taking index 0
            arr2 = jnp.take(arr2, 0, axis=i)

    coords2 = jnp.stack(coords_list, axis=0)
    return mapc(arr2, coords2, **kwargs)


def interpolate_flux_bosz_vmap(fluxgrid, tidx, gidx, midx, aidx, cidx, vidx, wavidx):
    """
    Multilinear interpolation over BOSZ-like grid using safe_map_coordinates.

    Assumes fluxgrid has shape:
        (Nspec, Nt, Ng, Nm, Na, Nc, Nv, Npix)

    and:
        tidx, gidx, midx, aidx, cidx, vidx : (Nspec,)
        wavidx : (Nspec, Npix)
    """

    def interp_one(flux, t, g, m, a, c, v, w):
        # flux: (Nt, Ng, Nm, Na, Nc, Nv, Npix)
        # t,g,m,a,c,v: scalars
        # w: (Npix,)
        n_pix = w.shape[-1]

        coords = jnp.stack(
            [
                jnp.full((n_pix,), t),  # Teff axis
                jnp.full((n_pix,), g),  # logg axis
                jnp.full((n_pix,), m),  # [M/H] axis
                jnp.full((n_pix,), a),  # [alpha/M] axis
                jnp.full((n_pix,), c),  # [C/M] axis
                jnp.full((n_pix,), v),  # vmic axis
                w,                      # wavelength axis
            ],
            axis=0,
        )

        return safe_map_coordinates(
            flux,
            coords,
            order=1,      # linear
            mode="nearest",
        )

    # vmap over the leading "grid" axis
    return vmap(
        interp_one,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )(fluxgrid, tidx, gidx, midx, aidx, cidx, vidx, wavidx)


class SpecGridBosz:
    """ a minimal class to handle grid spectrum data
    """

    def __init__(self, paths):
        """ initialization

            Args:
                paths: paths for grid files

        """
        grids = [np.load(path) for path in paths]
        self.t0 = np.array([grid['tgrid'][0] for grid in grids])
        self.t1 = np.array([grid['tgrid'][-1] for grid in grids])
        self.dt = np.array([np.diff(grid['tgrid'])[0] for grid in grids])
        self.g0 = np.array([grid['ggrid'][0] for grid in grids])
        self.g1 = np.array([grid['ggrid'][-1] for grid in grids])
        self.dg = np.array([np.diff(grid['ggrid'])[0] for grid in grids])
        self.m0 = np.array([grid['mgrid'][0] for grid in grids])
        self.m1 = np.array([grid['mgrid'][-1] for grid in grids])
        self.dm = np.array([np.diff(grid['mgrid'])[0] for grid in grids])
        self.a0 = np.array([grid['agrid'][0] for grid in grids])
        self.a1 = np.array([grid['agrid'][-1] for grid in grids])
        self.da = np.array([np.diff(grid['agrid'])[0] for grid in grids])
        self.c0 = np.array([grid['cgrid'][0] for grid in grids])
        self.c1 = np.array([grid['cgrid'][-1] for grid in grids])
        # self.dc = np.array([np.diff(grid['cgrid'])[0] for grid in grids])
        self.dc = np.array([
            np.diff(grid['cgrid'])[0] if len(grid['cgrid']) > 1 else 1.0
            for grid in grids
        ])
        self.v0 = np.array([grid['vgrid'][0] for grid in grids])
        self.v1 = np.array([grid['vgrid'][-1] for grid in grids])
        self.dv = np.array([np.diff(grid['vgrid'])[0] for grid in grids])

        self.wavgrid = np.array([grid['wavgrid'] for grid in grids])
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0 = np.array([grid['wavgrid'][0] for grid in grids])
        self.dwav = np.diff(self.wavgrid)[:, 0]
        self.wavmin = np.min(self.wavgrid, axis=1)
        self.wavmax = np.max(self.wavgrid, axis=1)
        self.fluxgrid = np.array([grid['flux'] for grid in grids])

        self.model = 'bosz'

        # check that the grids are equally spaced
        # otherwise the current interpolation does not work
        for key in ['tgrid', 'ggrid', 'mgrid', 'agrid', 'cgrid', 'vgrid', 'wavgrid']:
            # step = np.diff(grids[0][key])
            arr = grids[0][key]
            if len(arr) <= 1:
                continue
            step = np.diff(arr)
            smin, smax = np.min(step), np.max(step)
            assert np.abs(
                smax/smin - 1.) < 0.01, f"grid for {key} is not equally spaced."

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, mh, alpha, carbon, vmic, wav):
        """ compute flux values interpolating the model grids

            Args:
                teff: effective temperature
                logg: surface gravity
                feh: metallicity
                alpha: alpha enhancement
                wav: wavelengths (in angstrom??), (Norder, Npix)

            Returns:
                interpolated flux (Norder, Npix)

        """
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        midx = (mh - self.m0) / self.dm
        aidx = (alpha - self.a0) / self.da
        cidx = (carbon - self.c0) / self.dc
        vidx = (vmic - self.v0) / self.dv
        wavidx = (wav - self.wav0[:, np.newaxis]) / self.dwav[:, np.newaxis]
        return interpolate_flux_bosz_vmap(self.fluxgrid, tidx, gidx, midx, aidx, cidx, vidx, wavidx)


def interpolate_flux_tlusty(fgrid, tidx, gidx, zidx, wavidx):
    """interpolate flux grid using map_coordinates

        Args:
            fgrid: flux grid
            tidx: index for Teff
            gidx: index for logg
            zidx: index for log(Z/Zsun)
            wavidx: index for wavelength

        Returns:
            2D array: linearly interpolated flux at a given set of indices
                -inf returned if the index set is out of the valid range

    """
    return mapc(fgrid, [tidx, gidx, zidx, wavidx], order=1, cval=-jnp.inf)


# map interpolate_flux along the 1st axis
# return flux arrays for multiple orders
interpolate_flux_tlusty_vmap = vmap(
    interpolate_flux_tlusty, (0, 0, 0, 0, 0), 0)


class SpecGridTlusty:
    """ a minimal class to handle grid spectrum data
    """

    def __init__(self, paths):
        """ initialization

            Args:
                paths: paths for grid files

        """
        grids = [np.load(path) for path in paths]
        self.tgrid = np.array(grids[0]['tgrid'])
        self.ggrid = np.array(grids[0]['ggrid'])
        self.zgrid = np.array(grids[0]['zgrid'])
        self.num_grids = len(grids)
        # self.vgrid = np.array(grids[0]['vgrid'])

        self.wavgrid = np.array([grid['wavgrid'] for grid in grids])
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0 = np.array([grid['wavgrid'][0] for grid in grids])
        self.dwav = np.diff(self.wavgrid)[:, 0]
        self.wavmin = np.min(self.wavgrid, axis=1)
        self.wavmax = np.max(self.wavgrid, axis=1)
        self.fluxgrid = np.array([grid['flux'] for grid in grids])

        self.model = 'tlusty'

        step = np.diff(grids[0]['wavgrid'])
        smin, smax = np.min(step), np.max(step)
        assert np.abs(smax/smin - 1.) < 0.01, f"wavgrid is not equally spaced."

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, logZ, wav):
        """ compute flux values interpolating the model grids

            Args:
                teff: effective temperature
                logg: surface gravity
                logZ: log10(Z/Z_sun)
                vmic: microturbulence velocity
                wav: wavelengths (in angstrom??), (Norder, Npix)

            Returns:
                2D array: interpolated flux (Norder, Npix)

        """
        tidx = jnp.array(
            [jnp.interp(teff, self.tgrid, jnp.arange(len(self.tgrid)))]*self.num_grids)
        gidx = jnp.array(
            [jnp.interp(logg, self.ggrid, jnp.arange(len(self.ggrid)))]*self.num_grids)
        zidx = jnp.array(
            [jnp.interp(logZ, self.zgrid, jnp.arange(len(self.zgrid)))]*self.num_grids)
        wavidx = (wav - self.wav0[:, np.newaxis]) / self.dwav[:, np.newaxis]
        return interpolate_flux_tlusty_vmap(self.fluxgrid, tidx, gidx, zidx, wavidx)
