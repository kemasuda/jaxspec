__all__ = ["SpecGrid"]

#%%
import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates as mapc
from functools import partial

#%%
#"""
class SpecGrid:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synthe_5200-5280_normed.npz')
        self.dgrid = np.load(path)
        self.t0, self.dt = self.dgrid['tgrid'][0], np.diff(self.dgrid['tgrid'])[0]
        self.g0, self.dg = self.dgrid['ggrid'][0], np.diff(self.dgrid['ggrid'])[0]
        self.f0, self.df = self.dgrid['fgrid'][0], np.diff(self.dgrid['fgrid'])[0]
        self.wavgrid = self.dgrid['wavgrid']
        self.logwavgrid = np.log(self.wavgrid)
        self.wav0, self.dwav = self.wavgrid[0], np.diff(self.wavgrid)[0]
        self.wavmin, self.wavmax = np.min(self.wavgrid), np.max(self.wavgrid)
        #self.keys = ['flux']

    #def set_keys(self, keys):
    #    self.keys = keys

    @partial(jit, static_argnums=(0,))
    def values(self, teff, logg, feh, wav):
        tidx = (teff - self.t0) / self.dt
        gidx = (logg - self.g0) / self.dg
        fidx = (feh - self.f0) / self.df
        wavidx = (wav - self.wav0) / self.dwav
        idxs = [tidx, gidx, fidx, wavidx]
        #return [mapc(self.dgrid[key], idxs, order=1, cval=-jnp.inf) for key in self.keys]
        return mapc(self.dgrid['flux'], idxs, order=1, cval=-jnp.inf)
