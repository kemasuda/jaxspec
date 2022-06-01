#%%
import numpy as np
import pandas as pd
import sys, os, glob
from scipy.interpolate import interp1d

#%%
import pickle
import vidmapy

#%% synthe outputs (see A100)
filename = "synthe_5195-5285.pkl"
filename = "synthe_5355-5445.pkl"
filename = "synthe_5525-5615.pkl"
filename = "synthe_6095-6195.pkl"
filename = "synthe_6205-6265.pkl"

#%%
with open(filename, "rb") as f:
    res = pickle.load(f)

#%%
df_all = pd.DataFrame(data={})
for i in range(len(res)):
    di = pd.DataFrame(data={})
    params = res[i].parameters
    wavs = res[i].wave
    nflux = res[i].normed_flux
    di['wav'] = wavs
    di['flux'] = nflux
    di['teff'] = params.teff
    di['logg'] = params.logg
    di['feh'] = params.metallicity
    di['vmic'] = params.microturbulence
    df_all = df_all.append(di)
df_all = df_all.reset_index(drop=True)

#%%
d = df_all.sort_values(["teff", "logg", "feh", "wav"]).reset_index(drop=True)

#%%
tgrid = np.sort(list(set(d.teff)))
ggrid = np.sort(list(set(d.logg)))
fgrid = np.sort(list(set(d.feh)))
wavgrid = np.sort(list(set(d.wav)))
print (tgrid, len(tgrid))
print (ggrid, len(ggrid))
print (fgrid, len(fgrid))
print (wavgrid, len(wavgrid))

#%%
keys = ['flux']

#%% linear wavelength grid; should be logarithmic?
wavarr = np.linspace(wavgrid[0], wavgrid[-1], len(wavgrid))

#%%
pgrids2d = []
for key in keys:
    pgrid2d = np.zeros((len(tgrid), len(ggrid), len(fgrid), len(wavgrid)))
    for i,t in enumerate(tgrid):
        for j,g in enumerate(ggrid):
            for k,f in enumerate(fgrid):
                _d = d[(d.teff==t)&(d.logg==g)&(d.feh==f)]
                #farr = np.array(_d.flux)
                #pgrid2d[i][j][k] = farr
                pgrid2d[i][j][k] = interp1d(_d.wav, _d[key])(wavarr)
                #pgrid2d.append(eeparr.reshape(len(mgrid), len(fgrid)))
    #pgrid2d = np.array(pgrid2d)
    pgrids2d.append(pgrid2d)
print (np.shape(pgrids2d))

#%%
outname = filename.split(".")[0] + "_normed.npz"
print (outname)

#%%
np.savez(outname, tgrid=tgrid, ggrid=ggrid, fgrid=fgrid, wavgrid=wavarr, flux=pgrids2d[0])
