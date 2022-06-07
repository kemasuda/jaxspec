""" compute synthe spectra

* input: model_params.csv, models.pkl from prepare_atlas_grid.ipynb
* output: pkl containing spectra
"""

__all__ = ["create_grid"]

import os, sys, inspect
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import pickle
import pandas as pd
from scipy.interpolate import interp1d
from vidmapy.kurucz.atlas import Atlas
from vidmapy.kurucz.synthe import Synthe
from vidmapy.kurucz.parameters import Parameters

def get_atlas_model(teff, logg, feh, vmicro=2.):
    p = Parameters(teff=teff, logg=logg, metallicity=feh, microturbulence=vmicro)
    return atlas_worker.get_model(p)

def get_synthe_model(teff, logg, feh, wmin, wmax, resolution, vmicro=2.):
    model = get_atlas_model(teff, logg, feh)
    p_synthe = Parameters(wave_min=wmin, wave_max=wmax, resolution=resolution, metallicity=feh)
    spectrum = synthe_worker.get_spectrum(model, parameters=p_synthe, quiet=False)
    return model, spectrum

def compute_spectra(wmin, wmax):
    # necessary?
    #current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    #parent_dir = os.path.dirname(current_dir)
    #sys.path.insert(0, parent_dir)

    # load atlas models and parameters
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    with open(path+"/models.pkl", "rb") as f:
        models = pickle.load(f)
    model_params = pd.read_csv(path+"/model_params.csv")

    parameters = []
    for i in range(len(model_params)):
        _teff, _logg, _feh, _res = list(model_params[['teff', 'logg', 'feh', 'resolution']].iloc[i])
        p = Parameters(wave_min=wmin, wave_max=wmax, resolution=_res, metallicity=_feh)
        parameters.append(p)

    workers = [Synthe() for _ in range(len(parameters))]

    # Run in parallel
    no_of_processes = 30
    pool = mp.Pool(processes=no_of_processes)
    results = [pool.apply_async(worker.get_spectrum, args=(m, parameter, True))\
                                for m, worker, parameter in zip(models, workers, parameters)]
    results = [r.get() for r in results]
    pool.close()
    pool.join()

    return results

def create_grid(wmin, wmax, overwrite=False):
    #path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    specfile = "synthe_%d-%d.pkl"%(wmin, wmax)

    if os.path.exists(specfile) and (not overwrite):
        with open(specfile, "rb") as f:
            spectra = pickle.load(f)
            print ("%s loaded."%specfile)
    else:
        print ("computing spectra for %d-%dAA..."%(wmin, wmax))
        spectra = compute_spectra(wmin, wmax)
        with open(specfile, "wb") as f:
            pickle.dump(spectra, f)
        print ("spectra saved to %s."%specfile)

    print ("creating spectrum grid for %s..."%specfile)
    df_all = pd.DataFrame(data={})
    for i in range(len(spectra)):
        di = pd.DataFrame(data={})
        params = spectra[i].parameters
        wavs = spectra[i].wave
        nflux = spectra[i].normed_flux
        di['wav'] = wavs
        di['flux'] = nflux
        di['teff'] = params.teff
        di['logg'] = params.logg
        di['feh'] = params.metallicity
        di['vmic'] = params.microturbulence
        df_all = df_all.append(di)
    df_all = df_all.reset_index(drop=True)

    d = df_all.sort_values(["teff", "logg", "feh", "wav"]).reset_index(drop=True)

    tgrid = np.sort(list(set(d.teff)))
    ggrid = np.sort(list(set(d.logg)))
    fgrid = np.sort(list(set(d.feh)))
    wavgrid = np.sort(list(set(d.wav)))
    print ("teff grid:", tgrid, len(tgrid))
    print ("logg grid:", ggrid, len(ggrid))
    print ("feh grid:", fgrid, len(fgrid))
    print ("wavelength grid:", wavgrid, len(wavgrid))

    keys = ['flux']

    # linear wavelength grid; should be logarithmic?
    wavarr = np.linspace(wavgrid[0], wavgrid[-1], len(wavgrid))

    #
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
    print ("grid shape:", np.shape(pgrids2d))

    outname = specfile.split(".")[0] + "_normed.npz"

    np.savez(outname, tgrid=tgrid, ggrid=ggrid, fgrid=fgrid, wavgrid=wavarr, flux=pgrids2d[0])
    print ("spectrum grid saved to %s."%outname)

if __name__ == "__main__":
    wmin, wmax = int(sys.argv[1]), int(sys.argv[2])
    create_grid(wmin, wmax)
