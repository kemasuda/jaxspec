
__all__ = ["air_to_vac", "compute_grid_coelho", "compute_grid_ispec"]

import pandas as pd
import numpy as np
import pathlib, os
from astropy.io import fits
from scipy.interpolate import interp1d


def air_to_vac(wav_air_aa):
    """ wavelength conersion from air to vacuum following Donald Morton (2000, ApJ. Suppl., 130, 403)

        Args:
            wav_air: air wavelength in angstroms

        Returns:
            vacuum wavelength in angstroms

    """
    s = 1e4 / wav_air_aa
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s*s) + 0.0001599740894897 / (38.92568793293 - s*s)
    return wav_air_aa * n


def compute_grid_coelho(model_params, wmin, wmax, data_dir, output_dir, fixed_wavgrid_length=5000, air_or_vac="vac"):
    """ compute spectrum grid using the Coelho model

        Args:
            model_params: pandas DataFrame containing grid parameters (teff, logg, feh, alpha)
            wmin: minimum wavelength (AA)
            wmax: maximum wavelenght (AA)
            data_dir: directory where the models are stored
            output_dir: output directory
            fixed_wavgrid_length: length of wavelength grid
            air_or_vac: output the wavelength in air or vacuum

        Returns:
            name of the grid file

    """
    print ("# computing spectrum grid for %dAA-%dAA..."%(wmin, wmax))

    df_all = pd.DataFrame(data={})

    for i in range(len(model_params)):
        t, g, f, a = np.array(model_params.iloc[i])
        fsign = lambda f: "p" if f>=0 else "m"
        if t==4250 and g==5.0 and f==0.5 and a==0:
            filename = "4250_50_p05p04.ms.fits"
        elif t==4750 and f==0.5 and a==0.4:
            filename = "%d_%02d_%s%02dp%02d.ms.fits"%(t, g*10, fsign(f), np.abs(f)*10, 0)
        elif t==5250 and g==3.0 and f==0.2 and a==0.0:
            filename = "5250_30_p02p04.ms.fits"
        else:
            filename = "%d_%02d_%s%02dp%02d.ms.fits"%(t, g*10, fsign(f), np.abs(f)*10, a*10)
        filepath = pathlib.Path(data_dir) / pathlib.Path(filename)
        if not os.path.exists(filepath):
            print (filepath, "does not exist.")

        header = fits.open(filepath)[0].header
        flux = fits.open(filepath)[0].data[0]
        wavs = header['CRVAL1'] + np.arange(len(flux))*header['CD1_1']
        if air_or_vac == "vac":
            wavs = air_to_vac(wavs)

        idx = (wavs > wmin) & (wavs < wmax)

        di = pd.DataFrame(data={})
        di['wav'] = wavs[idx]
        di['flux'] = flux[idx]
        di['teff'] = t
        di['logg'] = g
        di['feh'] = f
        di['alpha'] = a
        df_all = pd.concat([df_all, di])

    df_all = df_all.reset_index(drop=True)
    d = df_all.sort_values(["teff", "logg", "feh", "alpha", "wav"]).reset_index(drop=True)

    tgrid = np.sort(list(set(d.teff)))
    ggrid = np.sort(list(set(d.logg)))
    fgrid = np.sort(list(set(d.feh)))
    agrid = np.sort(list(set(d.alpha)))
    wavgrid = np.sort(list(set(d.wav)))
    print ("teff grid:", tgrid, len(tgrid))
    print ("logg grid:", ggrid, len(ggrid))
    print ("feh grid:", fgrid, len(fgrid))
    print ("alpha grid:", agrid, len(agrid))
    print ("wavelength grid:", wavgrid, len(wavgrid))

    keys = ['flux']

    # linear wavelength grid; should be logarithmic? -> needs to be linear for grid interpolation
    if fixed_wavgrid_length is None:
        fixed_wavgrid_length = len(wavgrid) // 2
    wavarr = np.linspace(wavgrid[0], wavgrid[-1], fixed_wavgrid_length)
    print ("wavelength resolution in the output grid: %d"%(wavarr[0]/np.diff(wavarr)[0]))

    pgrids2d = []
    for key in keys:
        pgrid2d = np.zeros((len(tgrid), len(ggrid), len(fgrid), len(agrid), len(wavarr)))
        for i,t in enumerate(tgrid):
            for j,g in enumerate(ggrid):
                for k,f in enumerate(fgrid):
                    for l,a in enumerate(agrid):
                        _d = d[(d.teff==t)&(d.logg==g)&(d.feh==f)&(d.alpha==a)]
                        #farr = np.array(_d.flux)
                        #pgrid2d[i][j][k] = farr
                        pgrid2d[i][j][k][l] = interp1d(_d.wav, _d[key])(wavarr)
                        #pgrid2d.append(eeparr.reshape(len(mgrid), len(fgrid)))
        #pgrid2d = np.array(pgrid2d)
        pgrids2d.append(pgrid2d)
    print ("grid shape:", np.shape(pgrids2d))

    outname = pathlib.Path(output_dir) / pathlib.Path("%d-%d_normed.npz"%(wmin, wmax))
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir()

    np.savez(outname, tgrid=tgrid, ggrid=ggrid, fgrid=fgrid, agrid=agrid, wavgrid=wavarr, flux=pgrids2d[0].astype(np.float32))
    print ("spectrum grid saved to %s."%outname)
    print ()

    return outname


def compute_grid_ispec(wmin, wmax, data_dir, output_dir, wavgrid_length=5000, air_or_vac="vac"):
    from PyAstronomy.pyasl import read1dFitsSpec
    """ compute jaxspec grid using the synthetic grid from iSpec """
    print ("# computing spectrum grid for %dAA-%dAA"%(wmin, wmax))

    params = pd.read_csv(data_dir+"parameters.tsv", delim_whitespace=True)
    df_all = pd.DataFrame(data={})
    for i in range(len(params)):
        t, g, f, a = np.array(params.iloc[i][:4])
        filename = data_dir + params.iloc[i].filename[2:]
        wavnm, flux = read1dFitsSpec(filename)
        wavnm, flux = wavnm[1:-1], flux[1:-1]

        if air_or_vac == 'vac':
            wavaa = air_to_vac(wavnm * 10.)
        else:
            wavaa = wavnm * 10.

        idx = (wmin < wavaa) & (wavaa < wmax)

        di = pd.DataFrame(data={})
        di['wav'] = wavaa[idx].byteswap().newbyteorder()
        di['flux'] = flux[idx].byteswap().newbyteorder()
        di['teff'] = t
        di['logg'] = g
        di['feh'] = f
        di['alpha'] = a
        df_all = pd.concat([df_all, di])

    d = df_all.reset_index(drop=True)

    tgrid = np.sort(list(set(d.teff)))
    ggrid = np.sort(list(set(d.logg)))
    fgrid = np.sort(list(set(d.feh)))
    agrid = np.sort(list(set(d.alpha)))
    wavgrid = np.sort(list(set(d.wav)))
    wavarr = np.linspace(wavgrid[0], wavgrid[-1], wavgrid_length)
    print ("teff grid:", tgrid, len(tgrid))
    print ("logg grid:", ggrid, len(ggrid))
    print ("feh grid:", fgrid, len(fgrid))
    print ("alpha grid:", agrid, len(agrid))
    print ("wavelength grid:", wavgrid, len(wavgrid))
    print ("wavelength resolution in the output grid: %d"%(wavarr[0]/np.diff(wavarr)[0]))

    keys = ['flux']
    pgrids2d = []
    for key in keys:
        pgrid2d = np.zeros((len(tgrid), len(ggrid), len(fgrid), len(agrid), len(wavarr)))
        for i,t in enumerate(tgrid):
            for j,g in enumerate(ggrid):
                for k,f in enumerate(fgrid):
                    for l,a in enumerate(agrid):
                        _d = d[(d.teff==t)&(d.logg==g)&(d.feh==f)&(d.alpha==a)]
                        pgrid2d[i][j][k][l] = interp1d(_d.wav, _d[key])(wavarr)
        pgrids2d.append(pgrid2d)
    print ("grid shape:", np.shape(pgrids2d))

    outname = pathlib.Path(output_dir) / pathlib.Path("%d-%d_normed.npz"%(wmin, wmax))
    if not os.path.exists(output_dir):
        pathlib.Path(output_dir).mkdir()

    np.savez(outname, tgrid=tgrid, ggrid=ggrid, fgrid=fgrid, agrid=agrid, wavgrid=wavarr, flux=pgrids2d[0].astype(np.float32))
    print ("spectrum grid saved to %s."%outname)
    print ()

    return outname
