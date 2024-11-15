
__all__ = ["air_to_vac", "compute_grid_coelho", "compute_grid_ispec", "compute_grid_bosz"]

import pandas as pd
import numpy as np
import os
from astropy.io import fits
from scipy.interpolate import interp1d
from pathlib import Path
from itertools import product


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
        filepath = Path(data_dir) / Path(filename)
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

    outname = Path(output_dir) / Path("%d-%d_normed.npz"%(wmin, wmax))
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir()

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

    outname = Path(output_dir) / Path("%d-%d_normed.npz"%(wmin, wmax))
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir()

    np.savez(outname, tgrid=tgrid, ggrid=ggrid, fgrid=fgrid, agrid=agrid, wavgrid=wavarr, flux=pgrids2d[0].astype(np.float32))
    print ("spectrum grid saved to %s."%outname)
    print ()

    return outname


mfmt = lambda m: "+%.2f"%(np.abs(m)) if m>=0 else "-%.2f"%(np.abs(m))


def read_single_spectrum(filename, wminaa, wmaxaa, air_or_vac="vac"):
    _d = pd.read_csv(filename, names=["wavaa", "H", "continuum"], sep="\s+")
    if air_or_vac == "vac":
        _d['wavaa'] = air_to_vac(np.array(_d['wavaa']))
    wavidx = (wminaa < _d.wavaa) & (_d.wavaa < wmaxaa)
    d = _d[wavidx].reset_index(drop=True)
    #d = _d.query("@wminaa < wavaa < @wmaxaa").reset_index(drop=True)
    return d


def compute_grid_bosz(wminaa, wmaxaa, data_dir, output_dir, print_info=False, wavgrid_length=4000, air_or_vac="vac",
                   teff=np.arange(5000, 7250, 250.),
                   logg=np.arange(3.5, 5.5, 0.5),
                   mh=np.arange(-0.75, 0.75, 0.25),
                   alpha=np.arange(-0.25, 0.5, 0.5),
                   carbon=np.arange(-0.25, 0.5, 0.5),
                   vmic=np.array([0., 2., 4.])):
    """bosz grid in vacuum wavelength; air wavelengths above 200 nm and vacuum wavelength below 200 nm
    
        Args:
            wminaa: minimum wavelength (AA)
            wmaxaa: maximum wavelength (AA)
            data_dir: path to grid files
            output_dir: output directory
            wavgrid_length: length of wavelength grid. 4000 corresponds to ~300000 for IRD order

        Returns:
            path to output file
        
    """
    print ("# computing BOSZ spectrum grid for %dAA-%dAA"%(wminaa, wmaxaa))

    wmin_all, wmax_all = [], []
    df_all = []

    # 100files / min
    for idx, (t, g, m, a, c, v) in enumerate(product(teff, logg, mh, alpha, carbon, vmic)):
        if print_info:
            print (idx, t, g, m, a, c, v)

        data_dir_m = data_dir/("m%s"%mfmt(m))

        if g>3.:
            filename = "bosz2024_mp_t%d_g+%.1f_m%s_a%s_c%s_v%d_rorig_noresam.txt.gz"%(t, g, mfmt(m), mfmt(a), mfmt(c), v)
        else:
            filename = "bosz2024_ms_t%d_g+%.1f_m%s_a%s_c%s_v%d_rorig_noresam.txt.gz"%(t, g, mfmt(m), mfmt(a), mfmt(c), v)
        try:
            di = read_single_spectrum(data_dir_m/filename, wminaa, wmaxaa, air_or_vac=air_or_vac)
        except:
            print (filename, "not found.")
            filename = "bosz2024_mp_t%d_g+%.1f_m%s_a%s_c%s_v%d_rorig_noresam.txt.gz"%(t, 4.0, mfmt(m), mfmt(a), mfmt(c), v)
            di = read_single_spectrum(data_dir_m/filename, wminaa, wmaxaa, air_or_vac=air_or_vac)

        di['wav'] = np.array(di['wavaa'])
        di['flux'] = np.array(di['H'] / di['continuum'])
        di['teff'] = t
        di['logg'] = g
        di['mh'] = m
        di['alpha'] = a
        di['carbon'] = c
        di['vmic'] = v
        df_all.append(di)
        wmin_all.append(np.min(di.wav))
        wmax_all.append(np.max(di.wav))
        if print_info:
            print ("median wavres:", np.median(di.wav[1:] / np.diff(di.wav)))

    df_all = pd.concat(df_all).reset_index(drop=True)
    d = df_all.sort_values(["teff", "logg", "mh", "alpha", "carbon", "vmic", "wav"]).reset_index(drop=True)

    wavarr = np.linspace(np.max(wmin_all), np.min(wmax_all), wavgrid_length)
    print ("# wavelength resolution in the output grid: %d"%(wavarr[0]/np.diff(wavarr)[0]))

    pgrid2d = np.zeros((len(teff), len(logg), len(mh), len(alpha), len(carbon), len(vmic), len(wavarr)))
    for i,t in enumerate(teff):
        for j,g in enumerate(logg):
            for k,f in enumerate(mh):
                for l,a in enumerate(alpha):
                    for m,c in enumerate(carbon):
                        for n,v in enumerate(vmic):
                            _d = d[(d.teff==t)&(d.logg==g)&(d.mh==f)&(d.alpha==a)&(d.carbon==c)&(d.vmic==v)]
                            pgrid2d[i][j][k][l][m][n] = interp1d(_d.wav, _d["flux"])(wavarr)
    print ("# grid shape:", np.shape(pgrid2d))

    outname = output_dir / Path("%d-%d_normed.npz"%(wminaa, wmaxaa))
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir()

    np.savez(outname, tgrid=teff, ggrid=logg, mgrid=mh, agrid=alpha, cgrid=carbon, vgrid=vmic, wavgrid=wavarr, flux=pgrid2d.astype(np.float32))
    print ("# spectrum grid saved to %s."%outname)
    print ()

    return outname