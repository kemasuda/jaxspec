__all__ = ["define_grid_ranges", "precompute_synthetic_grid"]

# %%
import os
import sys
import numpy as np
import pathlib

# %%
def define_grid_ranges(tmin=3500., tmax=7500., tstep=250., gmin=3., gmax=5., gstep=0.5, fmin=-1, fmax=0.5, fstep=0.25, amin=0., amax=0.4, astep=0.4):
    ispec_dir = str(pathlib.Path("~/iSpec/").expanduser()) + "/"
    sys.path.insert(0, os.path.abspath(ispec_dir))
    import ispec

    teffs = np.arange(tmin, tmax+tstep, tstep)
    loggs = np.arange(gmin, gmax+gstep, gstep)
    fehs = np.arange(fmin, fmax+fstep, fstep)
    alphas = np.arange(amin, amax+astep, astep)
    grid_size = len(teffs)*len(loggs)*len(fehs)*len(alphas)

    params = []
    for t in teffs:
        for g in loggs:
            for f in fehs:
                for a in alphas:
                    params.append([t, g, f, a])
    params = np.array(params)

    ranges = np.recarray((grid_size,),  dtype=[(
        'teff', int), ('logg', float), ('MH', float), ('alpha', float), ('vmic', float)])

    for i in range(grid_size):
        t, g, m, a = params[i]
        vmic = ispec.estimate_vmic(t, g, m)
        ranges['teff'][i] = t
        ranges['logg'][i] = g
        ranges['MH'][i] = m
        ranges['alpha'][i] = a
        ranges['vmic'][i] = vmic

    return ranges

# %%
def precompute_synthetic_grid(ranges, wnm_min, wnm_max, wavgrid_length, dir_head, code, atmosphere_model='marcs',
                              number_of_processes=1):
    ispec_dir = str(pathlib.Path("~/iSpec/").expanduser()) + "/"
    sys.path.insert(0, os.path.abspath(ispec_dir))
    import ispec
    import logging
    import multiprocessing
    from multiprocessing import Pool

    precomputed_grid_dir = dir_head + "_%s/" % (code)

    ## - Read grid ranges from file
    #from astropy.io import ascii
    #ranges_filename = "input/minigrid/initial_estimate_grid_ranges.tsv"
    #ranges = ascii.read(ranges_filename, delimiter="\t")
    ## - or define them directly here (example of only 2 reference points):

    # wavelengths
    initial_wave = wnm_min
    final_wave = wnm_max
    wavelengths = np.linspace(initial_wave, final_wave, wavgrid_length)
    step_wav = np.diff(wavelengths)[0]
    res = np.median(wavelengths/step_wav)
    print ("# step wave (AA):", step_wav*10)
    print ("# resolution:", res)
    assert step_wav < 0.01
    assert res > 200000
    
    to_resolution = 400000 # Individual files will not be convolved but the grid will be (for fast comparison)
     # It can be parallelized for computers with multiple processors

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/"
    print ("# atmosphere model:", atmosphere_model)
    if atmosphere_model == 'marcs':
        model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    else:
        model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/"

    if wnm_min > 300 and wnm_max < 1100:
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    elif wnm_min > 1100 and wnm_max < 2400:
        #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
        atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    else:
        raise ValueError("Check the wavelength ranges.")
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=initial_wave, wave_top=final_wave)
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)

    if "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    else:
        # MARCS
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    ## Custom fixed abundances
    #fixed_abundances = ispec.create_free_abundances_structure(["C", "N", "O"], chemical_elements, solar_abundances)
    #fixed_abundances['Abund'] = [-3.49, -3.71, -3.54] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
    ## No fixed abundances
    fixed_abundances = None

    ispec.precompute_synthetic_grid(precomputed_grid_dir, ranges, wavelengths, to_resolution, \
                                    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, \
                                    segments=None, number_of_processes=number_of_processes, \
                                    code=code, steps=False)
