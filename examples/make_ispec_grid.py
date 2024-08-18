""" a script to compute a spectrum grid from iSpec modelgrid
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxspec.modelgrid import *

#%% input and output directories
#wavgrid_length = 5000
wavgrid_length = 3500
#wavgrid_length = 2500
data_dir = "/Users/k_masuda/data/ispecgrid_hband_turbospectrum/"
output_dir = "/Users/k_masuda/data/specgrid_irdh_turbospectrum%d/"%wavgrid_length
dwav = pd.read_csv("wavranges_ird_h.csv")
wavfactor = 10. # nm -> AA
air_or_vac = "vac"

#%% irdh, wide logg
wavgrid_length = 3500
data_dir = "/Users/k_masuda/data/ispecgrid_hband_turbospectrum_widelogg/"
output_dir = "/Users/k_masuda/data/specgrid_irdh_turbospectrum_widelogg%d/"%wavgrid_length
dwav = pd.read_csv("wavranges_ird_h.csv")
wavfactor = 10. # nm -> AA
air_or_vac = "vac"

#%% input and output directories: GAOES-RV
'''
data_dir = "/Users/k_masuda/data/ispecgrid_gaoes_turbospectrum/"
output_dir = "/Users/k_masuda/data/specgrid_gaoes_turbospectrum/"
dwav = pd.read_csv("wavranges_gaoes-rv.csv")
wavfactor = 1. 
air_or_vac = "air"
'''

#%%
wmargin = 5 # margin in AA
for i in range(len(dwav)):
    wmin, wmax = dwav.wavmin[i]*wavfactor - wmargin, dwav.wavmax[i]*wavfactor + wmargin
    print (wmin, wmax)
    #compute_grid_ispec(wmin, wmax, data_dir, output_dir, air_or_vac=air_or_vac)
    compute_grid_ispec(wmin, wmax, data_dir, output_dir, air_or_vac=air_or_vac, wavgrid_length=wavgrid_length)
