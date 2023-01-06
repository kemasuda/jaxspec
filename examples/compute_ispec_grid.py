""" a script to compute a spectrum grid from iSpec modelgrid
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxspec.modelgrid import *

#%% input and output directories
data_dir = "/Users/k_masuda/data/ipsecmodelgrid_irdh_turbospectrum/"
output_dir = "/Users/k_masuda/data/specgrid_irdh_turbospectrum_test/"

#%%
dwav = pd.read_csv("wavranges_ird_h.csv")

#%%
wmargin = 5 # margin in AA
for i in range(len(dwav)*0+1):
    wmin, wmax = dwav.wavmin[i]*10 - wmargin, dwav.wavmax[i]*10 + wmargin
    print (wmin, wmax)
    compute_grid_ispec(wmin, wmax, data_dir, output_dir)
