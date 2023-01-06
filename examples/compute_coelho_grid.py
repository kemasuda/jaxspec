""" a script to compute spectrum grid using the Coelho model
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxspec.modelgrid import *

#%% input and output directories
data_dir = "/Users/k_masuda/data/s_coelho05/"
output_dir = "/Users/k_masuda/data/specgrid_irdh_coelho/"

#%% grid parameters
teffs = np.arange(3500, 7001, 250)
loggs = np.arange(1, 5.1, 1.)
#fehs = np.array([-1., -0.5, 0., 0.2, 0.5])
fehs = np.array([-1., -0.5, 0., 0.5])
alphas = np.array([0, 0.4])

model_params = []
for _teff in teffs:
    for _logg in loggs:
        for _feh in fehs:
            for _alpha in alphas:
                model_params.append([_teff, _logg, _feh, _alpha])
model_params = np.array(model_params)
model_params = pd.DataFrame(data=model_params, columns=['teff', 'logg', 'feh', 'alpha'])

print ("teff:", teffs)
print ("logg:", loggs)
print ("feh:", fehs)
print ("alpha:", alphas)

#%%
d = pd.read_csv("wavranges_ird_h.csv")

#%%
for i in range(1):
    wmin_aa = int(d.iloc[i].wavmin*10) - 5
    wmax_aa = int(d.iloc[i].wavmax*10) + 5
    print ("#", wmin_aa, wmax_aa)
    output = compute_grid_coelho(model_params, wmin_aa, wmax_aa, data_dir=data_dir, output_dir=output_dir)
