#%%
import jax, numpyro
import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update('jax_platform_name', 'cpu')
config.update('jax_enable_x64', True)
numpyro.set_platform('cpu')
print (jax.local_devices())
import pandas as pd
from jaxspec.specfit import SpecFit

#%%
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'

#%%
def load_ird_spectrum(datapath, orders):
    orders = np.atleast_1d(orders)
    data = pd.read_csv(datapath)
    wav_obs = np.array(data.lam).reshape(-1,2048)[orders,:] * 10 # nm -> AA
    flux_obs = np.array(data.normed_flux).reshape(-1,2048)[orders,:]
    error_obs = np.array(data.flux_error).reshape(-1,2048)[orders,:]
    mask_obs = np.array(data.all_mask).reshape(-1,2048)[orders,:]
    for i in range(len(wav_obs)-1):
        mask_obs[i+1] += wav_obs[i+1] < np.max(wav_obs[i])
    return [wav_obs, flux_obs, error_obs, mask_obs]


#%%
datapath = "../../reach-kkll/data/IRDA00042313_H.csv"
gridpath = "/Users/k_masuda/data/specgrid_irdh_turbospectrum"
orders = [7,8,9]

#%%
wav_obs, flux_obs, error_obs, mask_obs = load_ird_spectrum(datapath, orders)

#%%
idx = ~mask_obs[0]
plt.plot(wav_obs[0][idx], flux_obs[0][idx], '.')

#%%
sf = SpecFit(gridpath, [wav_obs, flux_obs, error_obs, mask_obs], orders, vmax=50., wav_margin=3.)

#%%
sf.check_ccf()

#%%
sf.optim_iterate(output_dir="./")

#%%
sf.check_residuals()

#%%
import numpyro
from jax import random
from numpyro_model import initialize_HMC, model as hmcmodel, get_mean_models

#%%
init_strategy = initialize_HMC(sf)
kernel = numpyro.infer.NUTS(hmcmodel, target_accept_prob=0.90, init_strategy=init_strategy)
mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=500)

#%%
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, sf)
mcmc.print_summary()

#%%


#%%
plt.ylim(0.2, 1.5)
plt.plot(sf.sm.wav_obs[0], sf.sm.flux_obs[0], '.')
plt.plot(sf.sm.wav_obs[0], sf.sm.fluxmodel(sf.sm.wav_obs, popt[:-3])[0]+gp.predict(res, t=sf.sm.wav_obs[0]))



#%%
def run(gridpath, datapath, resfile, orders, vmax=50., dirtag=''):
    output_dir = datapath.split("/")[-1].split(".")[0]+dirtag+"/"
    if not os.path.exists(output_dir):
        os.system("mkdir %s"%output_dir)

    data = pd.read_csv(datapath)
    wav_obs = np.array(data.lam).reshape(-1,2048)[orders,:]*10
    flux_obs = np.array(data.normed_flux).reshape(-1,2048)[orders,:]
    error_obs = np.array(data.flux_error).reshape(-1,2048)[orders,:]
    mask_obs = np.array(data.all_mask).reshape(-1,2048)[orders,:]
    for i in range(len(wav_obs)-1):
        mask_obs[i+1] += wav_obs[i+1] < np.max(wav_obs[i])

    sf = SpecFit(gridpath, [wav_obs, flux_obs, error_obs, mask_obs], orders, vmax=vmax, wav_margin=3.)
    sf.add_wavresinfo(resfile)

    print ("# computing CCFs...")
    sf.check_ccf(output_dir=output_dir)

    print ()
    print ("# optimizing...")
    sf.optim_iterate()

    sf.check_residuals(output_dir=output_dir)

    print ()
    print ("# HMC initial parameters:")
    init_strategy = initialize_HMC(sf)
    kernel = numpyro.infer.NUTS(hmcmodel, target_accept_prob=0.90, init_strategy=init_strategy)
    mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=500)

    print ()
    print ("# running HMC...")
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, sf)
    mcmc.print_summary()

    with open(output_dir+"mcmc.pkl", "wb") as f:
        dill.dump(mcmc.get_samples(), f)

    samples = mcmc.get_samples()
    ms, mgps = get_mean_models(samples, sf)
    sf.plot_models(ms, mgps, output_dir=output_dir)

    names = sf.pnames[2:8] + sf.pnames[10+2:14]
    hyper = pd.DataFrame(data=dict(zip(names, [samples[k] for k in names])))
    fig = corner.corner(hyper, labels=names, show_titles="%.2f")
    plt.savefig(output_dir+"corner.png", dpi=200, bbox_inches="tight")
    plt.close()

#%%
gridpath = "/Users/k_masuda/data/s_coelho05/jsgrid5000/"
gridpath = "/Users/k_masuda/data/s_coelho05/jsgrid7500/"
gridpath = "/Users/k_masuda/data/specgrid_irdh_turbospectrum5000/"
gridpath = "/Users/k_masuda/data/specgrid_irdh_synthe5000/"
resfile = "../resolution/ipsummary_MMF.csv"
#resfile = "../resolution/ipsummary_REACH.csv"
orders = range(7, 19)

#%%
d = pd.read_csv("../targets/twins_merged_dr3.csv")
dmmf = d[~d.reach].reset_index(drop=True)
dr = d[d.reach].reset_index(drop=True)

#%%
datapath = "../data/IRDA00042313_H.csv"
#datapath = "../data/IRDA00042315_H.csv"
#datapath = "../data/IRDA00042317_H.csv"
#datapath = "../data/IRDA00042321_H.csv"
#datapath = "../data/IRDA000%d_H.csv"%dr.dataid[0]

#%%
run(gridpath, datapath, resfile, orders, dirtag='_synthe')
