#%%
#========================== imports ==========================
#env: RWSOS_environment.yml
#title: Plotting hourly model runs

import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
# from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(r'c:\git\RWSOS-scripts')
# from hydro_plotting import hydro_signatures
from file_methods.postprocess import find_model_dirs, find_toml_files, find_outputs, create_combined_hourly_dataset_FRBENL
# from metrics.peak_metrics import peak_timing_errors
from metrics.run_peak_metrics import store_peak_info
from hydro_plotting.peak_timing import plot_peaks_ts, peak_timing_for_runs


#%%
# ======================= Set up the working directory =======================
working_folder = r"p:\11209265-grade2023\wflow\wflow_meuse_julia\PreCalTest_KsatHorFrac"
os.chdir(working_folder)

# ======================= Define the runs and load model runs =======================
snippets = ['sbm']
model_dirs = [working_folder]

toml_files = find_toml_files(model_dirs)  #will be useful when we can use the Wflowmodel to extract geoms and results

output_files = find_outputs(model_dirs, filename='ouput_*.csv')

print(f'toml_files: {toml_files}', f'\nmodel_dirs: {model_dirs}', f'\noutput_files: {output_files}')

run_keys = ['Ksat_BRT', 'Base', 'Ksat_RF']

# ======================= Create the FR-BE-NL combined with HBV. dataset =======================  
ds, df_gaugetoplot = create_combined_hourly_dataset_FRBENL(working_folder, 
                                                           run_keys, 
                                                           model_dirs, 
                                                           output='ouput_*.csv',
                                                           overwrite=False)

print(f'Loaded dataset with dimensions: {ds.dims}')

#%%
#======================== Create Plotting Constants =======================
#TODO: automate the color list, make permanent for each working folder? 

color_list = ['#377eb8', 
              '#ff7f00', 
              '#4daf4a', 
              '#f781bf', 
              '#a65628', 
              '#984ea3', 
              '#999999', 
              '#e41a1c', 
              '#dede00', 
              '#ff7f00', 
              '#a65628', 
              '#f781bf']

run_keys = ds.runs.values

color_dict = {f'{key}': color_list[i] for i, key in enumerate(run_keys)}

start = datetime.strptime('2005-08-01', '%Y-%m-%d')

end = datetime.strptime('2018-02-21', '%Y-%m-%d')

peak_dict = store_peak_info(ds.sel(time=slice(start,end)), 'wflow_id', 72)
Folder_plots = os.path.join(working_folder,'_figures')  # folder to save plots
#What if we set the peak finding function to the threshold and then anything that is window width is a miss


#%%
# # ======================= Plot Peak Timing Hydrograph =======================
# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)


print(f'len ds time: {len(ds.time)}')

plot_peaks_ts(ds, 
              df_gaugetoplot,
              start, end,
              Folder_plots,
              color_dict,
              peak_dict=peak_dict,
              savefig=True)


#%% 
# ======================= peak timing errors =======================
# set figure fonts
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

peak_timing_for_runs(ds, 
                     df_gaugetoplot, 
                     Folder_plots, 
                     peak_dict=peak_dict,
                     plotfig=True, 
                     savefig=True)
# %%

fig, axs = plt.subplots(5, 1, figsize=(10, 24), sharex=True)  # Create 4 subplots

# Plot combined histogram in the top subplot
for var in run_keys:
    if var != 'Obs.':
        hist, bin_edges = np.histogram(peak_dict[16][var]['timing_errors'])
        print(hist)
        axs[0].hist(bin_edges[:-1], bin_edges, weights=hist, edgecolor='black', linewidth=1.2, label=var, alpha=0.5, color=color_dict[var])

axs[0].set_title('Combined Histogram of Relative Timing Data', fontsize=16)
axs[0].set_xlabel('Lead <-- Value --> Lag', fontsize=14)
axs[0].set_ylabel('Frequency', fontsize=14)
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend()

# Plot separate histograms in the subsequent subplots
for i, var in enumerate([v for v in run_keys if v != 'Obs.'], start=1):
    # if i == 4:
    #     break
    hist, bin_edges = np.histogram(peak_dict[16][var]['timing_errors'])
    axs[i].hist(bin_edges[:-1], bin_edges, weights=hist, edgecolor='black', linewidth=1.2, label=var, alpha=0.5, color=color_dict[var])
    axs[i].set_title(f'Histogram of {var}', fontsize=16)
    axs[i].set_xlabel('Lead <-- Value --> Lag', fontsize=14)
    axs[i].set_ylabel('Frequency', fontsize=14)
    axs[i].grid(True, linestyle='--', alpha=0.6)
    axs[i].set_ylim([0, 65])
    

plt.tight_layout()
plt.show()