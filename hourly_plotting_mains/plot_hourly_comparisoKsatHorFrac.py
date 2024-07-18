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
from hydro_plotting.peak_timing import plot_peaks_ts, peak_timing_for_runs, plot_peak_timing_distribution
from scipy.ndimage import gaussian_filter1d

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
'''
Working on smoothing the observations... I test some different kernels on the 1d series
I like gaussian 4, and mean 5
'''
# limit_range = 10
# for gauge in ds.wflow_id.values:
#     if gauge == 16:
#         fig, axs = plt.subplots(nrows=limit_range-2, figsize=(20,6.18*(limit_range-2)))
#         series = ds.Q.sel(runs='Obs.', wflow_id=gauge).isel(time=slice(1, 24*365))
#         series = pd.Series(index=series.time.values, data= series.values)
        
#         for i in range(2, limit_range):
#             window_size = i
#             sigma = i / 2.0
            
#             # Apply rolling mean
#             rolling_mean = series.rolling(window=window_size).mean()
            
#             # Apply Gaussian filter
#             gaussian_smooth = gaussian_filter1d(series, sigma)
#             gaussian_smooth = pd.Series(data=gaussian_smooth, index=series.index)
            
#             axs[i-2].plot(series, label='original') 
#             axs[i-2].plot(rolling_mean, label=f'rolling_mean_window={window_size}', alpha=0.6)
#             axs[i-2].plot(gaussian_smooth, label=f'gaussian_sigma={sigma}', alpha=0.6)
            
#             if i % 2 != 0:
                
#                 # Apply Median filter
#                 median_smooth = medfilt(series, window_size)
#                 median_smooth = pd.Series(data=median_smooth, index=series.index)
#                 axs[i-2].plot(median_smooth, label=f'median_window={window_size}', alpha=0.6)
                
#             axs[i-2].legend()
#             axs[i-2].set_title(df_gaugetoplot[df_gaugetoplot['wflow_id']==gauge]['location'].values[0])
            
#         plt.tight_layout()
#         plt.savefig(f'FiltersComparison_windowRange{limit_range}_Borgharen.png', dpi=600)
        
#%%
'''
Replace the obs at this location
'''
fig, axs = plt.subplots(nrows=1, figsize=(20,6.18))
axs.plot(ds.time, ds.Q.sel(runs='Obs.', wflow_id=16), label='original')
sigma = 6/2.0
series = ds.Q.loc[{'runs': 'Obs.', 'wflow_id': 16}]
gaussian_smooth = gaussian_filter1d(series, sigma)
ds1 = ds
ds1.Q.loc[{'runs': 'Obs.', 'wflow_id': 16}] = gaussian_smooth
ds1 = ds1.sel(wflow_id=[16])
axs.plot(ds1.time, ds1.Q.sel(runs='Obs.', wflow_id=16), label='smoothed')
plt.xlim(pd.Timestamp('2011-01-01'), pd.Timestamp('2011-12-31'))
plt.legend()
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
Folder_plots = os.path.join(working_folder,'_gauss_figures')  # folder to save plots
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
# ======================= Peak timing distribution =======================
'''
Peak timing distribution for each run
including the cumulative distribution of the timing errors

'''
peak_timing_distribution(run_keys, peak_dict, color_dict, Folder_plots)