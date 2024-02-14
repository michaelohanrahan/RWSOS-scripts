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
# from hydro_plotting import hydro_signatures
from file_methods.postprocess import find_model_dirs, find_toml_files, find_outputs, create_combined_hourly_dataset_FRBENL
# from metrics.peak_metrics import peak_timing_errors
from metrics.run_peak_metrics import store_peak_info
from hydro_plotting.peak_timing import plot_peaks_ts, peak_timing_for_runs


#%%
# ======================= Set up the working directory =======================
working_folder = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_20240125_riverN"
sys.path.append(working_folder)

# ======================= Define the runs and load model runs =======================
snippets = ['run_scale']
model_dirs = find_model_dirs(working_folder, snippets)


[model_dirs.append(folder) for folder in [r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_20240122\run_scale_river_n-0.9",
                    r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_20240122\run_scale_river_n-1.0",
                    r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_20240122\run_scale_river_n-1.1",
                    r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_20240122\run_scale_river_n-1.2",
                    r"P:\11209265-grade2023\wflow\wflow_meuse_julia\compare_fl1d_interreg\interreg"]]

toml_files = find_toml_files(model_dirs)  #will be useful when we can use the Wflowmodel to extract geoms and results

run_keys = [run.split('\\')[-1].split('_')[-1] for run in model_dirs]

# ======================= Create the FR-BE-NL combined dataset =======================  
ds, df_gaugetoplot = create_combined_hourly_dataset_FRBENL(working_folder, 
                                                           run_keys, 
                                                           model_dirs, 
                                                           output='csv',
                                                           toml_files= toml_files, 
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

start = datetime.strptime('2015-01-01', '%Y-%m-%d')

end = datetime.strptime('2018-02-21', '%Y-%m-%d')

peak_dict = store_peak_info(ds.sel(time=slice(start,end)), df_gaugetoplot, 'wflow_id', 72)

#%%
# # ======================= Plot Peak Timing Hydrograph =======================
# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)
Folder_plots = os.path.join(working_folder,'..','combined_run_plot' '_plotcombined_N')  # folder to save plots

print(f'len ds time: {len(ds.time)}')

peak_timing_info = plot_peaks_ts(ds, 
                                 df_gaugetoplot,
                                 start, end,
                                 Folder_plots,
                                 color_dict,
                                 peak_dict=peak_dict,
                                 savefig=True)




# whole TS
# plot_ts(ds, scales, df_GaugeToPlot, start, end, Folder_plots, peak_time_lag=False, savefig=False)


# plot_ts(ds, 
#         run_keys, 
#         df_GaugeToPlot, 
#         start, end, 
#         Folder_plots, 
#         action='scaling',
#         var='riverN',
#         peak_time_lag=False, 
#         savefig=True,
#         color_dict=color_dict, 
#         font=font) 

# # # peak event Jan-Feb
# # plot_ts(ds, df_GaugeToPlot, '2015-01-01', '2015-02-14', Folder_plots, peak_time_lag=True, savefig=False)
# # # peak event Feb-Mar
# # plot_ts(ds, df_GaugeToPlot, '2015-02-15', '2015-03-31', Folder_plots, peak_time_lag=True, savefig=False)
# # # peak event Apr-May
# # plot_ts(ds, df_GaugeToPlot, '2015-04-01', '2015-05-14', Folder_plots, peak_time_lag=True, savefig=False)
# # # peak event Sep-Oct
# # plot_ts(ds, df_GaugeToPlot, '2015-09-01', '2015-10-14', Folder_plots, peak_time_lag=True, savefig=False)
# # # peak event Nov-Dec
# # plot_ts(ds, df_GaugeToPlot, '2015-11-15', '2015-12-31', Folder_plots, peak_time_lag=True, savefig=False)


# # ======================= Plot signature =======================
# colors = [
#     '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
#     '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
# plot_colors = colors[:len(model_runs)]
# translate = {
#         's07': 'scale: 0.7',
#         's08': 'scale: 0.8',
#         's09': 'scale: 0.9',
#         's11': 'scale: 1.1',
#         's12': 'scale: 1.2',
#         's10': 'scale: 1.0',
#         'Obs.': 'Observed',
#         }

# for id in ds.wflow_id.values:
#     station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id, 'station_name'].values[0]
#     station_id = id
    
#     dsq = ds.sel(wflow_id=id)
#     # dsq.sel(time=dsq.time[~dsq.Q.sel(runs="Obs.").isnull()].values)
        
#     try:
#         plot_signatures(
#             dsq=dsq, labels=list(model_runs.keys()), colors=plot_colors,
#             Folder_out=Folder_plots, station_name=station_name, station_id=station_id, save=False,
#             )
            
#     except Exception as e:
#         print(e)
#         pass


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
