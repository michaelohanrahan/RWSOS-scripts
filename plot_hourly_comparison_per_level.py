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
from icecream import ic


#%%
# ======================= Set up the working directory =======================
working_folder = r"p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N"
sys.path.append(working_folder)

# ======================= Define the runs and load model runs =======================
# snippets = ['.', 'base']
model_dirs = [r"p:\11209265-grade2023\wflow\wflow_meuse_julia\compare_fl1d_interreg\fl1d_lakes",
            r"P:\11209265-grade2023\wflow\wflow_meuse_julia\compare_fl1d_interreg\interreg",
            r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\fl1d_level1\base",
            r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\fl1d_level2\base",
            r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\fl1d_level3\base",]
            # r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\fl1d_level4\base",]
            # r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\fl1d_level5\base",]

toml_files = find_toml_files(model_dirs)  #will be useful when we can use the Wflowmodel to extract geoms and results

run_keys = ['fl1d_lakes', 'interreg', 'level1', 'level2','level3']# 'level4', 'level5']

# ======================= Create the FR-BE-NL combined dataset =======================  
ds, df_gaugetoplot = create_combined_hourly_dataset_FRBENL(working_folder, 
                                                           run_keys, 
                                                           model_dirs, 
                                                           output='output.csv',
                                                           overwrite=True)

print(f'Loaded dataset with dimensions: {ds.dims}')
#%%
#====== Developing the Obs Dataset
#Function vars
output = 'output.csv'
overwrite = True

fn_ds = os.path.join(working_folder, '_output/ds_obs_model_combined.nc')
    
# ======================= Load stations/gauges to plot =======================
# load csv that contains stations/gauges info that we want to plot
fn_GaugeToPlot = 'wflow_id_add_HBV.csv'

df_GaugeToPlot = pd.read_csv(os.path.join(working_folder, fn_GaugeToPlot))

if not overwrite and os.path.exists(fn_ds):
    print(f'obs and model runs already combined in {fn_ds}')
    print('overwrite is false')
    print(f'loading {fn_ds}')
    
    ds = xr.open_dataset(fn_ds)
    
    return ds, df_GaugeToPlot

elif overwrite or not os.path.exists(fn_ds):
    # try:
                    
    # ======================= Load model runs =======================
    if overwrite:
        print('overwriting the combined dataset...')
    else:
        print('combined dataset does not exist, creating...')
    os.makedirs(os.path.join(working_folder, '_output'), exist_ok=True)

    #find the output files
    if output in ['csv', 'nc']:
        output_files = find_outputs(model_dirs, filetype=output)
    else:
        output_files = find_outputs(model_dirs, filename=output)
    
    # ====================== load the model results into memory
    print('\nloading model runs...\n')
    
    print(output_files)
    
    model_runs = {}
    
    total_len = len(run_keys)
    for n, (run, result) in enumerate(zip(run_keys, output_files), 1):
        model_runs[run] = pd.read_csv(result, parse_dates=True, index_col=0)
        print(f'from time: {model_runs[run].index[0]} to {model_runs[run].index[-1]}')
        print(f"Progress: {n}/{total_len} loaded ({run}, len: {len(model_runs[run])})")
    
    # ======================= Load observation/measurement data =======================
    # load observation/measurement data from France, Belgium and Netherlands in a dictionary
    fn_France = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\FR-Hydro-hourly-2005_2022.nc'
    fn_Belgium = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\qobs_hourly_belgian_catch.nc'
    fn_Netherlands = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\rwsinfo_hourly.nc'
    
    obs_dict = {}
    
    for country in ['France', 'Belgium', 'Netherlands']:
        obs_dict[f'{country}'] = xr.open_dataset(locals()['fn_'+country])


    # ======================= create xarray dataset to store modeled and obs data =======================
    # Convert time values from nc files to pandas Timestamp objects
    model_runs_time_values = [pd.to_datetime(model_runs[key].index) for key in model_runs.keys()]
    obs_time_values = [pd.to_datetime(obs_dict[key].time.values) for key in obs_dict.keys()]

    # Flatten the lists of DatetimeIndex objects into single lists
    model_runs_time_values_flat = [time for sublist in model_runs_time_values for time in sublist]
    obs_time_values_flat = [time for sublist in obs_time_values for time in sublist]

    # Determine the min/max time values
    model_run_min_time, model_run_max_time = min(model_runs_time_values_flat), max(model_runs_time_values_flat)
    obs_min_time, obs_max_time             = min(obs_time_values_flat), max(obs_time_values_flat)

    print(f'model_run_min_time: {model_run_min_time}\nmodel_run_max_time: {model_run_max_time}')
    print(f'obs_min_time: {obs_min_time}\nobs_max_time: {obs_max_time}')
    
    # Determine the common min/max range of time
    common_min_time = max(model_run_min_time, obs_min_time)
    common_max_time = min(model_run_max_time, obs_max_time)
    
    print(f'\ncommon_min_time: {common_min_time}\ncommon_max_time: {common_max_time}\n')

    # Generate the range of time values
    rng = pd.date_range(common_min_time, common_max_time, freq="H")

    #======================= create the combined dataset =======================
    # get the wflow_id to plot
    wflow_id_to_plot = [*df_GaugeToPlot.wflow_id.values]

    # get the runs name
    runs =['Obs.', *list(model_runs.keys())]

    # variables
    variables = ['Q']

    # create data_vars dimension
    S = np.zeros((len(rng), len(wflow_id_to_plot), len(runs)))
    v = (('time', 'wflow_id', 'runs'), S)
    h = {k:v for k in variables}

    # create empty ds that contains three coords: time, wflow_id (to plot), and runs (from obs and modeled)
    ds = xr.Dataset(
            data_vars=h,
            coords={'time': rng,
                    'wflow_id': wflow_id_to_plot,
                    'runs': runs})  # note: add obs
    
    print('\nEmpty Dataset:\n', ds)
    
    ds = ds * np.nan

    # fill in obs data
    for wflow_id in wflow_id_to_plot:
        
        country = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==wflow_id,'country'].values[0]
        
        if country=='France':
            # intersect the time ranges
            print(f'obs_dict[f\'{country}\'][\'Q\'].time.values: {obs_dict[f"{country}"]["Q"].time.values.shape}')
            print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.min: {obs_dict[f"{country}"]["Q"].time.values.min()}')
            print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.max: {obs_dict[f"{country}"]["Q"].time.values.max()}')
            
            time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
            print(f'time_intersection: {time_intersection.shape}')
            
            ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
        
        elif country=='Belgium':
            # intersect the time ranges
            time_intersection = np.intersect1d(obs_dict[f'{country}']['Qobs_m3s'].time.values, rng)
            ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Qobs_m3s'].sel({'catchments':wflow_id, 'time':time_intersection}).values
            
        else:
            time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
            ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
    
    print('\nmodel_runs:\n', model_runs)
    print('\nmodel_runs.keys():\n', model_runs.keys())
    print('\n Dataset:\n', ds)
    
    # fill in modeled results
    for run, item in model_runs.items():
        print(list(item.columns))
        
        for wflow_id in wflow_id_to_plot:
            try:
                col_name = f'Q_{wflow_id}'  # column name of this id in model results (from Qall)
                
                item_reindexed = item.reindex(rng)
                
                ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
            
            except Exception as e:
                print(f'Could not find: {run} {wflow_id} {col_name}')
                    
                try:
                    col_name = f'Q_locs_{wflow_id}'  # column name of this id in model results (from Qall)

                    item_reindexed = item.reindex(rng)
                    
                    ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
                
                except Exception as e:
                    print(f'Could not find: {run} {wflow_id} {col_name}')
                    print(e)
                
    # except Exception as e:
    #     print(col_name)
    #     print('Could not find the column name in model results')
    #     print(e)
                    
                    
    # save the combined dataset
    ds.to_netcdf(fn_ds)
    
    print(f'saved combined observations and model runs to\n{fn_ds}')

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

peak_dict = store_peak_info(ds.sel(time=slice(start,end)), 'wflow_id', 72)

#%%
# # ======================= Plot Peak Timing Hydrograph =======================
# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)
Folder_plots = os.path.join(working_folder,'_figures')  # folder to save plots

print(f'len ds time: {len(ds.time)}')

plot_peaks_ts(ds, 
              df_gaugetoplot,
              start, end,
              Folder_plots,
              color_dict,
              peak_dict=peak_dict,
              savefig=True)

#//////////////////////////////////////////////////////////////////////
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

