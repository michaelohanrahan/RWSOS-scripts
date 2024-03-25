
#%%
#========================== imports ==========================
#env: RWSOS_environment.yml
#title: Plotting hourly model runs

from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np

sys.path.append('..')

from file_methods.postprocess import find_toml_files, create_combined_hourly_dataset_FRBENL
from metrics.run_peak_metrics import store_peak_info
from hydro_plotting.peak_timing import plot_peaks_ts, peak_timing_for_runs
from hydro_plotting.hydro_signatures import plot_signatures
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
                                                           overwrite=False)

print(f'Loaded dataset with dimensions: {ds.dims}')
run_keys = ['HBV', *run_keys]

#%%
#=========================Summarize Run Dates=========================
ignore_runs = ['level1', 'level2', 'level3']

def summarize_run_dates(ignore_runs):
    cols = ['wflow_id', 'model', 'location', 'date_min', 'date_max']
    df = pd.DataFrame(columns=cols)
    
    for id in df_gaugetoplot.loc[:, 'wflow_id']:
        for run in ds.runs.values:
            if run in ignore_runs:
                continue
            
            # Filter the data for the current id and run
            data = ds.sel(wflow_id=id, runs=run)
            loc = df_gaugetoplot.loc[df_gaugetoplot['wflow_id'] == id, 'location'].values[0],
            ic(loc)
            # Drop rows where 'time' is NaN
            data = data.Q.dropna(dim='time')
            if data.isnull().all():
                # Append the min and max time to df
                df = df._append({
                    'wflow_id': id,
                    'model':run.lower(),
                    'location': str(loc).lower(),
                    'date_min': np.nan,
                    'date_max': np.nan
                }, ignore_index=True)
                continue
            
            # Get the first and last non-NaN date
            min_date = data['time'].min().values
            max_date = data['time'].max().values
            
            # Append the min and max time to df
            df = df._append({
                'wflow_id': id,
                'model':run.lower(),
                'location':str(loc).lower(),
                'date_min': min_date,
                'date_max': max_date
            }, ignore_index=True)
    
    return df

summary = summarize_run_dates(ignore_runs)

summary.to_excel(os.path.join(working_folder, 'run_dates.xlsx'))

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

start = datetime.strptime('2005-01-01', '%Y-%m-%d')
end = datetime.strptime('2018-02-21', '%Y-%m-%d')


Folder_plots = os.path.join(working_folder,'_figures')  # folder to save plots
print('Folder for plots: ', Folder_plots)
#%%
print('Calculating peak timing errors...')
peak_dict = store_peak_info(ds.sel(time=slice(start,end)), 'wflow_id', 72)
print('Peak timing errors calculated.')


#%% 
# # ======================= Plot Peak Timing Hydrograph ===============
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), 
# then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)


print('Plotting peak timing hydrographs...')
print('runs: ', run_keys)

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
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    
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


#//////////////////////////////////////////////////////////////////////
# %%
# ======================= Plot Hydrological Signatures ===============
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
for station_id, station_name in zip(df_gaugetoplot['wflow_id'], df_gaugetoplot['location']):
    try:
        plot_signatures(ds.sel(wflow_id=station_id), 
                        colors=color_dict, 
                        station_id=station_id, 
                        freq='H',
                        station_name=station_name, 
                        Folder_out=Folder_plots, save=True)
    except Exception as e:
        print(f'Error plotting station {station_id} - {station_name}')
        print(e)
#//////////////////////////////////////////////////////////////////////
# %%

import matplotlib.pyplot as plt

def plot_histogram(run1, run2, wflow_id):
    fig,ax = plt.subplots(figsize=(10,10))
    name = df_gaugetoplot.loc[df_gaugetoplot['wflow_id'] == wflow_id, 'location'].values[0]
    
    # Data
    data1 = peak_dict[wflow_id][run1]['timing_errors']
    data2 = peak_dict[wflow_id][run2]['timing_errors']
    
    # Add a vertical line at the mean.
    mean1 = np.nanmean(data1)
    stdev1= np.std(data1)
    
    mean2 = np.nanmean(data2)
    stdev2 = np.std(data2)
    
    # Create histogram
    plt.hist(data2, bins='auto', histtype='barstacked', rwidth=0.9, color='purple', alpha=0.7, edgecolor='black', label=f'{run2} n={len(data2)}\nmean={mean2:.2f}+/-{stdev2:.2f}')
    plt.hist(data1, bins='auto', histtype='barstacked', rwidth=0.9, color='green', alpha=0.7, edgecolor='black', label=f'{run1} n={len(data1)}\nmean={mean1:.2f}+/-{stdev1:.2f}')
    
    plt.axvline(mean1, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(mean2, color='cyan', linestyle='dashed', linewidth=2)
    plt.axvline(0, color='black', linestyle='dotted', alpha=0.4)
    
    # Add labels and title
    plt.title(f'Comparative Histograms with Timing Errors at {name}')
    plt.xlabel('$(leading)$ <--- Timing Errors in Hours ---> $(lagging)$')
    plt.ylabel('Frequency')

    # Add a grid
    plt.grid(axis='y', alpha=0.75)
    plt.legend()

    # Show plot
    plt.show()

plot_histogram('level3', 'fl1d_lakes', 16)