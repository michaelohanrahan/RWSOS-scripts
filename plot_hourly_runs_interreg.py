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

#TODO: move to plotting_methods
#======================= Plot hydrograph =================================================================
def plot_ts(ds:xr.Dataset, 
            scales:list, 
            df_GaugeToPlot:pd.DataFrame, 
            start:str, 
            end:str, 
            Folder_plots:str, 
            action:str, 
            var:str,
            color_dict:dict,
            peak_time_lag:bool=False, 
            savefig:bool=False, 
            font:dict={'family': 'serif', 'size': 16}
            )->None:
    """
    ds: xarray dataset that contains modeled results for all runs and observation data
    scales: list of scaling factors, eg ['0.7', '0.8', '0.9', '1.0', '1.1', '1.2']
    df_GaugeToPlot: pandas dataframe that contains the information about the wflow_id to plot,
                    see for instance: p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\wflow_id_to_plot.csv
    start: start date to plot, eg '2015-01-01'
    end: end date to plot, eg '2015-12-31'
    Folder_plots: directory to save figures
    savefig: save the figures or not
    Peak_time_lag: calculate the peak timing lag or not, and present in legend, only useful in the case of single events
    action: The method, for the saved figure name e.g. scaling, offsetting, etc.
    var: The variable, for the saved figure name e.g. riverN, that is being scaled or offsetted.
    """
    # Define the scales for the legend
    translate = {f's{scale.replace(".", "")}': f'scale: {scale}' for scale in scales}
    translate['Obs.'] = 'Observed'
    
    for id in ds.wflow_id.values:
        station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id, 'station_name'].values[0]
        station_id = id
        
        try:
            
            fig, ax = plt.subplots(figsize=(12,6.18))
            ax.set_title(f'{station_name} (id: {station_id})', fontdict=font)
            
            obs_max_time = ds.sel(time=slice(start, end), runs='Obs.', wflow_id=id).Q.idxmax().values

            for run in ds.runs.values:
                if str(run) in ['s07', 's08','s09', 's10', 's11', 's12', 'Obs.']:
                    # Select the specific run and station from ds
                    subset = ds.sel(time=slice(start, end),
                                    runs=run, 
                                    wflow_id=id).dropna(dim='time')  
                    # print('subset', subset)
                    # Get the time index of the maximum value in this run
                    run_max_time = subset.sel(time=slice(obs_max_time - pd.Timedelta(hours=72), obs_max_time + pd.Timedelta(hours=72))).Q.idxmax().values
                    # Calculate the difference in peak timing
                    dt = run_max_time - obs_max_time
                    dt_hours = dt.astype('timedelta64[h]').item().total_seconds() / 3600
                    # Set the y-axis label
                    ax.set_ylabel('Discharge ($m^3s^{-1}$)')
                    # Set the font properties for the y-axis labels
                    ax.tick_params(axis='y', labelsize=font['size'])
                    # Set the x-axis label
                    ax.set_xlabel('Date (hourly timestep)')
                    # Set the font properties for the x-axis labels
                    ax.tick_params(axis='x', labelsize=font['size'])
                    # Set labels
                    if run == 'Obs.':
                        label = f'{translate[run]}'
                    elif peak_time_lag==True:
                        label = f'{translate[run]}, model lag = {dt_hours:.2f} hours'
                    else:
                        label = f'{translate[run]}'
                    
                    # Plot the subset for this run
                    ax.plot(subset.time, subset.Q, label=label, c=color_dict[str(run)])
                
                else:
                    continue

            # Add the legend outside of the loop
            ax.legend()
            plt.tight_layout()
            # Set the x-axis limits to the time slice
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            ax.set_xlim([start - pd.Timedelta(hours=48), end + pd.Timedelta(hours=48)])
            ax.grid()
            
            
            # save plots
            if savefig == True:
                # # Create the directory if it doesn't exist
                print('saving...')
                plots_dir = os.path.join(working_folder, '_plots')
                os.makedirs(plots_dir, exist_ok=True)
                filename = f'timeseries_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}_{action}_{var}.jpg'
                # Save the figure
                fig.savefig(os.path.join(Folder_plots, filename), dpi=300)
                # print(f'saved to {timeseries_{station_name}_{station_id}_{start.month, start.day}_{end.month,end.day}.png}')
            else:
                pass
        except Exception as e:
            print('fail timeseries:', station_id)
            print(e)
            pass

#%%
# ======================= Set up the working directory =======================
working_folder = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\compare_fl1d_interreg"
sys.path.append(working_folder)

# ======================= Define the runs and load model runs =======================
snippets = ['fl1d', 'interreg']
model_dirs = find_model_dirs(working_folder, snippets)
toml_files = find_toml_files(model_dirs)  #will be useful when we can use the Wflowmodel to extract geoms and results

#TODO: make this more general with a list of model paths
run_keys = [run.split('\\')[-1] for run in model_dirs]

# ======================= Create the FR-BE-NL combined dataset =======================  
ds, df_gaugetoplot = create_combined_hourly_dataset_FRBENL(working_folder, 
                                                           run_keys, 
                                                           model_dirs, 
                                                           output='output.csv',
                                                           toml_files= toml_files, 
                                                           overwrite=True)

print(f'Loaded dataset with dimensions: {ds.dims}')

#%%
#======================== Create Plotting Constants =======================
#TODO: automate the color list, make permanent for each working folder? 

color_list = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999'] #blue, orange, green,pink,brown,purple,grey

run_keys = ds.runs.values

color_dict = {f'{key}': color_list[i] for i, key in enumerate(run_keys)}

start = datetime.strptime('2015-01-01', '%Y-%m-%d')

end = datetime.strptime('2018-02-21', '%Y-%m-%d')

peak_dict = store_peak_info(ds.sel(time=slice(start,end)), df_gaugetoplot, 'wflow_id', 72)

#%%
# # ======================= Plot Peak Timing Hydrograph =======================
# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)
Folder_plots = os.path.join(working_folder, '_plots')  # folder to save plots

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
