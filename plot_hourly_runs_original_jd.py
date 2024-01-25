#%%
# TO-DO: add in hourly obs data for Borgharen (wflow_id=16)

#%%
#env: hydromt
import hydromt
from hydromt_wflow import WflowModel
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
import plotly.graph_objects as go


import sys
import os

# self defined functions
from func_plot_signature import plot_signatures, plot_hydro
from func_io import read_filename_txt, read_lakefile
from peak_metrics import peak_timing_errors


# ======================= Functions =======================
# ------------ Load data --------------
def create_file_path(base_dir, scale):
    """
    Function to create file path
    base_dir: directory that contains the model run results
    scale: scaling factors (str), eg '0.7', '0.8', '0.9', '1.0', '1.1', '1.2'
    return: model run results (.csv) path
    """
    # Function to create file path
    return os.path.join(base_dir, f'run_scale_river_n-{scale}', f'run_river_n_{scale.replace(".", "")}', f'output_scalar_{scale.replace(".", "")}.csv')

# ------------ Calculate NSE and NSE log --------------

def calculate_nse_and_log_nse(observed, modelled):
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE) and Log-Nash-Sutcliffe Efficiency (NSE_log) 
    between observed and modelled data.
    
    Parameters:
    observed (array-like): Array of observed data.
    modelled (array-like): Array of modelled data.
    
    Returns:
    nse (float): Nash-Sutcliffe Efficiency.
    nse_log (float): Log-Nash-Sutcliffe Efficiency.
    """
    observed = np.array(observed)
    modelled = np.array(modelled)

    obs_mean = np.mean(observed)

    numerator = np.sum((observed - modelled) ** 2)

    denominator = np.sum((observed - obs_mean) ** 2)

    nse = 1 - (numerator / denominator)

    log_numerator = np.sum((np.log(observed + 1) - np.log(modelled + 1)) ** 2)
    log_denominator = np.sum((np.log(observed + 1) - np.log(obs_mean + 1)) ** 2)
    nse_log = 1 - (log_numerator / log_denominator)

    return nse, nse_log

#------------ Plot hydrograph --------------
def plot_ts(ds:xr.Dataset, 
            scales:list, 
            df_GaugeToPlot:pd.DataFrame, 
            start:str, 
            end:str, 
            Folder_plots:str, 
            action:str, 
            var:str,
            color_dict:dict,
            font:dict,
            peak_time_lag:bool=False, 
            savefig:bool=False, 
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

    
def plot_peaks_ts(ds:xr.Dataset, 
            scales:list, 
            df_GaugeToPlot:pd.DataFrame, 
            start:datetime, 
            end:datetime, 
            Folder_plots:str, 
            color_dict:dict,
            font:dict,
            savefig:bool=False, 
            window:int=72
            )->None:
    
    translate = {f's{scale.replace(".", "")}': f'scale: {scale}' for scale in scales}
    translate['Obs.'] = 'Observed'
    
    # Store peak timing information in a dictionary
    peak_dict = {}

    for id in ds.wflow_id.values:
        station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id, 'station_name'].values[0]
        station_id = id

        # select a station
        ds_sub = ds.sel(wflow_id=station_id)

        # get obs data
        obs = ds_sub.sel(runs='Obs.').Q

        peak_dict[id] = {}

        for run in ds_sub.runs.values:
            if run != 'Obs.':
                # print('first loop', run)
                sim = ds_sub.sel(runs=run).Q

                peaks_obs, timing_errors = peak_timing_errors(obs, sim, window=window)

                peaks_sim = (peaks_obs + timing_errors).astype(int)

                # Expand the inner dictionary with the inner loop
                peak_dict[id][run] = (peaks_sim, timing_errors)

                # print(f'id: {id}, run: {run}, mean error: {np.mean(peak_dict[id][run][1])}')
                
        peak_dict[id]['Obs.'] = (peaks_obs, timing_errors)
        
        # print('peak_dict', peak_dict)

        
        try:
            fig = go.Figure()
            
            for run in ds.runs.values:
                if str(run) in ['s07', 's08','s09', 's10', 's11', 's12', 'Obs.']:
                    
                    # print('second loop', id, run)
                    
                    subset = ds.sel(time=slice(start, end), runs=run, wflow_id=id).dropna(dim='time')  
                    
                    obs = ds.sel(time=slice(start, end), runs='Obs.', wflow_id=id)
                    sim = ds.sel(time=slice(start, end), runs=run, wflow_id=id)
                    
                    mask = ~obs.Q.isnull() & ~sim.Q.isnull()
                    
                    obs_filtered = obs.Q[mask]
                    sim_filtered = sim.Q[mask]
                    
                    if len(obs_filtered) == len(sim_filtered):
                        nse, nse_log = calculate_nse_and_log_nse(obs_filtered, sim_filtered)
                    else:
                        nse, nse_log = np.nan, np.nan
                        print('nse, nse_log = np.nan, np.nan')    
                        print(f'len(obs_filtered) {len(obs_filtered)} != len(sim_filtered) {len(sim_filtered)}')
                    
                    if run == 'Obs.':
                        label = f'{translate[run]}'
                        # fig add dots on the obs peaks using peak_dict[id][run][0]
                        obs_peaks = peak_dict[id][run][0]
                        fig.add_trace(go.Scatter(
                            x=subset.time.values,
                            y=subset.Q.values,
                            mode='lines',
                            name=label,
                            line=dict(color=color_dict[str(run)])
                        ))
                        fig.add_trace(go.Scatter(
                            x=subset.time[obs_peaks].values,
                            y=subset.Q[obs_peaks].values,
                            mode='markers',
                            name='Obs. Peaks',
                            marker=dict(color='red', size=8)
                        ))
                        # print('obs plotted')
                              
                    else:
                        label = f'{run}: {np.mean(peak_dict[id][run][1]):.2f} +/- {np.std(peak_dict[id][run][1]):.2f}h'+'\n$NSE$:'+f'{nse:.3f}'+' $NSE_{log}$:'+f'{nse_log:.3f}'
                        fig.add_trace(go.Scatter(
                            x=subset.time.values,
                            y=subset.Q.values,
                            mode='lines',
                            name=label,
                            line=dict(color=color_dict[str(run)])
                        ))
                        sim_peaks = peak_dict[id][run][0]
                        fig.add_trace(go.Scatter(
                            x=subset.time[sim_peaks].values,
                            y=subset.Q[sim_peaks].values,
                            mode='markers',
                            name=f'{run} Peaks',
                            marker=dict(color=color_dict[str(run)], size=8)
                        ))
                        # print('sim plotted')
                        

            fig.update_layout(
                title=f'{station_name} (id: {station_id})',
                xaxis_title='Date (hourly timestep)',
                yaxis_title='Discharge ($m^3s^{-1}$)',
                font=font
            )

            if savefig == True:
                interactive_folder = os.path.join(Folder_plots, 'interactive')
                os.makedirs(interactive_folder, exist_ok=True)
                fig.write_html(os.path.join(interactive_folder, f'PeakSelection_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}.html'))
                fig.show()
            else:
                fig.show()

        except Exception as e:
            print('\nfail peak plots, station:', station_id, '\n')
            print(e)
            pass
        
    return peak_dict


# ------------ Compute peak timing errors for all runs (and plot analysis results) --------------
def peak_timing_for_runs(ds, df_GaugeToPlot, folder_plots, action, var, plotfig=False, savefig=False):
    for id in ds.wflow_id.values:
        station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id, 'station_name'].values[0]
        station_id = id
        
        try:
            # select a station
            ds_sub = ds.sel(wflow_id=station_id)

            # get obs data
            obs = ds_sub.sel(runs='Obs.').Q

            # compute peak timing errors
            peak_dict = {}
            for run in ds_sub.runs.values:
                if run != 'Obs.':
                    sim = ds_sub.sel(runs=run).Q
                    
                    peaks, timing_errors = peak_timing_errors(obs, sim, window=72)
                    
                    mean_peak_timing = np.mean(np.abs(timing_errors)) if len(timing_errors) > 0 else np.nan
                    
                    obs_Q = obs[peaks].values
                    sim_Q = sim[peaks].values
                    peak_mape = np.sum(np.abs((sim_Q - obs_Q) / obs_Q)) / peaks.size * 100
                    
                    peak_dict[run] = {'peaks': peaks, 
                                    'timing_errors': timing_errors, 
                                    'mean_peak_timing': mean_peak_timing,
                                    'peak_mape': peak_mape}

            # print out peak timing in Obs.
            print(f'Peak timing for {station_name} (id: {station_id})')
            print(list(obs[peak_dict['s07']['peaks']].time.values.astype(str)))
            
            # plot figures
            if plotfig == True:
                fig = plt.figure(figsize=(15, 10))  # Wider figure to accommodate side-by-side bar plots

                # Scatter plot of Qobs vs timing error
                ax1 = plt.subplot2grid((2, 2), (0, 0))  # Scatter plot spans the first row
                markers = ['o', 's', '^', 'D', 'x', '*']
                for (run, data), marker in zip(peak_dict.items(), markers):
                    peaks = data['peaks']
                    timing_errors = data['timing_errors']
                    qobs = obs[peaks]
                    ax1.scatter(obs[peaks], timing_errors, marker=marker, label=run)
                ax1.legend()
                ax1.set_xlabel('Qobs (m\u00b3 s\u207b\u00b9)')
                ax1.set_ylabel('Timing Error (h)')
                ax1.set_title(f'Meuse at {station_name} (id: {station_id}) - Scatter plot of Qobs vs timing error')

                # Mean peak timing bar plot
                ax2 = plt.subplot2grid((2, 2), (1, 0))  # First plot in second row
                keys = list(peak_dict.keys())
                mean_peak_timings = [peak_dict[key]['mean_peak_timing'] for key in keys]
                colors = ['skyblue' if key != 's10' else 'grey' for key in keys]
                bars = ax2.bar(keys, mean_peak_timings, color=colors)
                for bar in bars:
                    yval = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, 
                            yval + 1,  # Add a small offset from the top of the bar
                            round(yval, 1), 
                            ha='center', 
                            va='bottom')
                # ax2.set_ylim(0, max(mean_peak_timings) + 5)
                ax2.set_ylim(0, 35)
                ax2.set_xlabel('Run')
                ax2.set_ylabel('Mean Peak Timing (h)')
                ax2.set_title('Mean peak timing')

                # Mean absolute percentage peak error bar plot
                ax3 = plt.subplot2grid((2, 2), (1, 1))  # Second plot in second row
                peak_mapes = [peak_dict[key]['peak_mape'] for key in keys]
                bars = ax3.bar(keys, peak_mapes, color=colors)
                for bar in bars:
                    yval = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, 
                            yval + 1,  # Add a small offset from the top of the bar
                            round(yval, 1), 
                            ha='center', 
                            va='bottom')
                # ax3.set_ylim(0, max(peak_mapes) + 5)
                ax3.set_ylim(0, 45)
                ax3.set_xlabel('Run')
                ax3.set_ylabel('Mean Absolute Percentage Peak Error (100%)')
                ax3.set_title('Mean absolute percentage peak error')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                plt.show()
                
                if savefig:
                    timeseries_folder = os.path.join(folder_plots, 'Event_Timing_Metrics')
                    os.makedirs(timeseries_folder, exist_ok=True)
                    filename = f'PeakTimingMetrics_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}_{action}_{var}.png'
                    plt.savefig(os.path.join(timeseries_folder, filename), dpi=300)
            if savefig == True and plotfig == False:
                print('plotfig is False, no figure saved.')
            else:
                pass
        
        except Exception as e:
            print('fail', station_name, station_id)
            print(e)
            pass
            


#%%
# ======================= Set working folder =======================
working_folder=r'p:/11209265-grade2023/wflow/wflow_meuse_julia/wflow_meuse_20240122'
sys.path.append(working_folder)
Folder_staticgeoms = r'/staticgeoms'  # folder that contain staticgeoms
Folder_plots = working_folder + r"/_plots"  # folder to save plots


# ======================= Load model results from run_scale_river_n-*  =======================
scales = ['0.7', '0.8', '0.9', '1.0', '1.1', '1.2']
# Reading files into a dictionary
model_runs = {f's{scale.replace(".", "")}': 
    pd.read_csv(create_file_path(working_folder, scale), parse_dates=True, index_col=0) for scale in scales}


# ======================= Get stations that are exist in the wflow model (not used in this script)=======================
gauges_maps = []
for file in os.listdir(working_folder+Folder_staticgeoms):
    if 'gauges' in file:
        gauges_maps.append(os.path.splitext(file)[0])

# get model station wflow_id and save them in mod_stations as numpy array
mod_name = "wflow_sbm_run_hourly_scale_river_n"  #toml file name
mod = WflowModel(working_folder, config_fn=os.path.basename(mod_name), mode="r")

mod_stations = {}
for gauge_map in gauges_maps:
    if 'wflow_id' in mod.staticgeoms[gauge_map].columns:
        mod_stations[f'{gauge_map}'] = mod.staticgeoms[gauge_map]["wflow_id"].values
        print(f'{gauge_map} : added')
    elif gauge_map == "gauges":
        mod_stations[f'{gauge_map}'] = np.array([2015], dtype=np.int64)  # Discharge outlet: Rhine 709, Meuse 2015
        print(f'{gauge_map} : added manually as 2015')
    else:
        print(f'{gauge_map} : no wflow_id')


# ======================= Load stations/gauges to plot =======================
# load csv that contains stations/gauges info that we want to plot
fn_GaugeToPlot = r'/wflow_id_to_plot.csv'
df_GaugeToPlot = pd.read_csv(working_folder+fn_GaugeToPlot)


# ======================= Load observation/measurement data =======================
# load observation/measurement data from France, Belgium and Netherlands in a dictionary
fn_France = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\FR-Hydro-hourly-2005_2022.nc'
fn_Belgium = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\qobs_hourly_belgian_catch.nc'
fn_Netherlands = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\qobs_xr.nc'
obs_dict = {}
for country in ['France', 'Belgium', 'Netherlands']:
    obs_dict[f'{country}'] = xr.open_dataset(locals()['fn_'+country])


# ======================= create xarray dataset to store modeled and obs data =======================
# the dataset has three coords: time, wflow_id (to plot), runs (all model run names and obs)
# the dataset has one variable; Q in m3/s

# get time: the overlapped time range from obs and modeled results
rng = pd.date_range(
    max(*[model_runs[key].index[0] for key in model_runs.keys()],
        *[obs_dict[key].time[0].values for key in obs_dict.keys()]),
    min(*[model_runs[key].index[-1] for key in model_runs.keys()],
        *[obs_dict[key].time[-1].values for key in obs_dict.keys()]),
    freq="H"
)

# get the wflow_id to plot
wflow_id_to_plot = [*df_GaugeToPlot.wflow_id.values]

# get the runs name
runs =['Obs.', *model_runs.keys()]

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
ds = ds * np.nan

# fill in obs data
# TO-DO: add in hourly obs data for Borgharen (wflow_id=16)
for id in wflow_id_to_plot:
    country = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id,'country'].values[0]
    if country=='France':
        ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Q'].loc[dict(wflow_id=id, time=rng)].values
    elif country=='Belgium':
        ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Qobs_m3s'].loc[dict(catchments=id, time=rng)].values
    #else: # country==Netherlands
        # ! Problem: no hourly data at Borgharen?
        #ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Qobs'].loc[dict(catchments=id, time=rng)].values
    

# fill in modeled results
for key, item in model_runs.items():
    for id in wflow_id_to_plot:
        col_name = f'Qall_{id}'  # column name of this id in model results (from Qall)
        ds['Q'].loc[dict(runs=key, wflow_id=id)] = item.loc[rng, col_name]


# # save dataset
fn_ds = r'/_output/ds_obs_model_combined.nc'
ds.to_netcdf(working_folder+fn_ds)
print(f'saved combined observations and model runs to\n{fn_ds}')


#%%

color_dict = {
        's07': '#377eb8',  # Blue
        's08': '#ff7f00',  # Orange
        's09': '#4daf4a',  # Green
        's11': '#f781bf',  # Pink
        's12': '#a65628',  # Brown
        's10': '#984ea3',  # Purple
        'Obs.': '#999999',  # Grey
    }

font = {'family': 'serif', 'size': 16}

# # ======================= Plot hydrograph =======================
# whole TS
# plot_ts(ds, scales, df_GaugeToPlot, start, end, Folder_plots, peak_time_lag=False, savefig=False)
start = datetime.strptime('2015-01-01', '%Y-%m-%d')
end = datetime.strptime('2018-02-21', '%Y-%m-%d')

# plot and store peak timing information in a dictionary ''peak timing info'
# dict is indexed by (key: station id), then by (key: run name), then a tuple of (0: obs indices, 1:sim indices and 2:timing errors)

peak_timing_info = plot_peaks_ts(ds, 
                                 scales,
                                 df_GaugeToPlot,
                                 start, end,
                                 Folder_plots,
                                 color_dict,
                                 font,
                                 savefig=True,
                                 window=30)

plot_ts(ds, 
        scales, 
        df_GaugeToPlot, 
        start, end, 
        Folder_plots, 
        action='scaling',
        var='riverN',
        peak_time_lag=False, 
        savefig=True,
        color_dict=color_dict, 
        font=font) 

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
                     df_GaugeToPlot, 
                     Folder_plots, 
                     action='scaling',
                     var='riverN',
                     plotfig=True, 
                     savefig=True)
# %%
