import os
import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime
# from hydromt_wflow import WflowModel
from metrics.objective_fn import calculate_nse_and_log_nse
import traceback
import matplotlib.pyplot as plt
from datetime import datetime




# #TODO: move to plotting_methods
# #======================= Plot hydrograph =================================================================
# def plot_ts(ds:xr.Dataset, 
#             scales:list, 
#             df_GaugeToPlot:pd.DataFrame, 
#             start:str, 
#             end:str, 
#             Folder_plots:str, 
#             action:str, 
#             var:str,
#             color_dict:dict,
#             peak_time_lag:bool=False, 
#             savefig:bool=False, 
#             font:dict={'family': 'serif', 'size': 16}
#             )->None:
#     """
#     ds: xarray dataset that contains modeled results for all runs and observation data
#     scales: list of scaling factors, eg ['0.7', '0.8', '0.9', '1.0', '1.1', '1.2']
#     df_GaugeToPlot: pandas dataframe that contains the information about the wflow_id to plot,
#                     see for instance: p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\wflow_id_to_plot.csv
#     start: start date to plot, eg '2015-01-01'
#     end: end date to plot, eg '2015-12-31'
#     Folder_plots: directory to save figures
#     savefig: save the figures or not
#     Peak_time_lag: calculate the peak timing lag or not, and present in legend, only useful in the case of single events
#     action: The method, for the saved figure name e.g. scaling, offsetting, etc.
#     var: The variable, for the saved figure name e.g. riverN, that is being scaled or offsetted.
#     """
#     # Define the scales for the legend
#     translate = {f's{scale.replace(".", "")}': f'scale: {scale}' for scale in scales}
#     translate['Obs.'] = 'Observed'
    
#     for id in ds.wflow_id.values:
#         station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id, 'station_name'].values[0]
#         station_id = id
        
#         try:
            
#             fig, ax = plt.subplots(figsize=(12,6.18))
#             ax.set_title(f'{station_name} (id: {station_id})', fontdict=font)
            
#             obs_max_time = ds.sel(time=slice(start, end), runs='Obs.', wflow_id=id).Q.idxmax().values

#             for run in ds.runs.values:
#                 if str(run) in ['s07', 's08','s09', 's10', 's11', 's12', 'Obs.']:
#                     # Select the specific run and station from ds
#                     subset = ds.sel(time=slice(start, end),
#                                     runs=run, 
#                                     wflow_id=id).dropna(dim='time')  
#                     # print('subset', subset)
#                     # Get the time index of the maximum value in this run
#                     run_max_time = subset.sel(time=slice(obs_max_time - pd.Timedelta(hours=72), obs_max_time + pd.Timedelta(hours=72))).Q.idxmax().values
#                     # Calculate the difference in peak timing
#                     dt = run_max_time - obs_max_time
#                     dt_hours = dt.astype('timedelta64[h]').item().total_seconds() / 3600
#                     # Set the y-axis label
#                     ax.set_ylabel('Discharge ($m^3s^{-1}$)')
#                     # Set the font properties for the y-axis labels
#                     ax.tick_params(axis='y', labelsize=font['size'])
#                     # Set the x-axis label
#                     ax.set_xlabel('Date (hourly timestep)')
#                     # Set the font properties for the x-axis labels
#                     ax.tick_params(axis='x', labelsize=font['size'])
#                     # Set labels
#                     if run == 'Obs.':
#                         label = f'{translate[run]}'
#                     elif peak_time_lag==True:
#                         label = f'{translate[run]}, model lag = {dt_hours:.2f} hours'
#                     else:
#                         label = f'{translate[run]}'
                    
#                     # Plot the subset for this run
#                     ax.plot(subset.time, subset.Q, label=label, c=color_dict[str(run)])
                
#                 else:
#                     continue

#             # Add the legend outside of the loop
#             ax.legend()
#             plt.tight_layout()
#             # Set the x-axis limits to the time slice
#             start = pd.to_datetime(start)
#             end = pd.to_datetime(end)
#             ax.set_xlim([start - pd.Timedelta(hours=48), end + pd.Timedelta(hours=48)])
#             ax.grid()
            
            
#             # save plots
#             if savefig == True:
#                 # # Create the directory if it doesn't exist
#                 print('saving...')
#                 plots_dir = os.path.join(working_folder, '_plots')
#                 os.makedirs(plots_dir, exist_ok=True)
#                 filename = f'timeseries_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}_{action}_{var}.jpg'
#                 # Save the figure
#                 fig.savefig(os.path.join(Folder_plots, filename), dpi=300)
#                 # print(f'saved to {timeseries_{station_name}_{station_id}_{start.month, start.day}_{end.month,end.day}.png}')
#             else:
#                 pass
#         except Exception as e:
#             print('fail timeseries:', station_id)
#             print(e)
#             pass

# ======================= Peak timing plotting =======================

def plot_peaks_ts(ds:xr.Dataset,  
            df_GaugeToPlot:pd.DataFrame, 
            start:datetime, 
            end:datetime, 
            Folder_plots:str, 
            color_dict:dict,
            run_keys:list=None,
            peak_dict:dict=None,
            savefig:bool=False,
            font:dict={'family': 'serif', 'size': 16},
            translate:dict=None,
            id_key:str='wflow_id'
            )->None:
    
    '''
    ds: xarray dataset that contains modeled results for all runs and observation data
        requires that the observations are indexed as 'Obs.'
    '''
    
    if run_keys == None:
        run_keys = ds.runs.values
    
    for station_id in ds[id_key].values:
        try:
            station_name = df_GaugeToPlot.loc[df_GaugeToPlot[id_key]==station_id, 'station_name'].values[0]

            print('Attempt html peak timing plotting for (station_name, station_id): ', (station_name, station_id))

            # try:
            fig = go.Figure()
            
            for run in ds.runs.values:
                # print('Inner loop: (id,station_name,run)', (id, station_name, run))
                # print('\nshape ds.sel(wflow_id=id).dropna(dim=time)\n', ds.sel(wflow_id=id).dropna(dim='time'))
                # print('\nshape ds.sel(runs=run, wflow_id=id).dropna(dim=time)\n', ds.sel(runs=run, wflow_id=id).dropna(dim='time'))
                # print('\nshape ds.sel(time=slice(start, end), runs=run, wflow_id=id).dropna(dim=time)\n', ds.sel(time=slice(start, end), runs=run, wflow_id=id).dropna(dim='time'))
                
                obs = ds.sel(time=slice(start, end), runs='Obs.', wflow_id=station_id)
                obs = obs.ffill('time') #observations are prone to nans
                sim = ds.sel(time=slice(start, end), runs=run, wflow_id=station_id)
                
                # print('len obs, len sim', len(obs.Q), len(sim.Q))
                
                # handle nan values in the 
                mask = ~obs.Q.isnull() & ~sim.Q.isnull()
                
                obs_filtered = obs.Q[mask]
                sim_filtered = sim.Q[mask]
                
                
                if len(obs_filtered) == len(sim_filtered):
                    nse, nse_log = calculate_nse_and_log_nse(obs_filtered, sim_filtered)
                    # print('nse, nse_log', (nse, nse_log))
                    
                else:
                    nse, nse_log = np.nan, np.nan
                    print('nse, nse_log = np.nan, np.nan')    
                    print(f'len(obs_filtered) {len(obs_filtered)} != len(sim_filtered) {len(sim_filtered)}')
                
                if run == 'Obs.':
                    label = f'{run}'
                    
                    obs_peaks = peak_dict[station_id][run]['peaks']
                    
                    print(obs_peaks)
                    
                    fig.add_trace(go.Scatter(
                        x=obs.time.values,
                        y=obs.Q.values,
                        mode='lines',
                        name=label,
                        line=dict(color=color_dict[str(run)])
                    ))
                    
                    x=obs.time.isel(time=obs_peaks).values
                    print(x)
                    fig.add_trace(go.Scatter(
                        x=obs.time.isel(time=obs_peaks).values,
                        y=obs.Q.isel(time=obs_peaks).values,
                        mode='markers',
                        name='Obs. Peaks',
                        marker=dict(color=color_dict[str(run)], size=8)
                    ))
                            
                else:
                    label = f"{run}: {np.mean(peak_dict[station_id][run]['timing_errors']):.2f} +/- {np.std(peak_dict[station_id][run]['timing_errors']):.2f}h" + "\n$NSE$:" + f"{nse:.3f}" + " $NSE_{log}$:" + f"{nse_log:.3f}"
                    
                    fig.add_trace(go.Scatter(
                        x=sim.time.values,
                        y=sim.Q.values,
                        mode='lines',
                        name=label,
                        line=dict(color=color_dict[str(run)])
                    ))
                    
                    sim_peaks = peak_dict[station_id][run]['peaks']
                    
                    fig.add_trace(go.Scatter(
                        x=sim.time.isel(time=sim_peaks).values,
                        y=sim.Q.isel(time=sim_peaks).values,
                        mode='markers',
                        name='Sim. Peaks',
                        marker=dict(color=color_dict[str(run)], size=8)
                    ))
                        
            fig.update_layout(
                title=f'{station_name} (id: {station_id})',
                xaxis_title='Date (hourly timestep)',
                yaxis_title='Discharge ($m^3s^{-1}$)',
                font=font
            )

            if savefig == True:
                interactive_folder = os.path.join(Folder_plots, 'interactive')
                os.makedirs(interactive_folder, exist_ok=True)
                filename = f'PeakTiming_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}.html'
                filepath = os.path.join(interactive_folder, filename)
                fig.write_html(filepath)
                if os.path.exists(filepath):
                    print(f'Peak timing plot for {station_name} (id: {station_id}) saved as\n{filename}')
                else:
                    print(f'Failed to save {filepath}')
            else:
                fig.show()

        except Exception as e:
            print('\nfail peak plots, station:', station_id, '\n')
            print(e)
            traceback.print_exc()
            
        
    return None


#======================= Compute peak timing errors for all runs (and plot analysis results) =============
def peak_timing_for_runs(ds:xr.Dataset,
                         df_GaugeToPlot:pd.DataFrame,
                         folder_plots:str, 
                         peak_dict:dict, 
                         plotfig:bool=False, 
                         savefig:bool=False) -> None:
    '''
    Plotting peak timing errors for all runs and stations in the dataset, displayed as distributions of timing errors and peak errors.
    ds: xarray dataset that contains modeled results for all runs and observation data
    df_GaugeToPlot: pandas dataframe that contains the information about the wflow_id to plot,
    
    '''
    
    for station_id in ds.wflow_id.values:
        # try:
        station_name = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==station_id, 'station_name'].values[0]
        
        try:
            # select a station
            ds_sub = ds.sel(wflow_id=station_id)

            # get obs data
            obs = ds_sub.sel(runs='Obs.').Q

            # print out peak timing in Obs.
            # print(f'Peak timing for {station_name} (id: {station_id})')
            # print(list(obs[peak_dict['s07']['peaks']].time.values.astype(str)))
            
            # plot figures
            if plotfig == True:
                fig = plt.figure(figsize=(15, 10))  # Wider figure to accommodate side-by-side bar plots

                # Scatter plot of Qobs vs timing error
                ax1 = plt.subplot2grid((2, 2), (0, 0))  # Scatter plot spans the first row
                
                markers = ['o', 's', '^', 'D', 'x', '*']
                peak_dict_sid = peak_dict[station_id]
                for (run, data), marker in zip(peak_dict_sid.items(), markers):
                    if run != 'Obs.':
                        peaks = data['peaks']
                        timing_errors = data['timing_errors']
                        # qobs = obs[data['Obs.'][peaks]]
                        # Assuming `data` is the variable you want to plot
                        # if np.all(np.isfinite(data)):
                            # Plot the data
                        ax1.scatter(obs[peaks], timing_errors, marker=marker, label=run)
                        # else:
                        #     print(f"Data contains non-finite values: {data}")
                
                ax1.legend()
                ax1.set_xlabel('Qobs (m\u00b3 s\u207b\u00b9)')
                ax1.axhline(0, color='black', alpha=0.5, linestyle='--')
                ax1.set_ylabel('Timing Error (h)')
                ax1.set_title(f'Meuse at {station_name} (id: {station_id}) - Scatter plot of Qobs vs timing error')
                
                ax2 = plt.subplot2grid((2, 2), (1, 0))  # First plot in second row
                keys = list(peak_dict_sid.keys())
                data = [peak_dict_sid[key]['timing_errors'] for key in keys]  # Use the raw data for the boxplot
                colors = ['skyblue' if key != 's10' else 'grey' for key in keys]

                # Create a boxplot
                bplot = ax2.boxplot(data, patch_artist=True, notch=True, vert=1, showfliers=False)  # Set showfliers=False to remove outliers

                # Calculate the means
                means = [np.mean(d) for d in data]

                # Draw a line between the means
                ax2.plot(range(1, len(means) + 1), means, color='r', linestyle='--')

                # Set colors for each box
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)

                # Display the mean value for each run
                for i, mean in enumerate(means, start=1):
                    ax2.text(i, mean, round(mean, 1), ha='center', va='bottom')

                # Set x-axis labels
                ax2.set_xticklabels(keys, rotation=15)
                
                ax2.set_xlabel('Run')
                ax2.set_ylabel('Peak Timing (h)')
                ax2.set_title('Peak Timing Per Run')

                # Mean absolute percentage peak error bar plot
                ax3 = plt.subplot2grid((2, 2), (1, 1))  # Second plot in second row
                peak_mapes = [peak_dict_sid[key]['peak_mape'] for key in keys]
                bars = ax3.bar(keys, peak_mapes, color=colors)  # add yerr parameter

                # Add the data values on top of the bars
                for bar in bars:
                    yval = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2, 
                            yval + 1,  # Add the error and a small offset from the top of the bar
                            round(yval, 2), 
                            ha='center', 
                            va='bottom')
                
                # ax3.set_ylim(0, max(peak_mapes) + 5)
                # ax3.set_ylim(0, 48)
                ax3.set_xticklabels(list(peak_dict_sid.keys()), rotation=15)
                ax3.set_xlabel('Run')
                ax3.set_ylabel('MAPE (100%)')
                ax3.set_title('Mean Absolute Percentage Peak error (MAPE, Instaneous Discharge)')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                # plt.show()
                
                #define start and end as the first and last timestamp in the ds
                start = pd.to_datetime(ds.time.min().values.astype(datetime))
                end = pd.to_datetime(ds.time.max().values.astype(datetime))
                
                
                if savefig:
                    timeseries_folder = os.path.join(folder_plots, 'Event_Timing_Metrics')
                    os.makedirs(timeseries_folder, exist_ok=True)
                    filename = f'PeakTimingMetrics_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}.png'
                    plt.savefig(os.path.join(timeseries_folder, filename), dpi=300)

            if savefig == True and plotfig == False:
                print('plotfig is False, no figure saved.')
            else:
                None
        
            
        except Exception as e:
            print(f'An error occurred for {station_name} (id: {station_id}): {str(e)}')
            traceback.print_exc()

