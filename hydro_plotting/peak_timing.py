import os
import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime
# from hydromt_wflow import WflowModel
from metrics.objective_fn import calculate_nse_and_log_nse
# import traceback
import matplotlib.pyplot as plt
import datetime


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
                sim = ds.sel(time=slice(start, end), runs=run, wflow_id=station_id)
                
                # print('len obs, len sim', len(obs.Q), len(sim.Q))
                
                # handle nan values in the 
                mask = ~obs.Q.isnull() & ~sim.Q.isnull()
                
                obs_filtered = obs.Q[mask]
                sim_filtered = sim.Q[mask]
                
                # print('len(obs_filtered)', len(obs_filtered))
                # print('len(sim_filtered)', len(sim_filtered))
                
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
                    
                    
                    fig.add_trace(go.Scatter(
                        x=obs.time.values,
                        y=obs.Q.values,
                        mode='lines',
                        name=label,
                        line=dict(color=color_dict[str(run)])
                    ))
                    
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
            pass
        
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
                    peaks = data['peaks']
                    timing_errors = data['timing_errors']
                    # qobs = obs[data['Obs.'][peaks]]
                    ax1.scatter(obs[peaks], timing_errors, marker=marker, label=run)
                
                ax1.legend()
                ax1.set_xlabel('Qobs (m\u00b3 s\u207b\u00b9)')
                ax1.axhline(0, color='black', alpha=0.5, linestyle='--')
                ax1.set_ylabel('Timing Error (h)')
                ax1.set_title(f'Meuse at {station_name} (id: {station_id}) - Scatter plot of Qobs vs timing error')

                # Mean peak timing bar plot
                # ax2 = plt.subplot2grid((2, 2), (1, 0))  # First plot in second row
                # keys = list(peak_dict.keys())
                # mean_peak_timings = [peak_dict[key]['mean_peak_timing'] for key in keys]
                # colors = ['skyblue' if key != 's10' else 'grey' for key in keys]
                # bars = ax2.bar(keys, mean_peak_timings, color=colors)
                
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
                            round(yval, 1), 
                            ha='center', 
                            va='bottom')
                
                # ax3.set_ylim(0, max(peak_mapes) + 5)
                ax3.set_ylim(0, 45)
                ax3.set_xticklabels(list(peak_dict_sid.keys()), rotation=15)
                ax3.set_xlabel('Run')
                ax3.set_ylabel('MAPE (100%)')
                ax3.set_title('Mean Absolute Percentage Peak error (MAPE, Instaneous Discharge)')

                # Adjust layout to prevent overlap
                plt.tight_layout()
                plt.show()
                
                #define start and end as the first and last timestamp in the ds
                start = ds.time.values[-1].astype(datetime)
                end = ds.time.values[0].astype(datetime)
                
                if savefig:
                    timeseries_folder = os.path.join(folder_plots, 'Event_Timing_Metrics')
                    os.makedirs(timeseries_folder, exist_ok=True)
                    filename = f'PeakTimingMetrics_{station_name}_{station_id}_{start.year}{start.month}{start.day}-{end.year}{end.month}{end.day}.png'
                    plt.savefig(os.path.join(timeseries_folder, filename), dpi=300)

            if savefig == True and plotfig == False:
                print('plotfig is False, no figure saved.')

            else:
                pass
        
            
        except Exception as e:
            print(f'An error occurred for {station_name} (id: {station_id}): {str(e)}')
            # traceback.print_exc()

