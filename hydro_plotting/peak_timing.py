import os
import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from datetime import datetime
from hydromt_wflow import WflowModel
from metrics.objective_fn import calculate_nse_and_log_nse

# ======================= Peak timing plotting =======================

def plot_peaks_ts(ds:xr.Dataset, 
            run_keys:list, 
            df_GaugeToPlot:pd.DataFrame, 
            start:datetime, 
            end:datetime, 
            Folder_plots:str, 
            color_dict:dict,
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
    
    #building a translation dict for legend very inflexible
    #TODO: make so if translate is not none, it will use that, otherwise it will use the run_keys
    # alternative is use ds.runs.values
    translate = {f'{run.replace(".", "")}': f'scale: {run}' for run in run_keys}
    translate['Obs.'] = 'Observed'
    
    id_key = 'wflow_id' #will grow with more ids
    
    # Store peak timing information in a dictionary
    # peak_dict = {}

    for id in ds[id_key].values:
        
        station_name = df_GaugeToPlot.loc[df_GaugeToPlot[id_key]==id, 'station_name'].values[0]
        station_id = id

        print('Attempt html peak timing plotting for (station_name, station_id): ', (station_name, station_id))

        try:
            fig = go.Figure()
            
            for run in ds.runs.values:
                # print('Inner loop: (id,station_name,run)', (id, station_name, run))
                # print('\nshape ds.sel(wflow_id=id).dropna(dim=time)\n', ds.sel(wflow_id=id).dropna(dim='time'))
                # print('\nshape ds.sel(runs=run, wflow_id=id).dropna(dim=time)\n', ds.sel(runs=run, wflow_id=id).dropna(dim='time'))
                # print('\nshape ds.sel(time=slice(start, end), runs=run, wflow_id=id).dropna(dim=time)\n', ds.sel(time=slice(start, end), runs=run, wflow_id=id).dropna(dim='time'))
                
                subset = ds.sel(time=slice(start, end), runs=run, wflow_id=id).Q.dropna(dim='time')
                
                # print(f'Start: {start} end {end} len subset {len(subset)}')
                
                if len(subset) <= 1147:
                    print(f'No data for {run} at {station_name} (id: {station_id})')
                    continue
                
                obs = ds.sel(time=slice(start, end), runs='Obs.', wflow_id=id)
                sim = ds.sel(time=slice(start, end), runs=run, wflow_id=id)
                
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
                    
                    # fig add dots on the obs peaks using peak_dict[id][run][0]
                    obs_peaks = peak_dict[id][run][0]
                    
                    # print('max obs_peaks', max(obs_peaks))
                    
                    fig.add_trace(go.Scatter(
                        x=subset.time.values,
                        y=subset.values,
                        mode='lines',
                        name=label,
                        line=dict(color=color_dict[str(run)])
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=subset.time[obs_peaks].values,
                        y=subset[obs_peaks].values,
                        mode='markers',
                        name='Obs. Peaks',
                        marker=dict(color='red', size=8)
                    ))
                    # print('obs plotted')
                            
                else:
                    label = f'{run}: {np.mean(peak_dict[id][run][1]):.2f} +/- {np.std(peak_dict[id][run][1]):.2f}h'+'\n$NSE$:'+f'{nse:.3f}'+' $NSE_{log}$:'+f'{nse_log:.3f}'
                    print(run)
                    fig.add_trace(go.Scatter(
                        x=subset.time.values,
                        y=subset.values,
                        mode='lines',
                        name=label,
                        line=dict(color=color_dict[str(run)])
                    ))
                    
                    sim_peaks = peak_dict[id][run][0]
                    
                    #a failed run will have be less than the max obs_peaks
                    sim_peaks = sim_peaks[sim_peaks < len(subset.time)]
                    
                    # print('peak_dict', peak_dict[id][run])
                    # print('sim_peaks', len(sim_peaks))
                    # print('max sim_peaks', max(sim_peaks))
                    # print('subset.time', len(subset.time))   
                                
                    fig.add_trace(go.Scatter(
                        x=subset.time[sim_peaks].values,
                        y=subset[sim_peaks].values,
                        mode='markers',
                        name=f'{run} Peaks',
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
                fig.write_html(filename=filename)
                print(f'Peak timing plot for {station_name} (id: {station_id}) saved as\n{filename}')
                
            else:
                fig.show()

        except Exception as e:
            print('\nfail peak plots, station:', station_id, '\n')
            print(e)
            pass
        
    return peak_dict