from metrics.peak_metrics import peak_timing_errors
import numpy as np
import pandas as pd

# def store_peak_info(ds, df_GaugeToPlot, id_key, window):
#     # Store peak timing information in a dictionary
#     peak_dict = {}

#     for id in ds[id_key].values:
#         station_id = id

#         # select a station using the id grouping and the sub selection id (station_id)
#         ds_sub = ds.sel({id_key:station_id})

#         # get obs data
#         obs = ds_sub.sel(runs='Obs.').Q

#         peak_dict[id] = {}

#         for run in ds_sub.runs.values:
#             if run != 'Obs.':
#                 sim = ds_sub.sel(runs=run).Q

#                 peaks_obs, timing_errors = peak_timing_errors(obs, sim, window=window)

#                 peaks_sim = (peaks_obs + timing_errors).astype(int)

#                 # Expand the inner dictionary with the inner loop
#                 peak_dict[id][run] = (peaks_sim, timing_errors)

#         peak_dict[id]['Obs.'] = (peaks_obs, timing_errors)

#     return peak_dict


def store_peak_info(ds, df_GaugeToPlot, id_key, window):
    # Store peak timing information in a dictionary
    peak_dict = {}
    
    #TODO: make any recordings that are exaxtly window distance to nan

    for id in ds[id_key].values:
        station_id = id

        # select a station using the id grouping and the sub selection id (station_id)
        ds_sub = ds.sel({id_key:station_id})

        # get obs data
        obs = ds_sub.sel(runs='Obs.').Q
        obs = obs.ffill('time')

        peak_dict[id] = {}

        for run in ds_sub.runs.values:
            if run != 'Obs.':
                
                #Sometimes Q is empty, we'd rather see what is empty in the plot than have an error
                if ds_sub.sel(runs=run).Q.isnull().all():
                    continue
                
                sim = ds_sub.sel(runs=run).Q
                
                peaks, timing_errors = peak_timing_errors(obs, sim, window=window)

                #some timing errors should be nan where they aree the same as the window.
                mask = np.abs(timing_errors) == window
                timing_errors[mask] = np.nan
                peaks[mask] = np.nan
                
                # print(f'run, peaks, timing_errors: {run}, \n {peaks}, \n  {timing_errors}')
                # Check if peaks is empty
                if len(peaks) > 0 and not np.isnan(peaks).all() and not np.isnan(timing_errors).all():
                    
                    # Convert timing_errors to timedelta (assuming timing_errors are in hours)
                    timing_errors_timedelta = pd.to_timedelta(timing_errors, unit='h')

                    # Add timedelta to datetime
                    peaks_sim = peaks + timing_errors_timedelta

                    mean_peak_timing = np.mean(np.abs(timing_errors))
                    
                    peaks_index = pd.DatetimeIndex(peaks)
                    obs_Q = obs.sel(time=peaks_index).values
                    sim_Q = sim.sel(time=peaks_index).values
                    
                    peak_mape = np.sum(np.abs((sim_Q - obs_Q) / obs_Q)) / peaks.size * 100
                
                else:
                    peaks_sim = np.nan
                    mean_peak_timing = np.nan
                    peak_mape = np.nan

                # Expand the inner dictionary with the inner loop
                peak_dict[id][run] = {'peaks': peaks_sim, 
                                      'timing_errors': timing_errors, 
                                      'mean_peak_timing': mean_peak_timing,
                                      'peak_mape': peak_mape}

        peaks_obs, _ = peak_timing_errors(obs, obs, window=window)  # Calculate peaks for 'Obs.' with itself
        peak_dict[id]['Obs.'] = {'peaks': peaks_obs, 
                                 'timing_errors': np.zeros_like(peaks_obs), 
                                 'mean_peak_timing': np.nan,
                                 'peak_mape': np.nan}

    return peak_dict