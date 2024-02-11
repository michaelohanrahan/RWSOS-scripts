from metrics.peak_metrics import peak_timing_errors
import numpy as np

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

    for id in ds[id_key].values:
        station_id = id

        # select a station using the id grouping and the sub selection id (station_id)
        ds_sub = ds.sel({id_key:station_id})

        # get obs data
        obs = ds_sub.sel(runs='Obs.').Q

        peak_dict[id] = {}

        for run in ds_sub.runs.values:
            if run != 'Obs.':
                sim = ds_sub.sel(runs=run).Q

                peaks, timing_errors = peak_timing_errors(obs, sim, window=window)

                peaks_sim = (peaks + timing_errors).astype(int)
                # Check if peaks is empty
                if len(peaks) > 0:
                    mean_peak_timing = np.mean(np.abs(timing_errors))
                    obs_Q = obs[peaks].values
                    sim_Q = sim[peaks_sim].values
                    peak_mape = np.sum(np.abs((sim_Q - obs_Q) / obs_Q)) / peaks.size * 100
                else:
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