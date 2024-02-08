from metrics.peak_metrics import peak_timing_errors

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

                peaks_obs, timing_errors = peak_timing_errors(obs, sim, window=window)

                peaks_sim = (peaks_obs + timing_errors).astype(int)

                # Expand the inner dictionary with the inner loop
                peak_dict[id][run] = (peaks_sim, timing_errors)

        peak_dict[id]['Obs.'] = (peaks_obs, timing_errors)

    return peak_dict
