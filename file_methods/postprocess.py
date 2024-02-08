import glob
import os
import pandas as pd
import numpy as np
import xarray as xr
from hydromt_wflow import WflowModel



#TODO: move to file_methods
#======================= search for model directories and configs ===================    
def find_model_dirs(path, snippets):
    result = []
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            if any(snippet in dir for snippet in snippets):
                result.append(os.path.join(root, dir))
    return result

def find_toml_files(filtered_dirs):
    # Find .toml files in filtered directories
    toml_files = []
    for dir in filtered_dirs:
        for file in glob.glob(os.path.join(dir, '*.toml')):
            toml_files.append(file)

    return toml_files

def find_outputs(filtered_dirs, filetype):
    # Find the output files in the filtered directories
    result = []
    for dir in filtered_dirs:
        for file in glob.glob(os.path.join(dir, '**', '*.' + filetype), recursive=True):
            result.append(file)
    return result


# ======================= Create the FR-BE-NL combined dataset =======================
def create_combined_hourly_dataset_FRBENL(working_folder, run_keys, model_dirs, toml_files, overwrite=False):
    fn_ds = os.path.join(working_folder, '_output/ds_obs_model_combined.nc')
    # ======================= Load stations/gauges to plot =======================
    # load csv that contains stations/gauges info that we want to plot
    fn_GaugeToPlot = r'/wflow_id_to_plot.csv'
    
    df_GaugeToPlot = pd.read_csv(working_folder+fn_GaugeToPlot)
    
    if not overwrite and os.path.exists(fn_ds):
        print(f'obs and model runs already combined in {fn_ds}')
        print('overwrite is false')
        print(f'loading {fn_ds}')
        
        ds = xr.open_dataset(fn_ds)
        
        return ds, df_GaugeToPlot
    
    elif overwrite or not os.path.exists(fn_ds):                    
        # ======================= Load model runs =======================
        if overwrite:
            print('overwriting the combined dataset...')
        else:
            print('combined dataset does not exist, creating...')
        
        #find the output files
        output_files = find_outputs(model_dirs, 'csv')
        
        # try:
        #     if os.path.exists(os.path.join(working_folder, 'staticgeoms')):
        #         folder_staticgeoms = os.path.join(working_folder, 'staticgeoms')
        #     else:
        #         folder_staticgeoms = os.path.join(model_dirs[0], 'staticgeoms')  # folder that contain staticgeoms
        # except Exception as e:
        #     print('Could not find staticgeoms folder')
        #     return None
        folder_staticgeoms = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202303\staticgeoms"
        
        #TODO: should be do-able with hydromt and staticgeoms
        gauges_maps = []
        for file in os.listdir(folder_staticgeoms):
            if 'gauges' in file:
                gauges_maps.append(os.path.splitext(file)[0])
        
        # mod = WflowModel(model_dirs[0], config_fn=toml_files[0], mode="r")
        mod = WflowModel(root=r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202303\run_default", mode='r', config_fn='wflow_sbm_eobs.toml')

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
        
        # load the model results into memory
        model_runs = {}
        # load model runs
        print('loading model runs...')
        
        total_len = len(run_keys)
        for n, (run, result) in enumerate(zip(run_keys, output_files), 1):
            model_runs[run] = pd.read_csv(result, parse_dates=True, index_col=0)
            print(f"Progress: {n}/{total_len} loaded ({run})")
        

        # get model station wflow_id and save them in mod_stations as numpy array
        # mod_name = "wflow_sbm_run_hourly_scale_river_n"  #toml file name
        

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
        # rng = pd.date_range(
        #     max(*[model_runs[key].index[0] for key in model_runs.keys()],
        #         *[obs_dict[key].time[0].values for key in obs_dict.keys()]),
        #     min(*[model_runs[key].index[-1] for key in model_runs.keys()],
        #         *[obs_dict[key].time[-1].values for key in obs_dict.keys()]),
        #     freq="H"
        # )
        
        rng = pd.date_range(
            max(*[model_runs[key].index[0] for key in model_runs.keys()],
                *[pd.Timestamp(obs_dict[key].time[0].values) for key in obs_dict.keys()]),
            min(*[model_runs[key].index[-1] for key in model_runs.keys()],
                *[pd.Timestamp(obs_dict[key].time[-1].values) for key in obs_dict.keys()]),
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
        #TODO: add in hourly obs data for Borgharen (wflow_id=16)
        for id in wflow_id_to_plot:
            
            country = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==id,'country'].values[0]
            
            if country=='France':
                # intersect the time ranges
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
                ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Q'].sel(dict(wflow_id=id, time=time_intersection)).values
            
            elif country=='Belgium':
                # intersect the time ranges
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Qobs_m3s'].time.values, rng)
                ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Qobs_m3s'].sel(dict(catchments=id, time=time_intersection)).values
                    #else: # country==Netherlands
                        # ! Problem: no hourly data at Borgharen?
                        #ds['Q'].loc[dict(runs='Obs.', wflow_id=id)] = obs_dict[f'{country}']['Qobs'].loc[dict(catchments=id, time=rng)].values
        
        # fill in modeled results
        for key, item in model_runs.items():
            for id in wflow_id_to_plot:
                col_name = f'Qall_{id}'  # column name of this id in model results (from Qall)
                item_reindexed = item.reindex(rng)
                ds['Q'].loc[dict(runs=key, wflow_id=id)] = item_reindexed.loc[:, col_name]
        
        ds.to_netcdf(fn_ds)
        print(f'saved combined observations and model runs to\n{fn_ds}')
        return ds, df_GaugeToPlot
