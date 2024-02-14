import glob
import os
import pandas as pd
import numpy as np
import xarray as xr
from hydromt_wflow import WflowModel


#======================= search for model directories and configs ===================    
def find_model_dirs(path, snippets):
    """
    Find directories that contain the specified snippets in the given path.

    Args:
        path (str): The path to search for directories.
        snippets (list): A list of snippets to search for in directory names.

    Returns:
        list: A list of directory paths that contain the specified snippets.
    """
    result = []
    for root, dirs, _ in os.walk(path):
        for directory in dirs:
            if any(snippet in directory for snippet in snippets):
                result.append(os.path.join(root, directory))
    return result

def find_toml_files(filtered_dirs):
    """
    Find .toml files in the given directories.

    Args:
        filtered_dirs (list): A list of directories to search for (usually from find_model_dirs function).toml files.

    Returns:
        list: A list of paths to the found .toml files.
    """
  
    toml_files = []
    for directory in filtered_dirs:
        for file in glob.glob(os.path.join(directory, '*.toml')):
            toml_files.append(file)

    return toml_files


def find_staticgeoms(working_dir:str=None, 
                     model_snippet:str =None, 
                     models:list=None)->list:
    """
    Find the staticgeoms directory in the working directory.
    
    Parameters:
    working_dir (str): The path to the working directory.
    model_snippet (str, optional): A snippet to filter model directories. Defaults to 'fl1d'.
    
    Returns:
    list: A list of staticgeoms directories associated with the given working directory.
    """
    
    
    
    if working_dir and models is None:
        main_folder_sg = os.path.join(working_dir, 'staticgeoms')
        if os.path.exists(main_folder_sg):
            print('Using staticgeoms from main folder')
            return main_folder_sg
        else:
            print('No staticgeoms found in main folder, try model subfolders')
            return None
    
    elif models is None and model_snippet is not None:
        models = find_model_dirs(working_dir, model_snippet)
        staticgeoms_dir = [os.path.join(model, 'staticgeoms') for model in models]
        staticgeoms_dir = [dir for dir in staticgeoms_dir if os.path.exists(dir)]
        print('staticgeoms_dir(s): ', staticgeoms_dir)
        return staticgeoms_dir
    
    elif models is not None:
        staticgeoms_dir = [os.path.join(model, 'staticgeoms') for model in models]
        staticgeoms_dir = [dir for dir in staticgeoms_dir if os.path.exists(dir)]
        print('staticgeoms_dir(s): ', staticgeoms_dir)
        return staticgeoms_dir
    
#======================= search for model outputs ===================

def find_outputs(filtered_dirs, filetype=None, filename=None):
    # Find the output files in the filtered directories
    result = []
    for directory in filtered_dirs:
        
        # directory = directory.replace('\\', '/')  # Replace backslashes with forward slashes
        print(f"Searching in directory: {directory}")  # Debug print
        if filename is not None:
            print(f"Searching for file: {filename}")  # Debug print
            # Search for the exact filename in the directory and its subdirectories
            for file in glob.glob(os.path.join(directory, '**', filename), recursive=True):
                print(f"Found file: {file}")  # Debug print
                result.append(file)
                # break
            # break   
    
        elif filetype is not None:
            print(f"Searching for filetype: {filetype}")  # Debug print
            # Search for any file with the given extension in the directory and its subdirectories
            for file in glob.glob(os.path.join(directory, '**', '*.' + filetype), recursive=True):
                print(f"Found file: {file}")  # Debug print
                result.append(file)
                
    return result

# ======================= Create the FR-BE-NL combined dataset =======================
def create_combined_hourly_dataset_FRBENL(working_folder:str,
                                          run_keys:list,
                                          model_dirs:list, 
                                          output:str, 
                                          toml_files:list, 
                                          overwrite:bool=False,
                                          model_snippet:str=None):
    '''
    This function creates a combined dataset of hourly model runs and observations for France, Belgium and Netherlands.
    The dataset is saved as a netCDF file in the working_folder.
    
    Args:
    working_folder: str, path to the working folder containing all model runs
    run_keys: list, list of run keys (derived from model runs if not supplied)
    model_dirs: list, list of directories containing the model runs
    output: str, filename pattern to search for in the model output files
    toml_files: list, list of TOML configuration files for the model runs
    overwrite: bool, flag indicating whether to overwrite the existing combined dataset if it exists
    
    Returns:
    ds: xarray.Dataset, combined dataset of model runs and observations
    df_GaugeToPlot: pandas.DataFrame, dataframe containing information about stations/gauges to plot
    '''
    
    fn_ds = os.path.join(working_folder, '_output/ds_obs_model_combined.nc')

    
    fn_ds = os.path.join(working_folder, '_output/ds_obs_model_combined.nc')
    
    # ======================= Load stations/gauges to plot =======================
    # load csv that contains stations/gauges info that we want to plot
    fn_GaugeToPlot = 'wflow_id_to_plot.csv'
    
    df_GaugeToPlot = pd.read_csv(os.path.join(working_folder, fn_GaugeToPlot))
    
    if not overwrite and os.path.exists(fn_ds):
        print(f'obs and model runs already combined in {fn_ds}')
        print('overwrite is false')
        print(f'loading {fn_ds}')
        
        ds = xr.open_dataset(fn_ds)
        
        return ds, df_GaugeToPlot
    
    elif overwrite or not os.path.exists(fn_ds):
        # try:
                        
        # ======================= Load model runs =======================
        if overwrite:
            print('overwriting the combined dataset...')
        else:
            print('combined dataset does not exist, creating...')
        os.makedirs(os.path.join(working_folder, '_output'), exist_ok=True)
    
        #find the output files
        if output in ['csv', 'nc']:
            output_files = find_outputs(model_dirs, filetype=output)
        else:
            output_files = find_outputs(model_dirs, filename=output)
        
        print(output_files)
        
        # try:
        #     if os.path.exists(os.path.join(working_folder, 'staticgeoms')):
        #         folder_staticgeoms = os.path.join(working_folder, 'staticgeoms')
        #     else:
        #         folder_staticgeoms = os.path.join(model_dirs[0], 'staticgeoms')  # folder that contain staticgeoms
        # except Exception as e:
        #     print('Could not find staticgeoms folder')
        #     return None
        folder_staticgeoms = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202303\staticgeoms"
        
        # folder_staticgeoms = find_staticgeoms(working_dir=working_folder)
        
        #TODO: should be do-able with hydromt and staticgeoms, Joost says there's no perfect approach here tbh, would have to do runs that are not modifying toml in memory. 
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
        
        print('\nmod_stations:\n', mod_stations)
        
        # ====================== load the model results into memory
        
        print('\nloading model runs...\n')
        
        print(output_files)
        
        model_runs = {}
        
        total_len = len(run_keys)
        for n, (run, result) in enumerate(zip(run_keys, output_files), 1):
            model_runs[run] = pd.read_csv(result, parse_dates=True, index_col=0)
            print(f'from time: {model_runs[run].index[0]} to {model_runs[run].index[-1]}')
            print(f"Progress: {n}/{total_len} loaded ({run}, len: {len(model_runs[run])})")
        
        # ======================= Load observation/measurement data =======================
        # load observation/measurement data from France, Belgium and Netherlands in a dictionary
        fn_France = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\FR-Hydro-hourly-2005_2022.nc'
        fn_Belgium = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\qobs_hourly_belgian_catch.nc'
        fn_Netherlands = R'p:\11209265-grade2023\wflow\wflow_meuse_julia\measurements\rwsinfo_hourly.nc'
        
        obs_dict = {}
        
        for country in ['France', 'Belgium', 'Netherlands']:
            obs_dict[f'{country}'] = xr.open_dataset(locals()['fn_'+country])


        # ======================= create xarray dataset to store modeled and obs data =======================
        # Convert time values from nc files to pandas Timestamp objects
        # Convert time values from nc files to pandas Timestamp objects
        model_runs_time_values = [pd.to_datetime(model_runs[key].index) for key in model_runs.keys()]
        obs_time_values = [pd.to_datetime(obs_dict[key].time.values) for key in obs_dict.keys()]

        # Flatten the lists of DatetimeIndex objects into single lists
        model_runs_time_values_flat = [time for sublist in model_runs_time_values for time in sublist]
        obs_time_values_flat = [time for sublist in obs_time_values for time in sublist]

        # Determine the min/max time values
        model_run_min_time, model_run_max_time = min(model_runs_time_values_flat), max(model_runs_time_values_flat)
        obs_min_time, obs_max_time             = min(obs_time_values_flat), max(obs_time_values_flat)

        print(f'model_run_min_time: {model_run_min_time}\nmodel_run_max_time: {model_run_max_time}')
        print(f'obs_min_time: {obs_min_time}\nobs_max_time: {obs_max_time}')
        
        # Determine the common min/max range of time
        common_min_time = max(model_run_min_time, obs_min_time)
        common_max_time = min(model_run_max_time, obs_max_time)
        
        print(f'\ncommon_min_time: {common_min_time}\ncommon_max_time: {common_max_time}\n')

        # Generate the range of time values
        rng = pd.date_range(common_min_time, common_max_time, freq="H")

        #======================= create the combined dataset =======================
        # get the wflow_id to plot
        wflow_id_to_plot = [*df_GaugeToPlot.wflow_id.values]

        # get the runs name
        runs =['Obs.', *list(model_runs.keys())]

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
        
        print('\nEmpty Dataset:\n', ds)
        
        ds = ds * np.nan

        # fill in obs data
        for wflow_id in wflow_id_to_plot:
            
            country = df_GaugeToPlot.loc[df_GaugeToPlot['wflow_id']==wflow_id,'country'].values[0]
            
            if country=='France':
                # intersect the time ranges
                print(f'obs_dict[f\'{country}\'][\'Q\'].time.values: {obs_dict[f"{country}"]["Q"].time.values.shape}')
                print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.min: {obs_dict[f"{country}"]["Q"].time.values.min()}')
                print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.max: {obs_dict[f"{country}"]["Q"].time.values.max()}')
                
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
                print(f'time_intersection: {time_intersection.shape}')
                
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
            
            elif country=='Belgium':
                # intersect the time ranges
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Qobs_m3s'].time.values, rng)
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Qobs_m3s'].sel({'catchments':wflow_id, 'time':time_intersection}).values
                
            else:
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
        
        print('\nmodel_runs:\n', model_runs)
        print('\nmodel_runs.keys():\n', model_runs.keys())
        print('\n Dataset:\n', ds)
        
        # fill in modeled results
        for run, item in model_runs.items():
            print(list(item.columns))
            
            for wflow_id in wflow_id_to_plot:
                try:
                    col_name = f'Q_{wflow_id}'  # column name of this id in model results (from Qall)
                    
                    item_reindexed = item.reindex(rng)
                    
                    ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
                
                except Exception as e:
                    print(f'Could not find: {run} {wflow_id} {col_name}')
                        
                    try:
                        col_name = f'Q_locs_{wflow_id}'  # column name of this id in model results (from Qall)

                        item_reindexed = item.reindex(rng)
                        
                        ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
                    
                    except Exception as e:
                        print(f'Could not find: {run} {wflow_id} {col_name}')
                        print(e)
                    
        # except Exception as e:
        #     print(col_name)
        #     print('Could not find the column name in model results')
        #     print(e)
                        
                        
        # save the combined dataset
        ds.to_netcdf(fn_ds)
        
        print(f'saved combined observations and model runs to\n{fn_ds}')
        return ds, df_GaugeToPlot
