import glob
import os
import pandas as pd
import numpy as np
import xarray as xr
from hydromt_wflow import WflowModel
import seaborn as sns 



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
        if root == path:  # Ensure that we are at the first level
            if not snippets:  # If snippets list is empty, add all directories.
                for directory in dirs:
                    result.append(os.path.join(root, directory))
            else:
                for directory in dirs:
                    if any(snippet in directory for snippet in snippets):
                        result.append(os.path.join(root, directory))
        break  # Exit after processing the first level to avoid going deeper
    return result

#==============================================================================	
def find_toml_files(filtered_dirs):
    """
    Find .toml files in the given directories.

    Args:
        filtered_dirs (list): A list of directories to search for (usually from find_model_dirs function).toml files.

    Returns:
        list: A list of paths to the found .toml files.
    """
    if filtered_dirs is str and not list:
        filtered_dirs = [filtered_dirs]
    
    toml_files = []
    for directory in filtered_dirs:
        for file in glob.glob(os.path.join(directory, '*.toml')):
            toml_files.append(file)

    return toml_files

#==============================================================================

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
                                          overwrite:bool=False,):
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
    
    # ======================= Load stations/gauges to plot =======================
    # load csv that contains stations/gauges info that we want to plot
    fn_gaugetoplot = 'wflow_id_add_HBV.csv'
    #currently 
    df_gaugetoplot = pd.read_csv(os.path.join(working_folder, fn_gaugetoplot))

    if not overwrite and os.path.exists(fn_ds):
        print(f'obs and model runs already combined in {fn_ds}')
        print('overwrite is false')
        print(f'loading {fn_ds}')
        
        ds = xr.open_dataset(fn_ds)
        
        return ds, df_gaugetoplot

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
        
        # ====================== load the model results into memory
        print('\nloading model runs...\n')
        
        print(output_files)
        
        model_runs = {}
        
        total_len = len(run_keys)
        
        for n, (run, result) in enumerate(zip(run_keys, output_files), 1):
            print(f'loading {run} from {result}')
            model_runs[run] = pd.read_csv(result, parse_dates=True, index_col=0)
            print(f'from time: {model_runs[run].index[0]} to {model_runs[run].index[-1]}')
            print(f"Progress: {n}/{total_len} loaded ({run}, len: {len(model_runs[run])})")
            
        #==================== Load the HBV data =======================
        #weird indexing but we have the link already from the new add HBV
        hbv_sdf = pd.read_excel(r"P:\11209265-grade2023\wflow\wflow_meuse_julia\HBV\HBV_60min_stations_2004-2016.xlsx", index_col=0, parse_dates=True, skiprows=[0,1,2,4,5])

        hbv_cdf = pd.read_excel(r"P:\11209265-grade2023\wflow\wflow_meuse_julia\HBV\HBV_60min_Centroids_2004-2016.xlsx", index_col=0, parse_dates=True, skiprows=[0,1,2,4,5])

        hbv_dict = {}

        hbv_cat = pd.concat([hbv_sdf, hbv_cdf], axis=1)

        HBV_IDs = df_gaugetoplot[['wflow_id', 'HBV_ID']].replace('0', np.nan)
        HBV_IDs = HBV_IDs.dropna()
                                    
        for n, (index, row) in enumerate(HBV_IDs.iterrows()):
            gauge = row['wflow_id']
            ids = row['HBV_ID']
            hbv_dict[f'HBV_{gauge}'] = hbv_cat[ids]

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
        wflow_id_to_plot = [*df_gaugetoplot.wflow_id.values]

        # get the runs name
        runs =['Obs.','HBV', *list(model_runs.keys())]

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
        
        # print('\nEmpty Dataset:\n', ds)
        
        ds = ds * np.nan

        # fill in obs data
        for wflow_id in wflow_id_to_plot:
            
            country = df_gaugetoplot.loc[df_gaugetoplot['wflow_id']==wflow_id,'country'].values[0]
            
            if country=='France':
                # intersect the time ranges
                # print(f'obs_dict[f\'{country}\'][\'Q\'].time.values: {obs_dict[f"{country}"]["Q"].time.values.shape}')
                # print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.min: {obs_dict[f"{country}"]["Q"].time.values.min()}')
                # print(f'obs_dict[f\'{country}\'][\'Q\'].time.values.max: {obs_dict[f"{country}"]["Q"].time.values.max()}')
                
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
                
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
                
                try:
                    ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}] = hbv_dict[f'HBV_{wflow_id}'].reindex(ds['Q'].coords['time'])
                    # ic(ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}])
                    
                except Exception as e:
                    print(e)
                    None
                
            elif country=='Belgium':
                # intersect the time ranges
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Qobs_m3s'].time.values, rng)
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Qobs_m3s'].sel({'catchments':wflow_id, 'time':time_intersection}).values
                
                try:
                    ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}] = hbv_dict[f'HBV_{wflow_id}'].reindex(ds['Q'].coords['time'])
                    # ic(ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}])
                    
                except Exception as e:
                    print(e)
                    None
                
            else:
                time_intersection = np.intersect1d(obs_dict[f'{country}']['Q'].time.values, rng)
                ds['Q'].loc[{'runs':'Obs.', 'wflow_id':wflow_id}] = obs_dict[f'{country}']['Q'].sel({'wflow_id':wflow_id, 'time':time_intersection}).values
                try:
                    ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}] = hbv_dict[f'HBV_{wflow_id}'].reindex(ds['Q'].coords['time'])
                    # ic(ds['Q'].loc[{'runs':'HBV', 'wflow_id':wflow_id}])
                    
                except Exception as e:
                    print(f'Could not find HBV_{wflow_id}')
                    # print(e)
                    
        print('\nmodel_runs:\n', model_runs)
        print('\nmodel_runs.keys():\n', model_runs.keys())
        print('\n Dataset:\n', ds)
        
        # fill in modeled results
        for run, item in model_runs.items():
            # print(list(item.columns))
            
            for wflow_id in wflow_id_to_plot:
                try:
                    col_name = f'Q_{wflow_id}'  # column name of this id in model results (from Qall)
                    item_reindexed = item.reindex(rng)
                    ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
                
                except Exception as e:
                    # print(f'Could not find: {run} {wflow_id} {col_name}')
                        
                    try:
                        col_name = f'Q_locs_{wflow_id}'  # column name of this id in model results (from Qall)
                        item_reindexed = item.reindex(rng)
                        ds['Q'].loc[{'runs': run, 'wflow_id': wflow_id}] = item_reindexed.loc[:, col_name]
                    
                    except Exception as e:
                        # print(f'CoSuld not find: {run} {wflow_id} {col_name}')
                        # print(e)
                        None
                    
        # except Exception as e:
        #     print(col_name)
        #     print('Could not find the column name in model results')
        #     print(e)
                        
                        
        # save the combined dataset
        ds.to_netcdf(fn_ds)

        print(f'saved combined observations and model runs to\n{fn_ds}')
        return ds, df_gaugetoplot
