
'''Still working on this script, 
    but the idea is to identify files, and check contents with a list of gauges of interest.
    we know we want hourly gauges, we want to know which set of gauges is most relevant and if we
    have to build another grid. 
    
    once established we then perform a sequential scaling of low to high dependency subcatchments
    
    NEEDS: To finish the overall approach, currently finished checks and moving onto masking and scaling.
    NICE: .'''

#%%

from hydromt_wflow import WflowModel
# from model_building.update_gauges import main as update_gauges
from file_methods.postprocess import find_model_dirs, find_toml_files
import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
# import numpy as np
from hydromt.log import setuplog


def list_files_containing(root:str, snippet:str)->list:
    """
    List all files containing a given snippet.
    
    Parameters:
    root (str): The path to the root directory.
    snippet (str): The snippet to search for.
    
    Returns:
    list: A list of filepaths containing the given snippet.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if snippet in filename:
                files.append(os.path.join(dirpath, filename))
    
    return files

def find_staticgeoms(working_dir:str=None, 
                     model_snippet:str = 'fl1d', 
                     models:list=None)->list:
    """
    Find the staticgeoms directory in the working directory.
    
    Parameters:
    working_dir (str): The path to the working directory.
    model_snippet (str, optional): A snippet to filter model directories. Defaults to 'fl1d'.
    
    Returns:
    list: A list of staticgeoms directories associated with the given working directory.
    """
    
    models = find_model_dirs(working_dir, model_snippet)
    
    
    if working_dir and models is None:
        
        main_folder_sg = os.path.join(working_dir, 'staticgeoms')
        if os.path.exists(main_folder_sg):
            print('Using staticgeoms from main folder')
            return main_folder_sg
        else:
            print('No staticgeoms found in main folder, try model subfolders')
            return None
    if models:
        staticgeoms_dir = [os.path.join(model, 'staticgeoms') for model in models]
        staticgeoms_dir = [dir for dir in staticgeoms_dir if os.path.exists(dir)]
        print('staticgeoms_dir:', staticgeoms_dir)
        return staticgeoms_dir

def build_dependency_dict(wflow_id:list):
    dependency_dict = {}

    for id in wflow_id:
        dependency_dict[id] = []

    low_dependency = [101, 201, 6, 9, 701, 801, 11, 10, 12, 13,15,703]

    for i in low_dependency:
        dependency_dict[i] = []

    dependency_dict[3] = [101]
    dependency_dict[4] = [3, *dependency_dict[3], 201, 5, 6]
    dependency_dict[1401] = [13,9,703,701,*dependency_dict[4], 4, 801]
    dependency_dict[1201] = [12,11,10]
    dependency_dict[1402] = [*dependency_dict[1401], 1401, *dependency_dict[1201], 1201]
    dependency_dict[16] = [15, *dependency_dict[1402], 1402]
    
    return dependency_dict

def process_gauges(models:list,
                   wflow_id:list,
                   staticgeoms_dir:str,
                   check_locs:bool = False)->list:
    """
    Process gauges associated with a given working directory and CSV directory.
    
    Parameters:
    working_dir (str): The path to the working directory.
    csv_dir (str): The path to the CSV directory.
    model_snippet (str, optional): A snippet to filter model directories. Defaults to 'fl1d'.
    
    Returns:
    list: A list of gauge files associated with the given working directory.
    """
    
    gauges_append = list_files_containing(staticgeoms_dir, 'gauges')
    
    gauges_files = [os.path.join(staticgeoms_dir, file) for file in gauges_append]
    
    wflow_gauge_data = []
    best_match_file_names = []
    
    for i, gauge in enumerate(gauges_files):
        
        file = gpd.read_file(gauge)
        
        if 'wflow_id' in file.columns:
            # Count how many of the wflow id's are in the file
            print(f'The gauges in the file {gauges_append[i]} have wflow ids:\n{file.wflow_id.isin(wflow_id).sum()}/{len(wflow_id)}')
            
            if file.wflow_id.isin(wflow_id).sum() == len(wflow_id):
                file.set_index('wflow_id', inplace=True)
                wflow_gauge_data.append(file)
                best_match_file_names.append(gauges_append[i])

    if len(gauges_files) > 0 and check_locs is True: 
        #perform pairwise comparison of point locations associated with wflow_id
        
        for id in wflow_id:
            points = []
            for file in wflow_gauge_data:
                # file.set_index('wflow_id', inplace=True)
                #get the geom at location of wflow_id
                point = file.loc[id].geometry
                points.append(point)
                
            #compare the points
            if points[0] == points[1]:
                print(f'wflow_id {id} has the same location in both files')
                
            else:
                print(f'wflow_id {id} has DIFFERENT locations')
    
    
    match_dict = {file:data for file,data in zip(best_match_file_names,wflow_gauge_data)}
    
    return match_dict, models, tomls, wflow_id

def plot_dep_catchments(dep_id:list, gdf_sb, gdf_gg, basin, ax):
    artists = []  # list to store the artists

    artist = basin.plot(ax=ax, color='lightgrey')
    artists.append(artist)

    artist = gdf_sb.plot(ax=ax, color='none', edgecolor='black')
    artists.append(artist)
    artist = gdf_sb[gdf_sb.index.isin(dep_id)].plot(ax=ax, color='green', edgecolor='black')
    artists.append(artist)

    artist = gdf_gg.plot(ax=ax, color='red')
    artists.append(artist)
    artist = gdf_gg[gdf_gg.index.isin(dep_id)].plot(ax=ax, color='blue')
    artists.append(artist)

    return artists  # return the list of artists

def dependency_solve(dependency_dict, done):
    
    #process now is cleared with each iteration
    process_now = []
    
    for key, dep_list in dependency_dict.items():
        
        # l3vel 1
        if len(dep_list) < 1 and not any(k in done for k in [key]):
            process_now.append(key)
        
        # for each dep list check if it is satisfied by the done list
        elif all(item in done for item in dep_list) and key not in done:
            process_now.append(key)
        else:
            None
    
    return process_now

def read_model(level, working_dir, model_snippet, models, tomls, scale:None):
    """
    #   For the level 1 we read the base model (fl1d_lakes) and then we set the root to the new model, and write with the new root
    #   The new root will contain the new model with altered grid
    """
    

    if level==1:
        mod = WflowModel(root=models[0], config_fn=tomls[0], mode='r', logger=logger)
        print(f'\n - Level {level}, reading model {models[0]}\n')

    elif level > 1:
        model_fn = find_model_dirs(os.path.join(working_dir, f'{model_snippet}_level{level-1}'), 'base')
        print(f' - Level {level}, reading model {model_fn[0]}')
        toml = find_toml_files(model_fn)
        mod = WflowModel(root=model_fn[0], config_fn=toml[0], mode='r', logger=logger)

    mod.read()
    mod.read_config()
    mod.read_geoms()
    mod.read_grid() 
    
    if scale:
        run=scale
        new_root = os.path.join(working_dir, f'{model_snippet}_level{level-1}', str(run))
        
    else:
        run='base'
        new_root = os.path.join(working_dir, f'{model_snippet}_level{level}', str(run))
    
    os.makedirs(new_root, exist_ok=True)
    
    mod.set_root(root=new_root, mode='w+')
    
    return mod

def set_scale_grid_per_subcatchment(process_now, mod, scale_table, level, subcatch_layer, varname):
    '''
    level: int, the level of catchments being processed, 1-indexed, 1 being the most upstream with no dependencies
    staticmap: xarray, the staticmap to be masked
    
    '''
    sm = mod.grid[varname]
    sub = mod.grid[subcatch_layer]
    
    for id in process_now:
        print('processing:', id)
        
        mask = sub == id
        try:
            scale = scale_table.loc[id, f'level{level}']
        
        except KeyError:
            scale = 1
            print(f'Catchment {id} not in scale table, using scale factor 1')
            continue
        
        sm.values[mask] = sm.values[mask] * scale
        
        # Append the scale factor to the text file
        with open('scale_factors.txt', 'a') as file:
            file.write(f"Catchment {id}: Scale Factor = {scale}\n")
            
        print(f' - Catchment {id} scaled by {scale}')
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(f'Level {level} catchments, {varname}')
    sm.plot(ax=ax, vmin=0, vmax=1)
    plt.savefig(os.path.join(mod.root, f'level{level}_{varname}.png'))
    print('overwriting staticmap')
    
    mod.grid[varname] = sm
    mod.write_grid()

def iterate_scales_for_next(process_now, mod, scale, level, subcatch_layer, varname):
    '''
    level: int, the level of catchments being processed, 1-indexed, 1 being the most upstream with no dependencies
    staticmap: xarray, the staticmap to be masked
    
    '''
    sm = mod.grid[varname]
    sub = mod.grid[subcatch_layer]
    
    for id in process_now:
        print('processing:', id)
        
        mask = sub == id

        sm.values[mask] = sm.values[mask] * scale
        
        # Append the scale factor to the text file
        with open('scale_factors.txt', 'a') as file:
            file.write(f"Catchment {id}: Test Scale Factor = {scale}\n")
            
        print(f' - Catchment {id} scaled by {scale}')
    
    fig, ax = plt.subplots(figsize=(10,10))
    plt.title(f'Level {level} catchments, {varname}')
    sm.plot(ax=ax, vmin=0, vmax=1)
    plt.savefig(os.path.join(mod.root, f'level{level}_{varname}.png'))
    print('overwriting staticmap')
    
    mod.grid[varname] = sm
    mod.write_grid()


#%%

#Declare the main working folder 
working_dir = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N"
#declare model snippets that indicate the model subfolders
model_snippet = 'fl1d'
models = find_model_dirs(working_dir, model_snippet)
tomls = find_toml_files(models)


#the csv that contains the translation of wflow_id to plot_id
csv_dir = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\wflow_id_to_plot.csv"
wflow_to_id = pd.read_csv(csv_dir,index_col='wflow_id')
wflow_id = list(wflow_to_id.index.values)


#The table by which the scales are applied, in this case already loaded with level 1 scales
scale_table = pd.read_csv(r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\wflow_id_scale_levels.csv", index_col='wflow_id')


#discover statigeoms in this dir so that we can find the gauges
staticgeoms_dir = find_staticgeoms(working_dir=working_dir)

#for checking locations and gauges
match_dict, models, tomls, wflow_id = process_gauges(models, 
                                                    wflow_id,
                                                    staticgeoms_dir[0],
                                                    check_locs=False)      

dependency_dict = build_dependency_dict(wflow_id)

#%%
#Fill this in manually for after reviewing the gauges
chosen = 'locs'

# use the data from associated with the key containing the chosen string
for key in match_dict.keys():
    if chosen in key:
        chosen_gauges = match_dict[key]
        chosen_key = key

print(f'Chosen Gauges', chosen_key, chosen_gauges)

#use this info to create the geoms that will alter the grid 
subcatchment_files = list_files_containing(staticgeoms_dir[0], 'subcatch')
chosen_subcatchment = [file for file in subcatchment_files if chosen in file]

#Subcatchments
gdf_sb = gpd.read_file(chosen_subcatchment[0])
gdf_sb.rename({'value':'wflow_id'}, inplace=True, axis=1)
gdf_sb.set_index('wflow_id', inplace=True, drop=True)

#Gauges
gdf_gg = chosen_gauges


#Basin shape for plotting
basin = gpd.read_file('P:\\11209265-grade2023\\wflow\\wflow_meuse_julia\\wflow_meuse_per_catchment_N\\fl1d_lakes\\staticgeoms\\basins.geojson')

#The wflow ID of the most dependent catchment
high_dep_wflow_id = 16

#The logger for the model building
logger = setuplog("build", log_level=20)

# %%

#%%
#After setting optimal values at each level, we access and scale the next level of catchments
#Skipping 1 as that will be accounted for by the 'base'
scales = [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3]

done = set()

#currently working with one level at a time
for level in range(1, 2):
    print(level)
    
    #=======================
    # This level dependency is determined and the done set is updated to inform the 
    # next iteration of the catchments what dependencies are satisfied
    #=======================
    process_now = dependency_solve(dependency_dict, done)
    
    done.update(process_now)    
    
    #plot to show progress
    fig, ax = plt.subplots(figsize=(10,10))
    plot_dep_catchments(process_now, gdf_sb, gdf_gg, basin, ax)
    plt.title(f'Level {level} catchments')
    plt.show()

    #=======================
    # find model, read, and set new root
    #=======================
    
    mod = read_model(level, working_dir, model_snippet, models, tomls, None)
    
    print(f'Modify grid at level {level}')

    #=======================
    # write staticmap for this new level base. Using the chosen scale in scale table
    #=======================
    
    staticmap = mod.grid
    varname = 'N_River'
    
    set_scale_grid_per_subcatchment(process_now, mod, scale_table, level, f'wflow_subcatch_{chosen}', varname)
    
    print('writing config')
    mod.write_config('wflow_sbm.toml', mod.root)
    print('writing geoms')
    mod.write_geoms("staticgeoms")
    print(f'Model written to {mod.root}')
    
    #We do not update the set done. So we can look ahead and scale the next level of catchments
    process_next = dependency_solve(dependency_dict, done)

    #TODO: Inefficient to read grid when we could just copy the grid from the previous model, but hey, it works
    # it also keeps things readable from the hydromt side of things re: file structure
    for scale in scales:
        mod = read_model(level+1, working_dir, model_snippet, models, tomls, scale)
        
        staticmap = mod.grid
        
        iterate_scales_for_next(process_next, mod, scale, level, f'wflow_subcatch_{chosen}', varname)
        
        print('writing config')
        mod.write_config('wflow_sbm.toml', mod.root)
        
        print('writing geoms')
        
        mod.write_geoms("staticgeoms")
        
        print(f'Model scale {scale} written to {mod.root}')
    
    print('finished iteration level:', level)
        
    print('\n - Done:', done)
    print('\n - Process_now:', process_now)
    print('\n - Finished iteration level:', level)
    
    if high_dep_wflow_id in done:
        print('done after iteration:', level)
        break
    else:
        continue    
    

        
    