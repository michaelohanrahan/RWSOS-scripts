
'''Still working on this script, 
    but the idea is to identify files, and check contents with a list of gauges of interest.
    we know we want hourly gauges, we want to know which set of gauges is most relevant and if we
    have to build another staticmaps. 
    
    once established we then perform a sequential scaling of low to high dependency subcatchments
    
    NEEDS: To finish the overall approach, currently finished checks and moving onto masking and scaling.
    NICE: .'''

#%%

from hydromt_wflow import WflowModel
from model_building.update_gauges import main as update_gauges
from file_methods.postprocess import find_model_dirs, find_toml_files
import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt


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
    dependency_dict[4] = [*dependency_dict[3], 201, 5, 6]
    dependency_dict[1401] = [13,9,703,701,*dependency_dict[4], 801]
    dependency_dict[1201] = [12,11,10]
    dependency_dict[1402] = [*dependency_dict[1401], *dependency_dict[1201]]
    dependency_dict[16] = [15, *dependency_dict[1402]]
    
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

    basin.plot(ax=ax, color='lightgrey')

    gdf_sb.plot(ax=ax, color='none', edgecolor='black')
    gdf_sb[gdf_sb.index.isin(dep_id)].plot(ax=ax, color='green', edgecolor='black')

    gdf_gg.plot(ax=ax, color='red')
    gdf_gg[gdf_gg.index.isin(dep_id)].plot(ax=ax, color='blue')

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

#discover statigeoms in this dir so that we can find the gauges
staticgeoms_dir = find_staticgeoms(working_dir=working_dir)

match_dict, models, tomls, wflow_id = process_gauges(models, 
                                                    wflow_id,
                                                    staticgeoms_dir[0],
                                                    check_locs=False)      

dependency_dict = build_dependency_dict(wflow_id)

#%%
chosen = 'locs'
# use the data from associated with the key containing the chosen string
for key in match_dict.keys():
    if chosen in key:
        chosen_gauges = match_dict[key]
        chosen_key = key

# print(f'Chosen Gauges', chosen_key, chosen_gauges)

subcatchment_files = list_files_containing(staticgeoms_dir[0], 'subcatch')
chosen_subcatchment = [file for file in subcatchment_files if chosen in file]
gdf_sb = gpd.read_file(chosen_subcatchment[0])
gdf_sb.rename({'value':'wflow_id'}, inplace=True, axis=1)
gdf_sb.set_index('wflow_id', inplace=True, drop=True)
gdf_gg = chosen_gauges

# %%

#plotting the gauges and subcatchments
basin = gpd.read_file('P:\\11209265-grade2023\\wflow\\wflow_meuse_julia\\wflow_meuse_per_catchment_N\\fl1d_lakes\\staticgeoms\\basins.geojson')
fig, ax = plt.subplots(figsize=(10,10))
plot_dep_catchments(wflow_id, gdf_sb, gdf_gg, basin, ax)


