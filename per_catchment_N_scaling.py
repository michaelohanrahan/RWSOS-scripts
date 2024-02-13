from hydromt_wflow import WflowModel
from model_building.update_gauges import main as update_gauges
from file_methods.postprocess import find_model_dirs, find_toml_files
import geopandas as gpd
import os
import pandas as pd

working_dir = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N"
model_snippet = 'fl1d'
csv_dir = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N\wflow_id_to_plot.csv"

def list_dirs_containing(root:str, snippet:str)->list:
    """
    List all directories containing a given snippet.
    
    Parameters:
    root (str): The path to the root directory.
    snippet (str): The snippet to search for.
    
    Returns:
    list: A list of directories containing the given snippet.
    """
    
    dirs = [os.path.join(root, dir) for root in roots for dir in os.listdir(root) if snippet in dir]
    
    return dirs

def find_staticgeoms(working_dir:str, model_snippet:str = 'fl1d')->list:
    """
    Find the staticgeoms directory in the working directory.
    
    Parameters:
    working_dir (str): The path to the working directory.
    model_snippet (str, optional): A snippet to filter model directories. Defaults to 'fl1d'.
    
    Returns:
    list: A list of staticgeoms directories associated with the given working directory.
    """
    
    models = find_model_dirs(working_dir, model_snippet)
    main_folder_sg = [os.path.join(working_dir, 'staticgeoms')]
    
    if os.path.exists(main_folder_sg):
        return main_folder_sg
    
    staticgeoms_dir = [os.path.join(model, 'staticgeoms') for model in models]
    staticgeoms_dir = [dir for dir in staticgeoms_dir if os.path.exists(dir)]
    print('staticgeoms_dir:', staticgeoms_dir)
    return staticgeoms_dir

def process_gauges(working_dir:str,
                   csv_dir:str,
                   model_snippet:str = 'fl1d', 
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
    
    models = find_model_dirs(working_dir, model_snippet)
    tomls = find_toml_files(models)
    wflow_to_id = pd.read_csv(csv_dir,index_col='wflow_id')
    wflow_id = list(wflow_to_id.index.values)
    print('models', models)
    staticgeoms_dir = find_staticgeoms(models, model_snippet)
    print(staticgeoms_dir)
    gauges_append = list_dirs_containing(staticgeoms_dir, 'gauges')
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

    if len(gauges_files) > 0 and check_locs == True: 
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
    
    return match_dict, models, tomls, wflow_id, staticgeoms_dir

match_dict, models, tomls, wflow_id, staticgeoms_dir = process_gauges(working_dir, csv_dir, model_snippet, check_locs=False)      

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

for key, item in dependency_dict.items():
    print(f'wflow_id {key} has dependencies (n={len(item)}) {item}')
    

