from hydromt_wflow import WflowModel
from model_building.update_gauges import main as update_gauges
from file_methods.postprocess import find_model_dirs, find_toml_files
import os

working_dir = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_per_catchment_N"
models = find_model_dirs(working_dir, 'fl1d')
tomls = find_toml_files(models)

print(models)
print(tomls)

# Initialize the model
model = WflowModel(root=models[0], config_fn=tomls[1], mode='r')

if 'staticgeoms' in os.listdir(models[0]):
    static_geoms = os.path.join(models[0], 'staticgeoms')
    files_list = os.listdir(static_geoms)
    
    gauges_append = []
    
    for file in files_list:
        if 'gauges' in file:
            gauges_append.append(file)

gauges_files = [os.path.join(static_geoms, file) for file in gauges_append]


# Load the default configuration
model.read()
model.config


# #load the hourly gauges
# new_gdf = gpd.read_file(hourly_gauges_path)

# update_gauges