import hydromt
import os
import shutil
from pathlib import Path
from hydromt_wflow import WflowModel


forcing_path = r"p:\11209265-grade2023\wflow\wflow_meuse_julia\forcing\forcing_Meuse_20050101_20180222_v2_wgs2_remapbil_semisstonn.nc"

# Initialize the model
mod = WflowModel(root = r"p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312",
                 config_fn = "wflow_sbm_hourly_get_instates.toml", mode='r')

mod.read_config()
mod.read_grid()

mod.set_root(r"p:\11209265-grade2023\wflow\RWSOS_Calibration\meuse\data\1-external")

#%%

mod.config['input']['path_static'] = 'staticmaps/staticmaps.nc'
