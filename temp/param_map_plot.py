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

mod.set_root(r"p:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_Meuse_Hourly_Base_20240305")

#%%
import tqdm 

params = [var for var in mod.grid.data_vars]

param_plot_dir = r"p:\11209265-grade2023\wflow\wflow_meuse_julia\param_plots"
os.makedirs(param_plot_dir, exist_ok=True)
cmap = plt.cm.get_cmap("viridis")
cmap.set_bad(color='none')

for p in tqdm.tqdm(params, desc="Plotting parameters", unit="parameter", total=len(params)):
    if not os.path.exists(os.path.join(param_plot_dir, f"{p}.png")):
        try:
            fig, ax = plt.subplots()
            mod.grid[p].plot(ax=ax, cmap=cmap, vmin=0)
            plt.title(p)
            plt.savefig(os.path.join(param_plot_dir, f"{p}.png"))
            plt.close()
        except Exception as e:
            print(f"Failed to plot {p}: {e}")