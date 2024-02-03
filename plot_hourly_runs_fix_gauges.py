#%%
#env: hydromt

import hydromt
from hydromt_wflow import WflowModel
import xarray as xr
import numpy as np
import os
import xarray as xr
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


##########/==-- Local Func --==/#############
import sys
sys.path.append(r"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\_scripts")

from func_plot_signature import plot_signatures, plot_hydro
from file_inspection.func_io import read_filename_txt, read_lakefile
# from func_plot_signature import plot_hydro


model_runs = dict(
    s07 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-0.7\output_scalar07.csv", parse_dates=True, index_col=0),
    s08 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-0.8\output_scalar08.csv", parse_dates=True, index_col=0),
    n072 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_mannings_n-0.072\output_0612_initialcond_0072.csv", parse_dates=True, index_col=0),
    s09 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-0.9\output_scalar09.csv", parse_dates=True, index_col=0),
    s11 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-1.1\output_scalar11.csv", parse_dates=True, index_col=0),
    s12 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-1.2\output_scalar12.csv", parse_dates=True, index_col=0)   
)


idx_start = 365
Folder_plots = "../_figures_hourly/"
toml_default_fn = r"..\wflow_sbm_hourly_updated_states_L.toml"


# Get stations within model
mod_name = "wflow_sbm_hourly_csv"
root = r"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311"
mod = WflowModel(root, config_fn=os.path.basename(mod_name), mode="r")

gauges_maps = [key for key in mod.staticgeoms.keys() if key.startswith('gauges')]

#Removing some groups because this relies on some individual fid values whicha are not present in the removed shapes' colums
# gauges_maps.remove('gauges_fews') #Index(['WFLOW_ID', 'fews id location', 'fews name location', 'geometry'], dtype='object')
# gauges_maps.remove('gauges_grdc') #Index(['grdc_no', 'wmo_reg', 'sub_reg', 'river', 'station', 'country', 'area','altitude', 'd_start', 'd_end', 'd_yrs', 'd_miss', 'm_start', 'm_end','m_yrs', 'm_miss', 't_start', 't_end', 't_yrs', 'lta_discharge','r_volume_yr', 'r_height_yr', 'geometry'],dtype='object')
# gauges_maps.remove('gauges_hbv_mainsub')   # Index(['gauge_ID', 'name', 'geometry'], dtype='object')
# gauges_maps.remove('gauges_SOBEK')   #Index(['wflow_ID', 'id', 'name', 'shortName', 'x', 'y', 'z', 'geometry'], dtype='object')
# gauges_maps.remove('gauges_wflow-gauges-ahr')   #fid not in station list


mod_stations = np.array([])

for gauge_map in gauges_maps:
    try:
        # print(gauge_map)
        if gauge_map  == 'gauges_fews':
            pass
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]['WFLOW_ID'].values)
        elif gauge_map == 'gauges_SOBEK':
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]['wflow_ID'].values)
        elif gauge_map == 'gauges_grdc':
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]['grdc_no'].values)
        elif gauge_map == 'gauges_hbv_mainsub':
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]['gauge_ID'].values)
        elif gauge_map == "gauges":
            #1 is the basin outlet but 709 is the Lobith observation station, rename for later
            mod_stations = np.append(mod_stations, 709)
        else:
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]['fid'].values)
    except KeyError as e:
        print(e)

obs_root = r"P:\archivedprojects\11205237-grade\wflow\wflow_rhine_julia\measurements\vanBart"
# obs = r"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\obs_2015_hourly.csv"

# Get names of the locations
fn_names = os.path.join(obs_root, "discharge_obs_hr_appended_station_list.csv")
df_locs = pd.read_csv(fn_names, index_col=0, encoding= 'unicode_escape')

#stations that dont exist in the station:name dataframe
removal = [station for station in mod_stations if station not in df_locs.index]

print('stations not found in station list', removal)

# Remove instances from mod_stations
re_mod_stations = [station for station in mod_stations if station not in removal]

df_locs = df_locs.loc[re_mod_stations,:]
df_locs["names"] = df_locs.station_names.str.split("'", expand=True)[1]

# Convert to dictionary
stations_dict = df_locs.names.to_dict()

print("start section: Observation Conversion")

df_2015 = pd.read_csv(r'..\obs_2015_hourly.csv', parse_dates=True, index_col=0)
print('loaded observations to memory')

if df_2015.index.name != "time":
    df_obs_tmp = df_2015.rename_axis("time")
else:
    df_obs_tmp = df_2015


df_obs_tmp.columns = [col.split('_')[-1] for col in df_obs_tmp.columns]
df_obs = df_obs_tmp


### prepare dataset to make plots
colors = [
    '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
    '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928'
    ]


def xr_to_q_df(xr_ds):
    """Convert xarray Dataset to DataFrame"""
    data = {}
    for i, var in enumerate(xr_ds.data_vars):
        if var.startswith('Q'):
            da = xr_ds[var]
            # print(da)
            
            time, gauge_ind  = da.dims
            # print(time, gauge_ind)
            
            time_vals = da[time].values
            gauge_vals  = da[gauge_ind].values
            
            for gauge in gauge_vals:
                
                vals = da.sel({gauge_ind:gauge}).values
                nancount = np.count_nonzero(np.isnan(vals))
                length = len(vals)
                
                # print(data['Q_'+str(gauge)])
                # print(nancount, length)
                
                if length > 0 and nancount/length < 0.8:
                    if gauge not in data:
                        data['Q_'+str(gauge)] = []
                    # print('pass', gauge, len(vals))
                    data['Q_'+str(gauge)].extend(da.sel({gauge_ind:gauge}).values)
                # plt.plot(np.arange(0, len(data['Q_'+str(gauge)]), data['Q_'+str(gauge)]))
                
    # print(data)
    df = pd.DataFrame(data, index='time')
    return df

# Initialize an empty dictionary to store the DataFrames
runs_dict = {}

# Loop through each item in model_runs
for key, item in model_runs.items():
    if isinstance(item, xr.Dataset):
        # Convert xarray Dataset to DataFrame
        df = xr_to_q_df(item)
    else:
        # Assign the DataFrame directly
        df = item

    # Remap column names
    df.columns = [col.split('_')[-1] for col in df.columns]

    # Replace column name "Q_1" with "Q_709"
    if '1' in df.columns:
        df.rename(columns={'1': '709'}, inplace=True)

    # Store in runs_dict
    runs_dict[key] = df

plot_colors = colors[:len(runs_dict)]

### #make dataset
variables = ['Q']
runs = ['Obs.', *model_runs.keys()]
# rng = pd.date_range('1979-01-01', '2019-12-31')
rng = pd.date_range('2015-01-01', '2015-12-31', freq='H')

# rng = df_obs.index
# rng = pd.date_range(
#     max(*[runs_dict[key].index[idx_start] for key in runs_dict.keys()],
#         df_obs.index[0]),
#     min(*[runs_dict[key].index[-1] for key in runs_dict.keys()],
#         df_obs.index[-1]),
#     freq="H"
# )

intersect = set(df_obs.columns) & set(runs_dict['s08'].columns)
#filter the stations_dict to only include the common keys
new_stations_dict = {int(key):stations_dict[int(key)] for key in intersect}


S = np.zeros((len(rng), len(new_stations_dict.values()), len(runs)))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h,
        coords={'time': rng,
                'stations': list(new_stations_dict.keys()),
                'runs': runs})
ds = ds * np.nan

# fill dataset with model and observed data
ds['Q'].loc[dict(runs = 'Obs.')] = df_obs.loc[rng, list(intersect)]

for key, item in runs_dict.items():
    print(key)
    
    for sub in list(map(str,list(new_stations_dict.keys()))):
        
        # Create a new DataFrame that has the same index as rng
        item_padded = pd.DataFrame(index=rng)
        
        # Assign the values from item to item_padded, aligning on the index
        item_padded[sub] = item[sub]
        
        # Assign item_padded to ds['Q']
        ds['Q'].loc[dict(runs = key, stations=int(sub))] = item_padded[sub]
# ds['Q'].loc[dict(runs = label_01)] = run01[['Q_' + sub for sub in list(map(str,list(stations_dic.values())))]][:len(rng)]

translate = {
    's07': 'scale: 0.7
    's08': 'scale: 0.8',
    's09': 'scale: 0.9',
    's11': 'scale: 1.1',
    's12': 'scale: 1.2',
    'n072': 'Default',
    'Obs.': 'Observed',
}

#%%
for station_name, station_id in tqdm(stations_dict.items()):
    try:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title(f'{station_name}, {station_id}')

        for run in ds['runs'].values:
            print(run)
            if str(run) in ['s08','s07','s09','n072', 's11', 'Obs.']:
                # Select time starting from '2015-01-03' for all runs and stations
                ds_time_selected = ds['Q'].sel()
                # Select the specific run and station from ds_time_selected
                subset = ds.sel(time=slice('2015-03-03', '2015-04-24'),
                                runs=run, 
                                stations=station_name).dropna(dim='time')  
                
                # Plot the subset for this run
                subset.Q.plot(ax=ax, label=translate[run])
            else:
                continue
        # Add the legend outside of the loop
        ax.legend()
    except:
        print('fail', station_id)
        pass


for station_name, station_id in tqdm(stations_dict.items()):
    try:
        idx = station_id.index("\\")
        station_id = station_id[:idx] + "x" + station_id[idx+4:]
    except:
        pass

    dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None))#.dropna(dim='time')

    # dsq.sel(time=dsq.time[~dsq.Q.sel(runs="Obs.").isnull()].values)

    #plot hydro
    # plot_hydro(dsq, label_00, label_01, color_00, color_01, Folder_plots, station_name)
    start_long, end_long = '2015-01-01', '2015-12-20'
    start_1, end_1 = '2015-03-01', '2015-04-30'
    start_2, end_2 = '2015-06-01', '2015-08-31'
    start_3, end_3 = '2015-10-01', '2015-12-20'
    # plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2,
    #            start_3, end_3, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
    #            save=True)

    plot_hydro(
        dsq=dsq,
        start_long=start_long, end_long=end_long,
        start_1=start_1, end_1=end_1, start_2=start_2, end_2=end_2,
        start_3=start_3, end_3=end_3, labels=list(runs_dict.keys()),
        colors=plot_colors, Folder_out=Folder_plots, station_name=station_id,
        station_id=station_name,
        save=True,
    )

    #make plot using function
    #dropna for signature calculations.
    dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None)).dropna(dim='time')
    # plot_signatures(
    #     dsq, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
    #     save=True)
    plot_signatures(
        dsq=dsq, labels=list(runs_dict.keys()), colors=plot_colors,
        Folder_out=Folder_plots, station_name=station_id, station_id=station_name, save=True,
    )