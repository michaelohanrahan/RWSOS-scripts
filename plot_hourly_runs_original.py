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
import numpy
import matplotlib.pyplot as plt

import sys
sys.path.append(r"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\_scripts")

from func_plot_signature import plot_signatures, plot_hydro
from file_inspection.func_io import read_filename_txt, read_lakefile
# from func_plot_signature import plot_hydro


model_runs = dict(
    s07 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\run_scale_river_n-0.7\run_river_n_07\output_scalar_07.csv", parse_dates=True, index_col=0),
    s08 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\run_scale_river_n-0.8\run_river_n_08\output_scalar_08.csv", parse_dates=True, index_col=0),
    s1 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\run_scale_river_n-1.0\run_river_n_10\output_scalar_10.csv", parse_dates=True, index_col=0),
    s09 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\run_scale_river_n-0.9\run_river_n_09\output_scalar_09.csv", parse_dates=True, index_col=0),
    s11 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312\run_scale_river_n-1.1\run_river_n_11\output_scalar_11.csv", parse_dates=True, index_col=0),
    # s12 = pd.read_csv(R"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\run_scale_river_n-1.2\output_scalar12.csv", parse_dates=True, index_col=0)   
)

idx_start = min([len(value) for item, value in model_runs.items()]) -1
Folder_plots = "../_plots/"
os.makedirs(Folder_plots, exist_ok=True)

toml_default_fn = r"P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202311\wflow_sbm_hourly_updated_states_L.toml"

gauges_maps = [
    # 'gauges_wflow-gauges-ahr',
    'gauges',
    'gauges_grdc',
    'gauges_hbv',
    'gauges_locs',
    'gauges_S01',
    'gauges_S01',
    'gauges_S02',
    'gauges_S03',
    'gauges_S04',
    'gauges_S05',
    'gauges_S06',
    'gauges_Sall',
]

# Get stations within model
mod_name = "wflow_sbm_run_hourly_scale_river_n"
root = r"P:\11209265-grade2023\wflow\wflow_meuse_julia\wflow_meuse_202312"
mod = WflowModel(root, config_fn=os.path.basename(mod_name), mode="r")


mod_stations = np.array([])

for gauge_map in gauges_maps:
    print(gauge_map)
    try:
        mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]["fid"].values)
    except KeyError:
        if gauge_map == "gauges":
            mod_stations = np.append(mod_stations, 709)
            
        elif gauge_map == "gauges_SOBEK":
            
            mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]["wflow_ID"].values)
            
        elif 1000.0 in gauge_map:
            pass
        else:
            raise ValueError()

for item, value in sobek_tight_match.items():
    if item in mod_stations:
        index = np.where(mod_stations == item)[0]
        if index.size > 0:
            mod_stations[index[0]] = value
            
mod_stations = mod_stations[~np.isnan(mod_stations)]
# mod_stations = mod.staticgeoms["gauges_wflow-gauges-ahr"].stations.values

# Get names of the locations
fn_root = r"P:\archivedprojects\11205237-grade\wflow\wflow_rhine_julia\measurements\vanBart"
# Get names of the locations
fn_names = os.path.join(fn_root, "discharge_obs_hr_appended_station_list.csv")
df_locs = pd.read_csv(fn_names, index_col=0, encoding= 'unicode_escape')
df_locs = df_locs.loc[mod_stations,:]
df_locs["names"] = df_locs.station_names.str.split("'", expand=True)[1]
# Convert to dictionary
stations_dict = df_locs.names.to_dict()

# Read observations
# fn_obs = R"p:\11205237-grade\wflow\wflow_rhine_julia\measurements\vanBart\discharge_obs_hr_appended.nc"
# ds_obs = xr.open_dataset(fn_obs)
df_obs = pd.read_csv(r'..\obs_2015_hourly.csv', parse_dates=True, index_col=0)
# time_index = np.unique(df_obs_tmp.index.get_level_values("time"))

# Convert to DataFrame
# df_obs = pd.DataFrame(index=pd.DatetimeIndex(time_index))
# for station in stations_dict.keys():
#     tmp = df_obs_tmp.iloc[df_obs_tmp.index.get_level_values("stations") == station]["Qm"]
#     tmp.index = tmp.index.droplevel("stations")
#     df_obs[f"Q_{station}"] = tmp


### prepare dataset to make plots
colors = [
    '#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
    '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

runs_dict = {}

for key, df in model_runs.items():
    # print(key)
    runs_dict[key] = df

plot_colors = colors[:len(runs_dict)]

# List of sobek gauges co-located with observations, based on matching name strings
ls = [] 

for col in df_obs.columns:
    station = float(col.split('_')[1])
    for key, item in sobek_tight_match.items():
        if station == item:
            ls.append(station)
print(ls)
    

### #make dataset

variables = ['Q']
runs = ['Obs.', *model_runs.keys()]
# rng = pd.date_range('1979-01-01', '2019-12-31')
# rng = df_obs.index
rng = pd.date_range(
    max(*[runs_dict[key].index[0] for key in runs_dict.keys()],
        df_obs.index[0]),
    min(*[runs_dict[key].index[-1] for key in runs_dict.keys()],
        df_obs.index[-1]),
    freq="H"
)



obs_keys = []

for key in df_obs.columns:
    if int(key.split("_")[1]) in mod_stations:
        obs_keys.append(key)

S = np.zeros((len(rng), len(obs_keys), len(runs)))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}


ds = xr.Dataset(
        data_vars=h,
        coords={'time': rng,
                'stations': [station.split("_")[1] for station in obs_keys],
                'runs': runs})

ds = ds * np.nan

# fill dataset with model and observed data
ds['Q'].loc[dict(runs = 'Obs.')] = df_obs.loc[rng, obs_keys]

for key, item in runs_dict.items():

    # ds['Q'].loc[dict(runs = key)] = item[['Q_' + sub for sub in list(map(str,list(stations_dict.keys())))]].loc[rng]
    for sub in list(map(str,list(stations_dict.keys()))):
        
        try:
            
            if sub == "709":
                ds['Q'].loc[dict(runs = key, stations=sub)] = item.loc[rng, "Q_1"]
            
            elif int(sub) in ls:
                sub_gauge = next((k for k, v in sobek_tight_match.items() if v == int(sub)), None)
                ds['Q'].loc[dict(runs = key, stations=sub)] = item.loc[rng, f"Q_sobek_{sub_gauge}"]
                print(sub, key, sub_gauge, 'appended')
            
            else:
                column = next((col for col in item.columns if col.endswith(str(sub))), None)
                if column:
                    ds['Q'].loc[dict(runs = key, stations=int(sub))] = item.loc[rng, column]

        except KeyError as e:
            print(e)
            pass
            
# ds['Q'].loc[dict(runs = label_01)] = run01[['Q_' + sub for sub in list(map(str,list(stations_dic.values())))]][:len(rng)]

#%%
translate = {
    's07': 'scale: 0.7',
    's08': 'scale: 0.8',
    's09': 'scale: 0.9',
    's11': 'scale: 1.1',
    's12': 'scale: 1.2',
    'n072': 'Default, fl1d',
    'Obs.': 'Observed',
}

color_dict = {
    's07': '#377eb8',  # Blue
    's08': '#ff7f00',  # Orange
    's09': '#4daf4a',  # Green
    's11': '#f781bf',  # Pink
    's12': '#a65628',  # Brown
    'n072': '#984ea3',  # Purple
    'Obs.': '#999999',  # Grey
}

# translate = {
#     'n07': 'n: 0.07',
#     'n08': 'n: 0.08',
#     'n09': 'n: 0.09',
#     # 's11': 'scale: 1.1',
#     # 's12': 'scale: 1.2',
#     'n072': 'Default, fl1d',
#     'Obs.': 'Observed',
# }

# color_dict = {
#     'n05': '#377eb8',  # Blue
#     'n08': '#ff7f00',  # Orange
#     'n09': '#4daf4a',  # Green
#     # 's11': '#f781bf',  # Pink
#     # 's12': '#a65628',  # Brown
#     'n072': '#984ea3',  # Purple
#     'Obs.': '#999999',  # Grey
# }

# Define the font properties
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }
# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(script_dir)

def plot_ts(start, end, savefig=False):

    for station_id, station_name in stations_dict.items():
        
        print('sid:', station_id, station_name)
        
        s = np.array(list(sobek_tight_match.values()))
        s = s[~np.isnan(s)]
        
        if station_id in s:
            # print('in mod_stations', station_id)
            try:
                fig, ax = plt.subplots(figsize=(12,6.18))
                ax.set_title(f'Name: {station_name}, id: {station_id}', fontdict=font)
                
                obs_max_time = ds.sel(time=slice(start, end), runs='Obs.', stations=str(station_id)).Q.idxmax().values

                for run in ds['runs'].values:
                    
                    if str(run) in ['s07', 's08','n072', 'Obs.']:
                    # if str(run) in ['n05', 'n09','n072', 'Obs.']:
                        # Select the specific run and station from ds
                        subset = ds.sel(time=slice(start, end),
                                        runs=run, 
                                        stations=str(station_id)).dropna(dim='time')  
                        print('subset', subset)
                        # Get the time index of the maximum value in this run
                        run_max_time = subset.sel(time=slice(obs_max_time - pd.Timedelta(hours=72), obs_max_time + pd.Timedelta(hours=72))).Q.idxmax().values
                        # Calculate the difference in peak timing
                        dt = run_max_time - obs_max_time
                        dt_hours = dt.astype('timedelta64[h]').item().total_seconds() / 3600
                        # Set the y-axis label
                        ax.set_ylabel('Discharge ($m^3s^{-1}$)')

                        # Set the font properties for the y-axis labels
                        ax.tick_params(axis='y', labelsize=font['size'])

                        # Set the x-axis label
                        ax.set_xlabel('Date (hourly timestep)')

                        # Set the font properties for the x-axis labels
                        ax.tick_params(axis='x', labelsize=font['size'])
                        
                        if run == 'Obs.':
                            label = f'{translate[run]}'
                        else:
                            label = f'{translate[run]}, model lag = {dt_hours:.2f} hours'
                        
                        # Plot the subset for this run
                        ax.plot(subset.time, subset.Q, label=label, c=color_dict[str(run)])
                    else:
                        continue
                # print('success', station_id)
                # Add the legend outside of the loop
                ax.legend()
                plt.tight_layout()
                # Set the x-axis limits to the time slice
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
                ax.set_xlim([start - pd.Timedelta(hours=48), end + pd.Timedelta(hours=48)])
                ax.grid()
                
                
                if savefig == True:
                    # Create the directory if it doesn't exist
                    print('saving...')
                    plots_dir = os.path.join(parent_dir, '_plots')
                    os.makedirs(plots_dir, exist_ok=True)
                    # Save the figure
                    fig.savefig(os.path.join(f'../_plots/timeseries_{station_name}_{station_id}_{start.month, start.day}_{end.month,end.day}_n.png'), dpi=300)
                    # print(f'saved to {timeseries_{station_name}_{station_id}_{start.month, start.day}_{end.month,end.day}.png}')
                else:
                    pass
                    
            except Exception as e:
                print('fail', station_id)
                print(e)
                pass
        else:
            print('not in mod_stations', station_id)


plot_ts('2015-01-01', '2015-12-29', savefig=False)
plot_ts('2015-04-15', '2015-06-01', savefig=True)




for station_id, station_name in stations_dict.items():
        
        # print('sid:', station_id, station_name)
        
        s = np.array(list(sobek_tight_match.values()))
        s = s[~np.isnan(s)]
        try:
            if station_id in s:

                # dsq = ds.sel(stations = str(station_id))#.sel(time = slice('1980-01-01', None))#.dropna(dim='time')

                # # dsq.sel(time=dsq.time[~dsq.Q.sel(runs="Obs.").isnull()].values)

                # #plot hydro
                # # plot_hydro(dsq, label_00, label_01, color_00, color_01, Folder_plots, station_name)
                # start_long, end_long = '2015-01-01', '2015-12-20'
                # start_1, end_1 = '2015-03-01', '2015-04-30'
                # start_2, end_2 = '2015-06-01', '2015-08-31'
                # start_3, end_3 = '2015-10-01', '2015-12-20'
                

                # plot_hydro(
                #     dsq=dsq,
                #     start_long=start_long, end_long=end_long,
                #     start_1=start_1, end_1=end_1, start_2=start_2, end_2=end_2,
                #     start_3=start_3, end_3=end_3, labels=list(runs_dict.keys()),
                #     colors=plot_colors, Folder_out=Folder_plots, station_name=station_name,
                #     station_id=station_id,
                #     save=True,
                # )

                #make plot using function
                #dropna for signature calculations.
                # dsq = ds.sel(stations = str(station_id))#.sel(time = slice('1980-01-01', None)).dropna(dim='time')
                
                # plot_signatures(
                #     dsq=dsq, labels=list(runs_dict.keys()), colors=plot_colors,
                #     Folder_out=Folder_plots, station_name=station_name, station_id=station_id, save=True,
                # )
        except Exception as e:
            print(e)
            pass
# #%%
# for station_name, station_id in tqdm(stations_dict.items()):
#     try:
#         idx = station_id.index("\\")
#         station_id = station_id[:idx] + "x" + station_id[idx+4:]
#     except:
#         pass

#     dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None))#.dropna(dim='time')

#     # dsq.sel(time=dsq.time[~dsq.Q.sel(runs="Obs.").isnull()].values)

#     #plot hydro
#     # plot_hydro(dsq, label_00, label_01, color_00, color_01, Folder_plots, station_name)
#     start_long, end_long = '1991-01-01', '2015-12-31'
#     start_1, end_1 = '2014-01-01', '2014-12-31'
#     start_2, end_2 = '2003-01-01', '2003-12-31'
#     start_3, end_3 = '2010-01-01', '2010-12-31'
#     # plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2,
#     #            start_3, end_3, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
#     #            save=True)

#     plot_hydro(
#         dsq=dsq,
#         start_long=start_long, end_long=end_long,
#         start_1=start_1, end_1=end_1, start_2=start_2, end_2=end_2,
#         start_3=start_3, end_3=end_3, labels=list(runs_dict.keys()),
#         colors=plot_colors, Folder_out=Folder_plots, station_name=station_id,
#         station_id=station_name,
#         save=True,
#     )

#     #make plot using function
#     #dropna for signature calculations.
#     dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None)).dropna(dim='time')
#     # plot_signatures(
#     #     dsq, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
#     #     save=True)
#     plot_signatures(
#         dsq=dsq, labels=list(runs_dict.keys()), colors=plot_colors,
#         Folder_out=Folder_plots, station_name=station_id, station_id=station_name, save=True,
#     )
