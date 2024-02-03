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

import sys
sys.path.append(R"c:\Users\buitink\OneDrive - Stichting Deltares\Documents\GitHub\Deltares_scripts")
sys.path.append(R"C:\Users\buitink\Documents\GitHub\Deltares_scripts")

from func_plot_signature import plot_signatures, plot_hydro
from file_inspection.func_io import read_filename_txt, read_lakefile
# from func_plot_signature import plot_hydro

model_runs = dict(
    de_bruin = R'P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\routing_fld1d_2014_latest_20230216_rivWD\output.csv',
    makkink = R'p:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\routing_fld1d_2014_latest_20230216_rivWD_mak\output.csv',
    HBV = R"p:\11205237-grade\climate_data\KNMI_scenarios_preparation\CMIP6_datasets_prepared_forWflow\2022-11-hbv_runs\output\Rhine\020151231_HBV_RHINE.csv"
)

idx_start = 365
basel_cor = True
Folder_plots = "../_figures_PET/"


toml_default_fn = "../routing_runs_floodplain1d_2014_latest_20230214.toml"

gauges_maps = [
    # 'gauges_wflow-gauges-ahr',
    'gauges',
    'gauges_wflow-gauges-extra',
    'gauges_wflow-gauges-mainsub',
    'gauges_wflow-gauges-rhineriv'
]

# Get stations within model
root = os.path.dirname(toml_default_fn)
mod = WflowModel(root, config_fn=os.path.basename(toml_default_fn), mode="r")
# mod_stations = mod.staticgeoms["gauges_gauges-obs_hr"].stations.values

mod_stations = np.array([])
for gauge_map in gauges_maps:
    try:
        mod_stations = np.append(mod_stations, mod.staticgeoms[gauge_map]["fid"].values)
    except KeyError:
        if gauge_map == "gauges":
            mod_stations = np.append(mod_stations, 709)
        else:
            raise ValueError()


# mod_stations = mod.staticgeoms["gauges_wflow-gauges-ahr"].stations.values

# Get names of the locations
fn_names = R"p:\11205237-grade\wflow\wflow_rhine_julia\measurements\vanBart\discharge_obs_hr_appended_station_list.csv"
df_locs = pd.read_csv(fn_names, index_col=0, encoding= 'unicode_escape')
df_locs = df_locs.loc[mod_stations,:]
df_locs["names"] = df_locs.station_names.str.split("'", expand=True)[1]
# Convert to dictionary
stations_dict = df_locs.names.to_dict()

# Read observations
fn_obs = R"p:\11205237-grade\wflow\wflow_rhine_julia\measurements\vanBart\discharge_obs_hr_appended.nc"
ds_obs = xr.open_dataset(fn_obs)
df_obs_tmp = ds_obs.Qm.resample(time="D").mean().to_dataframe()
time_index = np.unique(df_obs_tmp.index.get_level_values("time"))

# Convert to DataFrame
df_obs = pd.DataFrame(index=pd.DatetimeIndex(time_index))
for station in stations_dict.keys():
    tmp = df_obs_tmp.iloc[df_obs_tmp.index.get_level_values("stations") == station]["Qm"]
    tmp.index = tmp.index.droplevel("stations")
    df_obs[f"Q_{station}"] = tmp


### prepare dataset to make plots
colors = [
    '#a6cee3','#1f78b4','orange', '#b2df8a','#33a02c','orange', '#fb9a99','#e31a1c',
    '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

runs_dict = {}

for key in model_runs.keys():
    if key != "HBV":
        runs_dict[key] = pd.read_csv(model_runs[key], index_col=0, header=0, parse_dates=True)
    else:
        fn = model_runs[key]
        # df = pd.read_csv(fn, delim_whitespace=True, skiprows=[0], usecols = [0,1,2,3,4,11])
        # df.columns = ['YEAR','MONTH','DAY','MIN','HOUR','HBV']
        # df.index = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
        # df = df.drop(['YEAR','MONTH','DAY','MIN','HOUR'], axis=1)
        # runs_dict[key] = df

        df = pd.read_csv(fn, delim_whitespace=True, skiprows=[0])
        df = df.reset_index()

        df.columns = [
            'year', 'month', 'day', 'hour', 'minute',
            'HBV_Kaub', 'HBV_MidRhine2', 'HBV_Worms', 'HBV_Andernach', 'HBV_Koeln', 'HBV_Wesel', 'HBV_Lobith']

        df.index = pd.to_datetime(df[['year','month','day', 'hour', 'minute']])
        df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)
        df = df.replace(-999, np.nan)
        runs_dict[key] = df


# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.subplots(1)
# for run, df in runs_dict.items():
#     df['Q_1'].plot(ax=ax, label=run)
# ax.legend()

plot_colors = colors[:len(runs_dict)]

# start = 0
# stop = 0
# for key, df in runs_dict.items():
#     if start == 0:
#         start = df.index[0]
#         stop = df.index[-1]
#     else:
#         start = max(start, df.index[0])
#         stop = min(stop, df.index[-1])

### #make dataset

variables = ['Q']
runs = ['Obs.', *model_runs.keys()]
# rng = pd.date_range('1979-01-01', '2019-12-31')
# rng = df_obs.index
rng = pd.date_range(
    # start, stop
    max(runs_dict[list(runs_dict.keys())[0]].index[idx_start], df_obs.index[0]),
    min(runs_dict[list(runs_dict.keys())[0]].index[-1], df_obs.index[-1])
)

S = np.zeros((len(rng), len(stations_dict.values()), len(runs)))
v = (('time', 'stations', 'runs'), S)
h = {k:v for k in variables}

ds = xr.Dataset(
        data_vars=h,
        coords={'time': rng,
                'stations': list(stations_dict.keys()),
                'runs': runs})
ds = ds * np.nan


# fill dataset with model and observed data
ds['Q'].loc[dict(runs = 'Obs.')] = df_obs.loc[rng]
for key, item in runs_dict.items():
    if key == "HBV":
        hbvstart = max(rng[0], item.index[0])
        hbvstop = min(rng[-1], item.index[-1])
        ds['Q'].loc[dict(runs = key, stations=709, time=slice(hbvstart, hbvstop))] = item.loc[hbvstart:hbvstop , "HBV_Lobith"]
        ds['Q'].loc[dict(runs = key, stations=695, time=slice(hbvstart, hbvstop))] = item.loc[hbvstart:hbvstop , "HBV_Kaub"]
        ds['Q'].loc[dict(runs = key, stations=696, time=slice(hbvstart, hbvstop))] = item.loc[hbvstart:hbvstop , "HBV_Andernach"]
        ds['Q'].loc[dict(runs = key, stations=698, time=slice(hbvstart, hbvstop))] = item.loc[hbvstart:hbvstop , "HBV_Koeln"]
    else:
        # ds['Q'].loc[dict(runs = key)] = item[['Q_' + sub for sub in list(map(str,list(stations_dict.keys())))]].loc[rng]
        for sub in list(map(str,list(stations_dict.keys()))):
            if sub == "709":
                ds['Q'].loc[dict(runs = key, stations=int(sub))] = item.loc[rng, "Q_1"]
            else:
                ds['Q'].loc[dict(runs = key, stations=int(sub))] = item.loc[rng, f"Q_{sub}"]

# ds['Q'].loc[dict(runs = label_01)] = run01[['Q_' + sub for sub in list(map(str,list(stations_dic.values())))]][:len(rng)]



#%%
for station_name, station_id in tqdm(stations_dict.items()):
    try:
        idx = station_id.index("\\")
        station_id = station_id[:idx] + "x" + station_id[idx+4:]
    except:
        pass

    dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None))#.dropna(dim='time')

    if basel_cor and "Lobith" in station_id:
        # print(True)

        df = pd.read_csv(R'p:\11205237-grade\wflow\wflow_rhine_julia\measurements\6935051_Q_Day.Cmd.txt', sep=';', skiprows=35, encoding='ISO-8859-1', index_col=0, parse_dates=True)
        basel_obs = df[' Value']
        basel_obs = basel_obs[basel_obs.index.isin(set(ds.time.values))]
        basel_obs.index.name = 'time'
        basel_obs = xr.DataArray.from_series(basel_obs)
        # basel_obs = basel_runs.Q.sel(runs="Obs.")


        basel_runs = ds.sel(stations=705)
        basel_diff = basel_obs - basel_runs.Q#.sel(time=basel_obs.index)

        try: basel_diff.loc[dict(runs="HBV_Lobith")] = 0.0
        except: pass

        try: basel_diff.loc[dict(runs="HBV")] = 0.0
        except: pass


        basel_shift = basel_diff.shift(time=4)
        # basel_shift = basel_shift.drop_sel(runs=["HBV"])

        dsq = dsq + basel_shift

        station_name = f"{station_name}-BaselCor"

    # if "Lobith" not in station_id:
    #     dsq = dsq.sel(runs = np.delete(dsq.runs.values, np.where(dsq.runs.values == "HBV_Lobith")))
    # if "Kaub" not in station_id:
    #     dsq = dsq.sel(runs = np.delete(dsq.runs.values, np.where(dsq.runs.values == "HBV_Kaub")))
    # if "Andernach" not in station_id:
    #     dsq = dsq.sel(runs = np.delete(dsq.runs.values, np.where(dsq.runs.values == "HBV_Andernach")))
    # if "Koeln" not in station_id:
    #     dsq = dsq.sel(runs = np.delete(dsq.runs.values, np.where(dsq.runs.values == "HBV_Koeln")))

    if station_id not in ['Lobith', 'Kaub' ,'Andernach' , 'Koeln']:
        dsq = dsq.sel(runs = np.delete(dsq.runs.values, np.where(dsq.runs.values == "HBV")))

    labels = np.delete(dsq.runs.values, np.where(dsq.runs.values == "Obs."))

    # dsq.sel(time=dsq.time[~dsq.Q.sel(runs="Obs.").isnull()].values)

    #plot hydro
    # plot_hydro(dsq, label_00, label_01, color_00, color_01, Folder_plots, station_name)
    start_long, end_long = '1991-01-01', '2015-12-31'
    start_1, end_1 = '2014-01-01', '2014-12-31'
    start_2, end_2 = '2003-01-01', '2003-12-31'
    start_3, end_3 = '2010-01-01', '2010-12-31'

    # start_1, end_1 = '1991-01-01', None#, '1991-12-31'
    # start_2, end_2 = '1991-01-01', None#'1991-12-31'
    # start_3, end_3 = '1991-01-01', None#'1991-12-31'

    # plot_hydro(dsq, start_long, end_long, start_1, end_1, start_2, end_2,
    #            start_3, end_3, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
    #            save=True)

    plot_hydro(
        dsq=dsq,
        start_long=start_long, end_long=end_long,
        start_1=start_1, end_1=end_1, start_2=start_2, end_2=end_2,
        start_3=start_3, end_3=end_3, labels=labels,#list(runs_dict.keys()),
        colors=plot_colors, Folder_out=Folder_plots, station_name=station_id,
        station_id=station_name,
        save=True,
    )

    #make plot using function
    #dropna for signature calculations.
    # dsq = ds.sel(stations = station_name)#.sel(time = slice('1980-01-01', None)).dropna(dim='time')
    # plot_signatures(
    #     dsq, list(runs_dict.keys()), plot_colors, Folder_plots, station_name,
    #     save=True)
    try:
        plot_signatures(
            dsq=dsq, labels=labels, #list(runs_dict.keys()),
            colors=plot_colors,
            Folder_out=Folder_plots, station_name=station_id, save=True,
            station_id=station_name,
        )
    except:
        pass
