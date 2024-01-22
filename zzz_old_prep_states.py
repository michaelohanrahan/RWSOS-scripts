# import hydromt
# import xarray as xr
# from dask.diagnostics.progress import ProgressBar

# outmaps_fn = R'p:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\run_hourly\output.nc'

# dir_outstates = R'p:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\run_hourly\states_converted'

# chunksize = 1000

# chunks = dict(
#     time=chunksize,
#     # lat=50,
#     # lon=50
# )

# fn_out_gr = Rf'{dir_outstates}/outmaps_seasonal.nc'

# #sbm floodplain 1d
# outmaps = xr.open_dataset(outmaps_fn, chunks=chunks)

# season = outmaps.resample(time="QS-DEC").mean()

# chunksizes = (1, season.raster.ycoords.size, season.raster.xcoords.size, 4)

# encoding = {}
# for v in season.data_vars.keys():
#     if len(season[v].coords) == 5:
#         chunksizes = (1, 4, season.raster.ycoords.size, season.raster.xcoords.size)
#     else:
#         chunksizes = (1, season.raster.ycoords.size, season.raster.xcoords.size)
#     encoding[v] = {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}


# # encoding = {
# #     v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
# #     for v in season.data_vars.keys()
# #             }

# delayed_obj = season.to_netcdf(
#                     fn_out_gr, encoding=encoding, mode="w", compute=False
#                 )

# with ProgressBar():
#     delayed_obj.compute()

# # outmaps_monthly_mean = outmaps.groupby("time.season").mean()
# # with ProgressBar():
# #     outmaps_monthly_mean.to_netcdf(Rf'{dir_outstates}/outmaps_season_mean.nc')


# # # del outmaps
# # del outmaps_monthly_mean

# # # outmaps = xr.open_dataset(outmaps_fn, chunks={"time":chunksize})
# # outmaps_monthly_mean = outmaps.groupby("time.season").quantile(0.25)
# # with ProgressBar():
# #     outmaps_monthly_mean.to_netcdf(Rf'{dir_outstates}/outmaps_season_q25.nc')

# # # del outmaps
# # del outmaps_monthly_mean

# # # outmaps = xr.open_dataset(outmaps_fn, chunks={"time":chunksize})
# # outmaps_monthly_mean = outmaps.groupby("time.season").quantile(0.75)
# # with ProgressBar():
# #     outmaps_monthly_mean.to_netcdf(Rf'{dir_outstates}/outmaps_season_q75.nc')