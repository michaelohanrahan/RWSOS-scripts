import xarray as xr

ds = xr.open_dataset(R"../routing_floodplain1d/output.nc")

start = '2014-04-01'
stop = '2014-07-01'

dsslice = ds.sel(time=slice(start, stop))

dsslice.to_netcdf('../routing_floodplain1d/output_short.nc')