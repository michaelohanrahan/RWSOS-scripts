#%%
import xarray as xr
import hydromt

def add_dim(ds, season):
    time = xr.DataArray([season], dims='time', name='season')
    ds_new = ds.expand_dims(time=time)
    return ds_new

seasonal_states_dir = R'p:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\run_hourly\cdo_output'
dir_outstates = R'p:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202301_floodplain1d\run_hourly\final_states'
seasons = ['DJF', 'MAM', 'JJA', 'SON']

#%%
for season in seasons:
    fn = f'{seasonal_states_dir}/cdo_seasmean_output_{season}.nc'
    # print(fn)

    ds = xr.open_dataset(fn)

    seas_mean = ds.mean(dim='time')

    seas_mean = add_dim(ds=seas_mean, season=season)

    seas_mean.to_netcdf(f'{dir_outstates}/{season}_mean.nc')


#%% Bugfix missings

for season in seasons:

    fn = f'{dir_outstates}/{season}_mean.nc'
    ref = xr.open_dataset(fn)

    fn = f'{seasonal_states_dir}/cdo_timpctl25_{season}.nc'
    ds = xr.open_dataset(fn)

    ds = ds.mean(dim='time')

    ds = add_dim(ds=ds, season=season)

    for var in ds.data_vars:
    # var = 'ssf'
        if var != 'ustorelayerdepth':

            new = ds[var].raster.interpolate_na(method='nearest')
            ds[var] = xr.where(ref[var].isnull(), ref[var], new)
        else:
            da = ds[var].copy()

            for layer in da.layer:
                new = da.sel(layer=layer).raster.interpolate_na(method='nearest')
                da.loc[dict(layer=layer)] = xr.where(ref[var].sel(layer=layer).isnull(), ref[var].sel(layer=layer), new)
                # ds[v
            ds[var] = da


    ds.to_netcdf(f'{dir_outstates}/{season}_dry_v2.nc')


#%% Bugfix missings
for season in seasons:

    fn = f'{dir_outstates}/{season}_mean.nc'
    ref = xr.open_dataset(fn)

    fn = f'{seasonal_states_dir}/cdo_timpctl75_{season}.nc'
    ds = xr.open_dataset(fn)

    ds = ds.mean(dim='time')

    ds = add_dim(ds=ds, season=season)

    for var in ds.data_vars:
    # var = 'ssf'
        if var != 'ustorelayerdepth':

            new = ds[var].raster.interpolate_na(method='nearest')
            ds[var] = xr.where(ref[var].isnull(), ref[var], new)
        else:
            da = ds[var].copy()

            for layer in da.layer:
                new = da.sel(layer=layer).raster.interpolate_na(method='nearest')
                da.loc[dict(layer=layer)] = xr.where(ref[var].sel(layer=layer).isnull(), ref[var].sel(layer=layer), new)
                # ds[v
            ds[var] = da


    ds.to_netcdf(f'{dir_outstates}/{season}_wet_v2.nc')
