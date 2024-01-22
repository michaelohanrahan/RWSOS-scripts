import xarray as xr
import pandas as pd

filepaths = [
    r"P:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\data_hourly_hydroeau\FR-Hydro-hourly-2005_2022.nc",
    r"P:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_xr_gds_nl.nc",
    r"P:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_xr_gds.nc",
    r"P:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_hourly_belgian_catch.nc",
    r"P:\archivedprojects\11205237-grade\wflow\wflow_meuse_julia\_obs\qobs_xr.nc"
]

for fp in filepaths:
    ds = xr.open_dataset(fp)
    print(f'\n{fp}\n{ds.variables}\n\n')
    print(f'\n{fp}\nstart: {ds.time.min().values}\nend: {ds.time.max().values}\n\n')
    ts = pd.Series(ds.time.values)
    freq = pd.infer_freq(ts)
    print(f'\nFrequency\nfreq: {freq}\n\n')
