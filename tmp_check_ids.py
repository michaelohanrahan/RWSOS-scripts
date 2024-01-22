#%%
import glob
import geopandas as gpd

files = glob.glob(R'P:\11209265-grade2023\wflow\wflow_rhine_julia\wflow_rhine_202303\staticgeoms/gauges_*.geojson')

dfs = {}

ids = []

for file in files:
    gdf = gpd.read_file(file)
    dfs[file] = gdf
    print(gdf.columns)

    # ids.append(gdf.index.values)
    for col in ['WFLOW_ID', 'grdc_no', 'gauge_ID', 'wflow_ID', 'fid']:
        try:
            ids.append(gdf[col].values)
        except:
            pass

#%%
import numpy as np
out = np.concatenate(ids).ravel()


val, occ = np.unique(out, return_counts=True)