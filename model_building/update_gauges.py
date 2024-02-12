import os
import sys
import geopandas as gpd
from hydromt.log import setuplog
from hydromt_wflow import WflowModel
from pathlib import Path

def main(root:str, 
         gauges, 
         new_root:bool = None, 
         mode:str = "w", 
         basename:str = "obs", 
         index_col:str = "SiteNumber",
         snap_to_river:bool = True,
         max_dist:int = 10000,
         derive_subcatch:bool = True,
         crs='EPSG:4326',
         config_fn="wflow_sbm.toml",):
    
    if new_root is None:
        new_root = root
        mode = "w+"

    logger = setuplog("build", log_level=20)

    if not Path(gauges).is_absolute():
        gauges = Path(Path.cwd(), gauges)

    w = WflowModel(
        root=root,
        mode="r",
        config_fn = config_fn,
        data_libs = [],
        logger=logger,
        )
    
    w.read()
    
    w.set_root(
        root=new_root,
        mode=mode
        )

    # Updating
    # Based on gauges geojson
    gauges = gpd.read_file(gauges)
    gauges = gauges.set_crs(allow_override=True, crs=crs)
    
    w.setup_gauges(
        gauges_fn=gauges,
        snap_to_river=snap_to_river,
        derive_subcatch=True,
        index_col=index_col,
        basename=basename,
    )
    
    print('writing config')
    w.write_config()
    print('writing grid')
    w.write_grid()
    print('writing geoms')
    w.write_geoms()

if __name__ == "__main__":
    root = os.getcwd()
    new_root = None
    if sys.argv.__len__() < 3:
        raise ValueError(f"Update wflow requires 2 arguments -> {sys.argv.__len__()-1} given")
    if len(sys.argv) > 3:
        new_root = os.path.join(root, sys.argv[3])
    root = os.path.join(root, sys.argv[1])
    gauges = sys.argv[2]
    
    main(root, gauges, new_root)