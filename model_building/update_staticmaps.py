import hydromt
import logging
from hydromt_wflow import WflowModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')

mod = WflowModel(root="../model_orig", mode="r", config_fn="routing_runs_kin.toml")

mod.data_catalog.from_predefined_catalogs('deltares_data')

mod.staticmaps

mod.setup_river_floodplain(
    hydrography_fn="merit_hydro",
    )

mod.set_root('../model_new', mode='w')
mod.write()