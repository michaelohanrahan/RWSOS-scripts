import pandas as pd
import numpy as np


# def prepare_HBV_maas():
hbv = pd.read_excel(r"P:\11209265-grade2023\wflow\wflow_meuse_julia\HBV\HBV_60min_stations_2004-2016.xlsx", 
                  index_col=0, 
                  parse_dates=True,
                  skiprows=[0,1,4,5])


#%%

hbv 