import xarray as xr
import pandas as pd

def print_variables_dates_and_frequencies(filepaths):
    """
    Prints the variables, start and end dates, and frequency of time series data
    for each file in the given list of filepaths.

    Parameters:
    - filepaths (list): A list of filepaths to the NetCDF files.

    Returns:
    None
    """
    for fp in filepaths:
        ds = xr.open_dataset(fp)
        print(f'\n{fp}\n{ds.variables}\n\n')
        print(f'\n{fp}\nstart: {ds.time.min().values}\nend: {ds.time.max().values}\n\n')
        ts = pd.Series(ds.time.values)
        freq = pd.infer_freq(ts)
        print(f'\nFrequency\nfreq: {freq}\n\n')

# # Example usage
# filepaths = ['/path/to/file1.nc', '/path/to/file2.nc', '/path/to/file3.nc']
# print_variables_dates_and_frequencies(filepaths)
