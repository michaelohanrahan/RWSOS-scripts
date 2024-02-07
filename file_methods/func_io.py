import pandas as pd
from datetime import datetime

def read_filename_txt(filename):
    """
    Reads a CSV file, separated by ';' and returns a pandas DataFrame.

    Parameters:
    filename (str): The path to the CSV file.

    Returns:
    dfm (pandas.DataFrame): The DataFrame containing the data from the CSV file.
    """
    dfm = pd.read_csv(filename, sep=';', parse_dates=[0], header=None, skiprows=36, na_values=-999, encoding_errors='ignore')
    dfm = dfm.drop([1], axis=1)
    dfm = dfm.drop([0], axis=0)
    dfm.index = dfm[0]
    dfm.index.name = None
    dfm = dfm.drop([0], axis=1)
    return dfm


def read_lakefile(filename):
    """
    Read lake file and return a DataFrame with resampled daily data.

    Parameters:
    filename (str): The path to the lake file.

    Returns:
    dfm (pandas.DataFrame): DataFrame with resampled daily data.
    """
    dateparse = lambda x: datetime.strptime(x[17:34], '%Y.%m.%d %M:%S')
    dfm = pd.read_csv(filename, sep=';', parse_dates=[1], header=None, skiprows=7, na_values=-999, date_parser=dateparse)
    dfm = dfm.drop([0], axis=1)
    dfm.index = dfm[1]
    dfm = dfm.drop([1], axis=1)
    dfm = dfm.resample('D', label='right').mean()
    return dfm
