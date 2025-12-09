import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np


import warnings

def table_adf(series, **kwargs):
    """Returns ADF test results in a pandas DataFrame"""
    res = adfuller(series, **kwargs)
    output = pd.DataFrame({
        "Statistic": [res[0]],
        "p-value": [res[1]],
        "Lags used": [res[2]],
        "N obs": [res[3]],
    })
    for key, val in res[4].items():
        output[f"Crit ({key})"] = [val]
    return output

def table_kpss(series, **kwargs):
    """Returns KPSS test results in a pandas DataFrame"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kpss(series, **kwargs)

    output = pd.DataFrame({
        "Statistic": [res[0]],
        "p-value": [res[1]],
        "Lags used": [res[2]],
    })
    for key, val in res[3].items():
        output[f"Crit ({key})"] = [val]
    return output



