import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
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




def arma_model_selection(series, max_p=3, max_q=3, verbose=False):
    results = []

    for p, q in itertools.product(range(max_p+1), range(max_q+1)):
        if p == 0 and q == 0:
            continue
        
        try:
            model = ARIMA(series, order=(p,0,q)).fit()
            
            results.append({
                'p': p,
                'q': q,
                'AIC': model.aic,
                'BIC': model.bic,
                'LogLik': model.llf
            })
            
            if verbose:
                print(f"ARMA({p},{q}) AIC={model.aic:.3f} BIC={model.bic:.3f}")
        
        except Exception as e:
            if verbose:
                print(f"ARMA({p},{q}) fallo: {e}")
            continue
    
    df = pd.DataFrame(results)
    return df.sort_values(by="AIC").reset_index(drop=True)


def significant_params(model, alpha=0.05):
    results = pd.DataFrame({
        'coef': model.params,
        'std_err': model.bse,
        'pvalue': model.pvalues
    })

    significant = results[results['pvalue'] < alpha]
    not_significant = results[results['pvalue'] >= alpha]

    print(f"Significant at {alpha*100}% level:")
    print(significant)
    print(f"\nNot significant at {alpha*100}% level:")
    print(not_significant)


def forecast_with_ci(model, steps=24):

    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()  

    ci_df = pd.DataFrame({
        "Forecast": pred_mean,
        "Lower 95%": conf_int[:, 0],
        "Upper 95%": conf_int[:, 1]
    })

    print("95% Confidence Intervals:")
    print(ci_df)
    return ci_df



def calculate_rmse(model, y_train, y_test):
    """
    Calculates RMSE for training and test data using a fitted ARIMA model.

    Parameters:
    -----------
    model : ARIMAResults
        Fitted ARIMA model.
    y_train : array-like
        Training data.
    y_test : array-like
        Test data.

    Returns:
    --------
    rmse_train : float
        RMSE on training data.
    rmse_test : float
        RMSE on test data.
    """
    # RMSE on training data
    fitted_vals = model.fittedvalues
    rmse_train = np.sqrt(mean_squared_error(y_train, fitted_vals))

    # RMSE on test data
    forecast_test = model.forecast(steps=len(y_test))
    rmse_test = np.sqrt(mean_squared_error(y_test, forecast_test))

    print("RMSE (Training):", rmse_train)
    print("RMSE (Test):", rmse_test)

    return rmse_train, rmse_test
