import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_process import arma_acf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

def plot_acf_sample(series, lags=20, save=False, path=None):
    acf_vals = acf(series, nlags=lags, fft=False)

    plt.figure(figsize=(7,4))
    plt.stem(range(lags+1), acf_vals)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Sample ACF (lags up to {lags})")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.grid(True)

    if save and path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()




def plot_pacf_sample(series, lags=20, save=False, path=None):
    pacf_vals = pacf(series, nlags=lags, method='yw')

    plt.figure(figsize=(7,4))
    plt.stem(range(lags+1), pacf_vals)
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f"Sample PACF (lags up to {lags})")
    plt.xlabel("Lag")
    plt.ylabel("PACF")
    plt.grid(True)

    if save and path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_acf_theoretical(phi, theta, lags=20, save=False, path=None):
    acf_vals = arma_acf(ar=phi, ma=theta, lags=lags)
    acf_vals = np.insert(acf_vals, 0, 1.0)

    plt.figure(figsize=(7,4))
    plt.stem(range(lags+1), acf_vals)
    plt.axhline(0, linewidth=1)
    plt.title(f"Theoretical ACF (lags up to {lags})")

    if save and path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pacf_theoretical(phi, theta=None, lags=20, save=False, path=None):
    # Simular largo para aproximación teórica
    from statsmodels.tsa.arima_process import ArmaProcess
    
    arma_process = ArmaProcess(np.r_[1, -np.array(phi)],
                               np.r_[1, np.array(theta if theta else [])])
    
    sim = arma_process.generate_sample(nsample=5000)
    
    plt.figure(figsize=(7,4))
    plot_pacf(sim, lags=lags, method='ywm')
    plt.title(f"Theoretical PACF (approx, lags up to {lags})")

    if save and path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()