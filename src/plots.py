import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import arma_acf, ArmaProcess
import scipy.stats as stats

# Estilo global
plt.rcParams.update({
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.5
})


def plot_acf_sample(series, lags=20, save=False, path=None):
    fig, ax = plt.subplots(figsize=(8,5))
    plot_acf(series, lags=lags, fft=False, alpha=0.05, ax=ax)
    ax.set_title(f"Sample ACF (lags={lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticks(range(0, lags+1))
    if save and path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pacf_sample(series, lags=20, save=False, path=None):
    fig, ax = plt.subplots(figsize=(8,5))
    plot_pacf(series, lags=lags, method='ywm', alpha=0.05, ax=ax)
    ax.set_title(f"Sample PACF (lags={lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticks(range(0, lags+1))
    if save and path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_acf_theoretical(phi, theta, lags=20, save=False, path=None):
    acf_vals = arma_acf(ar=np.r_[1, -np.array(phi)],
                        ma=np.r_[1, np.array(theta)],
                        lags=lags)
    acf_vals = np.insert(acf_vals, 0, 1.0)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.stem(range(lags+1), acf_vals, basefmt=" ")
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Theoretical ACF (lags={lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticks(range(0, lags+1))
    if save and path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_pacf_theoretical(phi, theta=None, lags=20, save=False, path=None):
    arma_process = ArmaProcess(np.r_[1, -np.array(phi)],
                               np.r_[1, np.array(theta if theta else [])])
    sim = arma_process.generate_sample(nsample=5000)

    fig, ax = plt.subplots(figsize=(8,5))
    plot_pacf(sim, lags=lags, method='ywm', alpha=0.05, ax=ax)
    ax.set_title(f"Theoretical PACF (approx, lags={lags})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("PACF")
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_xticks(range(0, lags+1))
    if save and path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def arma_forecast_plot(model, X_train, X_test, steps=24, title='Forecast'):
    """
    Produce forecast and plot training data, test data, and forecast with confidence intervals.

    Parameters
    ----------
    model : fitted ARIMA/ARMA model
        The model already trained (e.g., ARIMA(...).fit())
    X_train : array-like
        Training data
    X_test : array-like
        Testing data (plotted for comparison)
    steps : int
        Number of forecast steps
    title : str
        Plot title

    Returns
    -------
    pred_mean : numpy array
        Forecasted values
    conf_int : numpy array
        Confidence intervals
    """
    
    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(X_train)), X_train, label='Training Data')
    plt.plot(range(len(X_train), len(X_train)+steps), pred_mean, label='Forecast')

    plt.fill_between(range(len(X_train), len(X_train)+steps),
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.3, label='95% CI')

    plt.plot(range(len(X_train), len(X_train)+len(X_test)), X_test, label='Actual Data')

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return pred_mean, conf_int



def plot_residuals(residuals):
    """
    Prints basic statistics of residuals and plots them.

    Parameters:
    -----------
    residuals : array-like
        Residuals of a fitted time series model.

    Returns:
    --------
    stats : dict
        Dictionary containing mean, standard deviation, min, and max of residuals.
    """
    stats = {
        "mean": np.mean(residuals),
        "std": np.std(residuals),
        "min": np.min(residuals),
        "max": np.max(residuals)
    }

    print("=== Residuals Statistics ===")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Minimum: {stats['min']:.4f}, Maximum: {stats['max']:.4f}")

    plt.figure(figsize=(10,4))
    plt.plot(residuals)
    plt.title("Residuals of Fitted Model")
    plt.xlabel("Time Index")
    plt.ylabel("Residual")
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.show()

    return stats



def plot_acf_pacf_residuals(residuals, lags=20, alpha=0.05):
    """
    Plots ACF and PACF of residuals side by side.

    Parameters:
    -----------
    residuals : array-like
        Residuals of a fitted model.
    lags : int
        Number of lags to display.
    alpha : float
        Confidence level for significance bounds (default 0.05).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # ACF
    plot_acf(residuals, lags=lags, alpha=alpha, ax=axes[0])
    axes[0].set_title("ACF of Residuals")

    # PACF
    plot_pacf(residuals, lags=lags, alpha=alpha, method='ywm', ax=axes[1])
    axes[1].set_title("PACF of Residuals")

    plt.tight_layout()
    plt.show()


def residuals_normality_test(residuals):
    """
    Generates QQ-plot and performs normality tests on residuals.

    Parameters:
    -----------
    residuals : array-like
        Residuals of a fitted model.

    Returns:
    --------
    results : dict
        Dictionary with Shapiro-Wilk and Jarque-Bera statistics and p-values.
    """
    # QQ-plot
    plt.figure(figsize=(6,6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ-Plot of Residuals")
    plt.grid(True)
    plt.show()

    # Shapiro-Wilk test
    shapiro_test = stats.shapiro(residuals)
    print("Shapiro-Wilk Test:")
    print(f"Statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")

    # Jarque-Bera test
    jb_test = stats.jarque_bera(residuals)
    print("\nJarque-Bera Test:")
    print(f"Statistic: {jb_test.statistic:.4f}, p-value: {jb_test.pvalue:.4f}")

    results = {
        "shapiro_stat": shapiro_test.statistic,
        "shapiro_pvalue": shapiro_test.pvalue,
        "jb_stat": jb_test.statistic,
        "jb_pvalue": jb_test.pvalue
    }

    return results



def arma_forecast_plot2(model, X_train, X_test, steps=24, title='Forecast'):
    """
    Produce forecast and plot training data, test data, and forecast with confidence intervals.
    """
    forecast = model.get_forecast(steps=steps)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()  # DataFrame con columnas ['lower Value', 'upper Value']

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(X_train)), X_train, label='Training Data')
    plt.plot(range(len(X_train), len(X_train)+steps), pred_mean, label='Forecast')

    # Usar iloc para acceder a las columnas
    plt.fill_between(range(len(X_train), len(X_train)+steps),
                     conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     alpha=0.3, label='95% CI')

    # Graficar los datos reales de test
    plt.plot(range(len(X_train), len(X_train)+len(X_test)), X_test, label='Actual Data')

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    return pred_mean, conf_int
