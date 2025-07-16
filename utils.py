# flake8: noqa: E501
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def adstock(x, rate):
    """
    Apply adstock transformation to a media variable.
    x: array-like, media spend or GRP
    rate: float, carryover rate between 0 and 1
    Returns: numpy array of adstocked values
    """
    result = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            result[i] = x[i]
        else:
            result[i] = x[i] + rate * result[i-1]
    return result


def s_curve(grps, saturation, inflection, height):
    """
    Applies the transformation: arctan((grps**inflection)/(saturation**inflection)) / (pi/2) * height
    grps: array-like, input variable
    saturation, inflection, height: scalars
    Returns: numpy array of transformed values
    """
    return (np.arctan((grps ** inflection) / (saturation ** inflection)) / (np.pi / 2)) * height


def create_transformed_tables(df_data, df_var_spec):
    """
    For each variable in df_var_spec, apply adstock (if carryover is not None), then s_curve (if saturation and inflection are not None),
    then multiply by beta. Returns df_trans and df_values tables, with df_values including an 'actual' column summing each row.
    If 'normalize' is True in var spec, applies mean normalization: (x - mean) / (max - min)
    """
    df_trans = pd.DataFrame(index=df_data.index)
    df_trans_norm = pd.DataFrame(index=df_data.index)
    df_values = pd.DataFrame(index=df_data.index)
    for _, row in df_var_spec.iterrows():
        var = row['variable']
        if var in df_data.columns:
            series = df_data[var].values.copy()
            # Apply adstock if carryover is not None and not NaN
            if row.get('carryover') is not None and not pd.isnull(row['carryover']):
                series = adstock(series, row['carryover'])
            # Apply s_curve if saturation and inflection are not None and not NaN
            if (row.get('saturation') is not None and not pd.isnull(row['saturation']) and
                row.get('inflection') is not None and not pd.isnull(row['inflection'])):
                series = s_curve(series, row['saturation'], row['inflection'], row['height'])
            df_trans[var] = series

            # Mean normalization if 'normalize' is True
            if (row.get('normalize') is not None and not pd.isnull(row['normalize']) and
                row['normalize'] == True):
                mean = np.mean(series)
                max_ = np.max(series)
                min_ = np.min(series)
                denom = max_ - min_
                # Avoid division by zero
                if denom == 0:
                    norm_series = series - mean
                else:
                    # norm_series = (series - mean) / denom
                    norm_series = (series) / denom ## use this as otherwise get weird confidence range

                series = norm_series
            df_trans_norm[var] = series

            # Multiply by beta
            beta_val = row['beta'] if not pd.isnull(row['beta']) else 1
            df_values[var] = df_trans_norm[var] * beta_val
        else:
            # If column missing, fill with NaN
            df_trans[var] = np.nan
            df_trans_norm[var] = np.nan
            df_values[var] = np.nan
    # Add 'actual' column as row sum
    df_values['actual'] = df_values.sum(axis=1)
    return df_trans, df_trans_norm, df_values


def plot_stacked_area_with_actual(df_values, y_axis_min=None, y_axis_max=None):
    """
    Plots a stacked area chart for all variables except 'error' and 'actual' on the left y-axis,
    and plots 'actual' as a line on the right y-axis. Both axes share the same scale.
    y_axis_min, y_axis_max: set min/max for both y-axes. If None, use default.
    """
    cols = [col for col in df_values.columns if col not in ['error', 'actual']]
    x = df_values.index
    y = df_values[cols].values.T
    actual = df_values['actual'].values
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.stackplot(x, y, labels=cols)
    ax1.set_ylabel('Stacked Variables')
    ax1.set_xlabel('Date')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(x, actual, color='black', label='Actual', linewidth=2)
    ax2.set_ylabel('Actual')
    if y_axis_min is not None and y_axis_max is not None:
        ax1.set_ylim(y_axis_min, y_axis_max)
        ax2.set_ylim(y_axis_min, y_axis_max)
    else:
        min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(min_y, max_y)
        ax2.set_ylim(min_y, max_y)
    ax2.legend(loc='upper right')
    plt.title('Stacked Area Chart of Variables with Actual as Line')
    plt.tight_layout()
    plt.show()


def plot_transformed_vs_raw(raw, transformed, xlabel='Raw values', ylabel='Transformed values', title='Line Plot: Raw vs Transformed'):
    """
    Plots raw vs transformed values as a line, with raw values sorted ascending.
    """
    sort_idx = np.argsort(raw)
    raw_sorted = raw[sort_idx]
    transformed_sorted = transformed[sort_idx]
    plt.figure(figsize=(10, 6))
    plt.plot(raw_sorted, transformed_sorted, linestyle='-', marker=None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_data_set(df):
    cols_to_plot = [col for col in df.columns if col != 'c']
    n_cols = 2
    n_rows = int(np.ceil(len(cols_to_plot) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True)
    axes = axes.flatten()
    for i, col in enumerate(cols_to_plot):
        df[col].plot(ax=axes[i], title=col)
        axes[i].set_ylabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_actual_fitted_residuals(model,y):
    """
    Plots actual, fitted, and residuals from a statsmodels OLS model and target series.
    Args:
        model: fitted statsmodels OLS model,
        y: actual target series as a pandas Series
    """
    import matplotlib.pyplot as plt
    fitted = model.fittedvalues
    actual = y
    residual = actual - fitted

    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(actual.index, actual, label='Actual', color='black', linestyle='-')
    ax1.plot(actual.index, fitted, label='Fitted', color='red', linestyle='-')
    ax2.plot(actual.index, residual, label='Residual (Actual - Fitted)', color='green', linestyle='-')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Actual / Fitted')
    ax2.set_ylabel('Residual')

    # Scale residual axis so residuals are shown below actual/fitted
    resid_min = min(residual.min(), 0)
    resid_max = max(residual.max()*4, 0)
    ax2.set_ylim(resid_min, resid_max)
    ax2.axhline(0, color='gray', linestyle=':')

    plt.title('Actual vs. Fitted Values and Residuals')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Example usage: 
# plot_actual_fitted_residuals(model,df_values['actual'])


def plot_response_curve(model, series_name,adstock_series, trans_series, actual_series):
    """
    Plots coef * trans value vs trans value for a given series, with shaded 95% CI and actuals as dots.
    Inputs:
        model: fitted OLS model object
        series_name: str, name of the variable,
        adsock_series: Series of adstocked values
        trans_series: Series of transformed values (to be multiplied by coef)
        actual_series: Series of actual values (used in dgp)
        coef: regression coefficient for the variable
        std_err: standard error for the variable
    """
    import numpy as np
    import matplotlib.pyplot as plt

    coef = model.params['media1']
    std_err = model.bse['media1']

    conf95 = 1.96
    fit = coef * trans_series
    err = conf95 * std_err * trans_series.abs()
    # Sort by trans_series for a smooth line
    sort_idx = np.argsort(adstock_series)
    # trans_sorted = trans_series.values[sort_idx]
    adstock_sorted = adstock_series.values[sort_idx]
    fit_sorted = fit.values[sort_idx]
    err_sorted = err.values[sort_idx]
    actuals_sorted = actual_series.values[sort_idx]
    plt.figure(figsize=(10, 6))
    plt.plot(adstock_sorted, fit_sorted, label=f'{series_name} (fit)', color='blue')
    plt.fill_between(adstock_sorted, fit_sorted - err_sorted, fit_sorted + err_sorted, color='blue', alpha=0.2, label='95% CI')
    plt.scatter(adstock_sorted, actuals_sorted, color='black', label='actuals', s=30, alpha=0.7)
    plt.xlabel(f'Adstocked {series_name}')
    plt.ylabel('Contribution (coef * trans)')
    plt.title(f'{series_name} Estimated Contribution vs. Adstocked media with 95% Confidence Bounds and Actuals')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_contribution_with_actuals('media1', df_trans['media1'], df_values['media1'], )