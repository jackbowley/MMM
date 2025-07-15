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
    """
    df_trans = pd.DataFrame(index=df_data.index)
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
                series = s_curve(series, row['saturation'], row['inflection'], height=1)
            df_trans[var] = series
            # Multiply by beta
            beta_val = row['beta'] if not pd.isnull(row['beta']) else 1
            df_values[var] = df_trans[var] * beta_val
        else:
            # If column missing, fill with NaN
            df_trans[var] = np.nan
            df_values[var] = np.nan
    # Add 'actual' column as row sum
    df_values['actual'] = df_values.sum(axis=1)
    return df_trans, df_values


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
