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

def decomposition(df_var_spec, df_values):
    """
    Decompose the model into variable contributions for each date.
    Returns a DataFrame df_decomp with each variable's contribution and a 'total' column.
    """
    dc = df_values.copy()
    # Sum all columns except the last two in dc
    var_cols = df_var_spec['variable'].tolist()
    # Sum all columns except the last two in dc
    dc['model'] = dc[var_cols].sum(axis=1)
    dc['actual'] = np.log(dc['actual'])
    # Populate dc2 based on decomp_ref in df_var_spec
    dc2 = dc.copy()
    dc2['ref_sum'] = 0
    for _, row in df_var_spec.iterrows():
        var = row['variable']
        if var in dc2.columns and 'decomp_ref' in row and not pd.isnull(row['decomp_ref']):
            series = dc2[var]
            ref_type = str(row['decomp_ref']).lower()
            if ref_type == 'min':
                ref_val = series.min()
                dc2[var] = series - ref_val
                dc2['ref_sum'] += ref_val
            elif ref_type == 'max':
                ref_val = series.max()
                dc2[var] = series - ref_val
                dc2['ref_sum'] += ref_val
            elif ref_type == 'average':
                ref_val = series.mean()
                dc2[var] = series - ref_val
                dc2['ref_sum'] += ref_val

    # # Check that the sum of dc2 columns in var_spec plus ref_sum equals the model column
    # var_cols = [row['variable'] for _, row in df_var_spec.iterrows() if row['variable'] in dc2.columns]

    dc2['var_sum_plus_ref'] = dc2[var_cols].sum(axis=1) + dc2['ref_sum']
    check_equal = np.allclose(dc2['var_sum_plus_ref'], dc['model'])
    print('Check passed:', check_equal)

    dc2['check']=dc2['var_sum_plus_ref'] - dc['model']
    dc2['check'].sum()

    # Create dc3 by grouping dc2 columns into the matching group in var_spec, with ref_sum in 'base'
    group_map = {row['variable']: row['group'] for _, row in df_var_spec.iterrows() if row['variable'] in dc2.columns and 'group' in row}
    grouped_cols = {}
    for var, group in group_map.items():
        if group not in grouped_cols:
            grouped_cols[group] = []
        grouped_cols[group].append(var)
    grouped_cols['base'].append('ref_sum')  # Ensure 'base' group includes ref_sum
    groups = list(grouped_cols.keys())
    dc3 = pd.DataFrame(index=dc2.index)
    for group, cols in grouped_cols.items():
        dc3[group] = dc2[cols].sum(axis=1)


    # Cross-check the sum equals the model column
    dc3['model'] = dc3.sum(axis=1)
    check_equal = np.allclose(dc3['model'], dc['model'])
    print('Group sum check passed:', check_equal)

    dc4 = pd.DataFrame(index=dc3.index)
    dc4['base'] = np.exp(dc3['base'])
    model = np.exp(dc3['model'])
    model_col='model'
    for col in dc3.columns:
        if col not in ['base', model_col]:
            dc4[col] = model - np.exp(dc3[model_col] - dc3[col])

    # Sum all columns except 'model' and 'sum' to get model1
    dc4['model1'] = dc4[groups].sum(axis=1)
    dc4['model'] = np.exp(dc3['model'])
    dc4['diff'] = dc4['model'] - dc4['model1']

    # Create df_decomp by apportioning diff in dc4 across variables (excluding model1, Model1, model), then add apportioned diff to dc4 values using abs(var_cols) div abs_sum * diff
    exclude_cols = ['model1', 'Model1', 'model', 'diff']
    var_cols = [col for col in dc4.columns if col not in exclude_cols]
    abs_sum = dc4[var_cols].abs().sum(axis=1)
    apportioned = dc4[var_cols].abs().div(abs_sum, axis=0).multiply(dc4['diff'], axis=0)
    apportioned = apportioned.fillna(0)  # In case of division by zero
    dc5 = dc4[var_cols] + apportioned
    dc5['model'] = dc5.sum(axis=1)
    dc5['model_og'] = df_values['model']
    dc5['diff'] = dc5['model'] - dc5['model_og']

    check_equal = np.allclose(dc5['model'], dc5['model_og'])
    print('Check passed:', check_equal)
    df_decomp = dc5.copy()
    return df_decomp

def create_transformed_tables(df_data, df_var_spec, log_dep_var = False):
    """
    For each variable in df_var_spec, apply adstock (if carryover is not None), 
    then multiply by beta. Returns df_trans and df_values tables, with df_values including an 'actual' column summing each row.
    """
    df_trans = pd.DataFrame(index=df_data.index)
    df_values = pd.DataFrame(index=df_data.index)
    for _, row in df_var_spec.iterrows():
        var = row['variable']
        if var in df_data.columns:
            series = df_data[var].values.copy()
            if row.get('carryover') is not None and not pd.isnull(row['carryover']):
                series = adstock(series, row['carryover'])          

            if (row.get('log') is not None and not pd.isnull(row['log']) and
                row['log'] == True):
                series = np.log(series)

            df_trans[var] = series
            beta_val = row['beta'] if not pd.isnull(row['beta']) else 1
            df_values[var] = df_trans[var] * beta_val
        else:
            # If column missing, fill with NaN
            df_trans[var] = np.nan
            df_values[var] = np.nan
    # Add 'actual' column as row sum
    df_values['model'] = df_values.sum(axis=1)
    if log_dep_var:
        # Log transform the dependent variable if specified
        df_values['model'] = np.exp(df_values['model'])
    
    df_values['actual'] = df_values['model'] * (1+ df_data['error'])
    return df_trans, df_values

def calc_roi(df_decomp, df_var_spec,df_data,format=True):
    # For each variable in df_decomp that has a value in the spend_variable column in df_var_spec,
    # create a table dividing the sum of the value in df_decomp by the sum of the variable in df_data.

    # Get mapping of variable to spend_variable (non-null only)
    spend_map = df_var_spec[['variable', 'spend_variable']].dropna()

    results = []
    for _, row in spend_map.iterrows():        
        var = row['variable']
        spend_var = row['spend_variable']
        # Only proceed if var is in df_decomp columns and spend_var is in df_data columns
        if var in df_decomp.columns and spend_var in df_data.columns:
            decomp_sum = df_decomp[var].sum()
            spend_sum = df_data[spend_var].sum()
            roi = decomp_sum / spend_sum if spend_sum != 0 else float('nan')
            results.append({'variable': var, 'value': decomp_sum, 'spend_sum': spend_sum, 'roi': roi})


    df_rois = pd.DataFrame(results)
    # Calculate totals before formatting
    total_value = df_rois['value'].sum()
    total_spend = df_rois['spend_sum'].sum()
    total_roi = total_value / total_spend if total_spend != 0 else float('nan')
    total_row = {
        'variable': 'Total',
        'value': total_value,
        'spend_sum': total_spend,
        'roi': total_roi
    }
    df_rois = pd.concat([df_rois, pd.DataFrame([total_row])], ignore_index=True)
    if format:
        # Format columns
        df_rois['spend_sum'] = df_rois['spend_sum'].apply(lambda x: f"{int(round(x)):,}")
        df_rois['value'] = df_rois['value'].apply(lambda x: f"{int(round(x)):,}")
        df_rois['roi'] = df_rois['roi'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else '')
    return df_rois