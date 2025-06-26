import pandas as pd
import numpy as np
import scipy.stats as st
import os


def align_dataframe(data_df, results_df, freq='h'):
    '''
    Converts dataframe from a row being predictions shifted from the last observed date
    to each row being the prediction of the observed date at k steps ahead (not loseless)
    '''
    df = pd.merge(results_df, data_df[['unique_id', 'ds', 'y']], on=['ds', 'unique_id'], how='left')
    col = df.pop('y') 
    df.insert(2, '0', col)  
    df_new = df[['unique_id', 'ds']].copy()
    for i in range(int(df.columns[-1])+1):
        temp = df[['unique_id', 'ds', str(i)]].copy()
        temp['ds'] = temp['ds'] + pd.Timedelta(i, unit=freq)
        df_new = df_new.join(temp.set_index(['unique_id', 'ds']), on=['unique_id', 'ds'])
    return df_new

def unalign_dataframe(df, freq='h'):
    '''
    Inverts align_dataframe (not loseless)
    '''
    df_new = df[['unique_id', 'ds']].copy()
    for i in range(int(df.columns[-1])+1):
        temp = df[['unique_id', 'ds', str(i)]].copy()
        temp['ds'] = temp['ds'] - pd.Timedelta(i, unit=freq)
        df_new = df_new.join(temp.set_index(['unique_id', 'ds']), on=['unique_id', 'ds'])
    return df_new

def get_p_values(df, lower_df, higher_df, allowed_error = 0.01, min_val = -1, min_std = -1, unalign=True):
    '''
    Get p-values from each prediction using a mixture of linear/gaussian assumptions with lower/higher
    dataframes representing the value at 0.1 p-value (not loseless)
    Takes aligned df's as input
    '''    
    ks = [str(i) for i in range(1,int(df.columns[-1])+1)]
    if min_std < 0:
        min_std = np.nanquantile((df[ks] - lower_df[ks]), 0.5) / 1.282
    p_df = df.copy()
    p_df[ks] = np.nan
    for k in ks:
        # get cells where true is less than prediction (and lower is less than median)
        lower_val = lower_df.loc[(df[k] > df["0"]) & (df[k] > lower_df[k]), k]
        pred_val = df.loc[(df[k] > df["0"]) & (df[k] > lower_df[k]), k]
        true_val = df.loc[(df[k] > df["0"]) & (df[k] > lower_df[k]), "0"]
        std = (pred_val-lower_val) / 1.282
        std = std.clip(lower=min_std)
        z_score = (true_val-pred_val) / std
        p_df.loc[(df[k] > df["0"]) & (df[k] > lower_df[k]), k] = st.norm.cdf(z_score)

        # get cells where true is more than prediction (and higher is greater than median)
        higher_val = higher_df.loc[(df[k] <= df["0"]) & (df[k] < higher_df[k]), k]
        pred_val = df.loc[(df[k] <= df["0"]) & (df[k] < higher_df[k]), k]
        true_val = df.loc[(df[k] <= df["0"]) & (df[k] < higher_df[k]), "0"]
        std = (higher_val-pred_val) / 1.282
        std = std.clip(lower=min_std)
        z_score = (pred_val-true_val) / std
        p_df.loc[(df[k] <= df["0"]) & (df[k] < higher_df[k]), k] = st.norm.cdf(z_score)

        # get cells where higher is less than median or lower is higher than median (else)
        condition = ((df[k] > df["0"]) & (df[k] < lower_df[k])) | ((df[k] <= df["0"]) & (df[k] > higher_df[k]))
        higher_val = higher_df.loc[condition, k]
        lower_val = lower_df.loc[condition, k]
        pred_val = df.loc[condition, k]
        true_val = df.loc[condition, "0"]
        std = (np.minimum(lower_val, pred_val) - np.maximum(higher_val, pred_val)) / 1.282
        std = std.clip(lower=min_std)
        z_score = -abs(pred_val-true_val) / std
        p_df.loc[condition, k] = st.norm.cdf(z_score)
    if unalign:
        p_df = unalign_dataframe(p_df)
        p_df = p_df.dropna(subset=ks, how='any')
    return p_df

def get_score(df, lower_df, higher_df, method='geometric', min_std=-1):
    '''
    Returns an aggregate score single
    Options: Arithmetic Mean, Geometric Mean, Fisher's method
    method: {'arithmetic', 'geometric', 'fisher'}
    '''
    p_df = get_p_values(df, lower_df, higher_df, min_std=min_std, unalign=False)
    ks = [str(i) for i in range(1,int(p_df.columns[-1])+1)]
    score_df = p_df.copy()
    if method == 'arithmetic':
        for i in range(1,len(ks)):
            score_df[ks[i]] = score_df[ks[i]] + score_df[ks[i-1]]
        score_df[ks] = score_df[ks].div(range(1,len(ks)+1))
    elif method == 'geometric':
        for i in range(1,len(ks)):
            score_df[ks[i]] = score_df[ks[i]] * score_df[ks[i-1]]
        score_df[ks] = score_df[ks].pow(1/np.arange(1,len(ks)+1))
    elif method == 'fisher':
        score_df['1'] = np.log(score_df['1'])
        for i in range(1,len(ks)):
            score_df[ks[i]] = np.log(score_df[ks[i]]) + score_df[ks[i-1]]
        score_df[ks] = -2*score_df[ks]
    return score_df

def get_score_from_p_values(p_df, method='geometric'):
    '''
    Returns an aggregate score given p-values
    Options: Arithmetic Mean, Geometric Mean, Fisher's method
    method: {'arithmetic', 'geometric', 'fisher'}
    '''
    ks = [str(i) for i in range(1,int(p_df.columns[-1])+1)]
    score_df = p_df.copy()
    if method == 'arithmetic':
        for i in range(1,len(ks)):
            score_df[ks[i]] = score_df[ks[i]] + score_df[ks[i-1]]
        score_df[ks] = score_df[ks].div(range(1,len(ks)+1))
    elif method == 'geometric':
        for i in range(1,len(ks)):
            score_df[ks[i]] = score_df[ks[i]] * score_df[ks[i-1]]
        score_df[ks] = score_df[ks].pow(1/np.arange(1,len(ks)+1))
    elif method == 'fisher':
        score_df['1'] = np.log(score_df['1'])
        for i in range(1,len(ks)):
            score_df[ks[i]] = np.log(score_df[ks[i]]) +score_df[ks[i-1]]
        score_df[ks] = -2*score_df[ks]
    return score_df

def load_model_results(dataset):
    results = {}
    path = f"model_results/{dataset}/"
    models = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    for model in models:
        results[model] = {}
        folder_path = f'model_results/{dataset}/{model}/'
        csv_files = [f for f in os.listdir(folder_path) if (f.endswith('.csv') and ('id_mapping' not in f))]
        for csv_file in csv_files:
            point_estimate = csv_file[:-4]
            file_path = f"model_results/{dataset}/{model}/{csv_file}"
            df = pd.read_csv(file_path, index_col=None, parse_dates=['ds'])
            results[model][point_estimate] = df
    return results

def unique_last_indices(arr):
    reversed_arr = arr[::-1]
    unique_values, unique_indices_reversed = np.unique(reversed_arr, return_index=True)
    unique_indices = len(arr) - 1 - unique_indices_reversed
    return unique_indices

def roc_curve(score_df, method, anomaly_loc, k):
    # return True Positive Rate and False Positive Rate
    steps_ahead = int(k)
    if method == 'fisher': # higher is worse
        worst_args = np.argsort(score_df[k])[::-1]
    else: # lower is worse
        worst_args = np.argsort(score_df[k])[::1]

    # roc = np.zeros((worst_args.shape[0], 2))
    tpr = np.zeros_like(worst_args, dtype=float)
    fpr = np.zeros_like(worst_args, dtype=float)
    correct = np.zeros_like(worst_args) # split this into each timeseries
    for counter, i in enumerate(worst_args):
        id = score_df.iloc[i,0]
        timestamp = score_df.iloc[i,1]
        # act_pos: 1 for TP and 0 for FP
        act_pos = ((anomaly_loc[str(id)] >= timestamp) & (anomaly_loc[str(id)] < (timestamp + pd.Timedelta(steps_ahead, unit='h')))).any()
        correct[counter] = act_pos

    for counter, i in enumerate(worst_args):
        tp = np.sum(correct[:counter+1])
        fp = np.sum(1-correct[:counter+1])
        fn = np.sum(correct[counter+1:])
        tn = np.sum(1-correct[counter+1:])
        if (tp + fn) == 0:
            tpr[counter] = 1
        else:
            tpr[counter] = tp / (tp + fn)
        if (fp + tn) == 0:
            fpr[counter] = 1
        else:
            fpr[counter] = fp / (fp + tn)
    idx = unique_last_indices(fpr)
    return tpr[idx], fpr[idx]

def pr_curve(score_df, method, anomaly_loc, k):
    # return True Positive Rate and False Positive Rate
    steps_ahead = int(k)
    if method == 'fisher': # higher is worse
        worst_args = np.argsort(score_df[k])[::-1]
    else: # lower is worse
        worst_args = np.argsort(score_df[k])[::1]

    # roc = np.zeros((worst_args.shape[0], 2))
    recall = np.zeros_like(worst_args, dtype=float)
    precision = np.zeros_like(worst_args, dtype=float)
    correct = np.zeros_like(worst_args) # split this into each timeseries
    for counter, i in enumerate(worst_args):
        id = score_df.iloc[i,0]
        timestamp = score_df.iloc[i,1]
        # act_pos: 1 for TP and 0 for FP
        act_pos = ((anomaly_loc[str(id)] >= timestamp) & (anomaly_loc[str(id)] < (timestamp + pd.Timedelta(steps_ahead, unit='h')))).any()
        correct[counter] = act_pos

    for counter, i in enumerate(worst_args):
        tp = np.sum(correct[:counter+1])
        fp = np.sum(1-correct[:counter+1])
        fn = np.sum(correct[counter+1:])
        tn = np.sum(1-correct[counter+1:])
        if (tp + fn) == 0:
            recall[counter] = 1
        else:
            recall[counter] = tp / (tp + fn)
        if (tp + fp) == 0:
            precision[counter] = 1
        else:
            precision[counter] = tp / (tp + fp)
    return precision, recall