import numpy as np
import pandas as pd

UNIT_DICT = {"amazon-google": "H", "m5": "D", "glucose": "T", "meditation": "S", "amazon": "H", "patents": "M", "police": "D",
             "amazon-google_anomaly": "H", "m5_anomaly": "D", "glucose_anomaly": "T", "meditation_anomaly": "S", 
             "amazon_anomaly": "H", "patents_anomaly": "M", "police_anomaly": "D",
             "m5_small": "D", "meditation_small": "S", "glucose_small": "T", "patents_small": "M", "police_small": "D"}
UNIT_NUM_DICT = {"amazon-google": 1, "m5": 1, "glucose": 5, "meditation": 1, "amazon": 1, "patents": 1, "police": 1,
                 "amazon-google_anomaly": 1, "m5_anomaly": 1, "glucose_anomaly": 5, "meditation_anomaly": 1, 
                 "amazon_anomaly": 1, "patents_anomaly": 1, "police_anomaly": 1,
                 "m5_small": 1, "meditation_small": 1, "glucose_small": 5, "patents_small": 1, "police_small": 1}

def load_results(dataset, model):
    unit = UNIT_DICT[dataset]
    unit_num = UNIT_NUM_DICT[dataset]
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    results_fn = f"model_results/{dataset}/{model}/median_preds.csv"
    data_fn = f"data/{dataset}/y_{dataset}.csv"
    results_df = pd.read_csv(results_fn, index_col=0, parse_dates=['ds'])
    data_df = pd.read_csv(data_fn, index_col=0, parse_dates=['ds'])
    if unit == 'M':
        freq_delta = pd.DateOffset(months=unit_num)
    else:
        freq_delta = pd.Timedelta(unit_num, unit)
        
    quantiles_df = []
    for quantile in quantiles:
        quantile_fn = f"model_results/{dataset}/{model}/quantile_{round(quantile * 100)}_preds.csv"
        quantiles_df.append(pd.read_csv(quantile_fn, index_col=0, parse_dates=['ds']))
    quantiles_df.insert(len(quantiles_df)//2, results_df)
    quantiles.insert(len(quantiles)//2, 0.5)
    quantiles_dict = dict(zip(quantiles, quantiles_df))
    return data_df, results_df, freq_delta, quantiles_dict

def mae(results_df, data_df, freq_delta):
    # Mean Absolute Error
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    return np.mean(mae_arr), np.array(mae_arr)


def mase(results_df, data_df, freq_delta):
    # Mean Absolute Scaled Error
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] += freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))
    return np.mean(mae_arr) / mae_n, np.array(mae_arr) / mae_n


def tce(lower_df, upper_df, data_df, freq_delta, confidence):    
    # Tailed Calibration Error
    pred_length = int(lower_df.columns[-1])
    outside_ratio = (1-confidence)/2
    tce_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_upper_outside = np.mean(merged_upper['y'] > merged_upper[str(h)])
        mean_lower_outside = np.mean(merged_lower['y'] < merged_lower[str(h)])
        tce_arr.append(abs(outside_ratio - mean_upper_outside) + abs(outside_ratio - mean_lower_outside))
    return np.mean(tce_arr)/2, np.array(tce_arr)/2

def pce(quantiles_dict, data_df, freq_delta):    
    # Probabilistic Calibration Error
    pce_arr = []
    for quantile, quantile_df in quantiles_dict.items():
        pred_length = int(quantile_df.columns[-1])
        ce_arr = []
        for h in range(1,pred_length+1):
            shift_quantile = quantile_df[['ds', 'unique_id', str(h)]]
            shift_quantile.loc[:,'ds'] += freq_delta * h
            merged_quantile = pd.merge(data_df, shift_quantile, on=['unique_id', 'ds'], how='inner')
            mean_quantile_inside = np.mean(merged_quantile['y'] <= merged_quantile[str(h)])
            ce_arr.append(abs(quantile - mean_quantile_inside))
        pce_arr.append(ce_arr)
    pce_arr = np.mean(pce_arr, axis=0)
    return np.mean(pce_arr), pce_arr

def wql(quantiles_dict, data_df, freq_delta):
    # Weighted Quantile Loss
    ql_arr = []
    for quantile, quantile_df in quantiles_dict.items():
        quantile_ql_arr = []
        pred_length = int(quantile_df.columns[-1])
        for h in range(1,pred_length+1):
            shift_results = quantile_df[['ds', 'unique_id', str(h)]]
            shift_results.loc[:,'ds'] += freq_delta * h
            merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
            quantile_loss = np.sum((2*(1-quantile)*(merged_results[str(h)] - merged_results['y'])*(merged_results[str(h)] >= merged_results['y'])) \
                            + (2*(quantile)*(merged_results['y'] - merged_results[str(h)])*(merged_results[str(h)] < merged_results['y'])))
            quantile_ql_arr.append(quantile_loss)
        ql_arr.append(quantile_ql_arr)

    scale = np.sum(merged_results['y'])
    wql_arr = np.array(ql_arr) / scale
    return np.sum(wql_arr), np.sum(wql_arr, axis=0)


def msis(lower_df, upper_df, data_df, freq_delta, confidence):  
    # Mean Scaled Interval Score  
    pred_length = int(lower_df.columns[-1])
    mis_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_interval_score = np.mean( (merged_upper[str(h)] - merged_lower[str(h)]) \
                                      + 2/(1-confidence) * (merged_lower[str(h)] - merged_lower['y']) * (merged_lower['y'] < merged_lower[str(h)]) \
                                      + 2/(1-confidence) * (merged_upper['y'] - merged_upper[str(h)]) * (merged_upper['y'] > merged_upper[str(h)]) )
        mis_arr.append(mean_interval_score)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] += freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))

    return np.mean(mis_arr) / mae_n, np.array(mis_arr) / mae_n

def msiw(lower_df, upper_df, data_df):
    # Mean Scaled Interval Width - Scaled by MAE
    pred_length = int(lower_df.columns[-1])
    mis_arr = []
    for h in range(1,pred_length+1):
        mis = np.mean((upper_df[str(h)] - lower_df[str(h)]))
        mis_arr.append(mis)
    
    # naive mae
    scale = np.mean(pd.merge(data_df, upper_df, on=['unique_id', 'ds'], how='inner')['y'])
    return np.mean(mis_arr) / scale, np.array(mis_arr) / scale

def siw_mean(quantiles_dict, data_df, confidences):
    # Scaled Interval Width (Mean) - Scaled by True Quantiles
    pred_length = int(quantiles_dict[0.5].columns[-1])
    merged_results = pd.merge(data_df, quantiles_dict[0.5][['ds', 'unique_id', '1']], on=['unique_id', 'ds'], how='inner')
    siw_arr = []
    h = [str(i) for i in range(1,pred_length+1)]
    for confidence in confidences:
        if confidence == 0:
            continue
        upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
        lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
        iqr = upper_df[h] - lower_df[h]
        iqr /= (np.quantile(merged_results['y'], q=round(0.5 + confidence/2, 1)) \
                - np.quantile(merged_results['y'], q=round(0.5 - confidence/2, 1)))
        siw_arr.append(np.mean(iqr, axis=0))
    return np.mean(siw_arr), np.mean(siw_arr, axis=0)


def siw(lower_df, upper_df, data_df, confidence):
    # Scaled Interval Width - Scaled by True Quantiles
    pred_length = int(upper_df.columns[-1])
    merged_results = pd.merge(data_df, upper_df[['ds', 'unique_id', '1']], on=['unique_id', 'ds'], how='inner')
    h = [str(i) for i in range(1,pred_length+1)]
    iqr = upper_df[h] - lower_df[h]
    iqr /= (np.quantile(merged_results['y'], q=round(0.5 + confidence/2, 1)) \
            - np.quantile(merged_results['y'], q=round(0.5 - confidence/2, 1)))
    siw_arr = np.mean(iqr, axis=0)
    return np.mean(siw_arr), siw_arr.to_numpy().flatten()

def cce(lower_df, upper_df, data_df, freq_delta, confidence):    
    # Centered Calibration Error
    pred_length = int(lower_df.columns[-1])
    cce_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        middle = (merged_upper['y'] <= merged_upper[str(h)]) & (merged_lower['y'] >= merged_lower[str(h)])
        mean_middle = np.mean(middle)
        cce_arr.append(confidence - mean_middle)
    return np.mean(cce_arr), np.array(cce_arr)

def stce(quantile_df, data_df, freq_delta, quantile):
    # Single Tailed Calibration Error
    pred_length = int(quantile_df.columns[-1])
    tce_arr = []
    for h in range(1,pred_length+1):
        shift_quantile = quantile_df[['ds', 'unique_id', str(h)]]
        shift_quantile.loc[:,'ds'] += freq_delta * h
        merged_quantile = pd.merge(data_df, shift_quantile, on=['unique_id', 'ds'], how='inner')
        if quantile >= 0.5:
            mean_quantile_outside = np.mean(merged_quantile['y'] > merged_quantile[str(h)])
            tce_arr.append(mean_quantile_outside - (1-quantile)) # over confidence: outside is greater
        else:
            mean_quantile_outside = np.mean(merged_quantile['y'] < merged_quantile[str(h)])
            tce_arr.append(mean_quantile_outside - (quantile)) # over confidence: outside is greater
    return np.mean(tce_arr), np.array(tce_arr)
