# Anomaly Detection using Foundation Models

## Prerequisites
To install required packages for the foundation models (not required if only evaluating) follow instructions on the original code repositories: ([MOIRAI](https://github.com/SalesforceAIResearch/uni2ts), [TimesFM](https://github.com/google-research/timesfm), [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama), [Chronos/Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting/tree/main))

AutoARIMA and N-BEATS can be installed using Nixtla's forecasting packages: `pip install statsforecast` and `pip install neuralforecast`

For evaluation the only required packages are the standard `pandas`, `numpy`, `matplotlib`, `sklearn`, and `scipy` packages

## Datasets
Datasets must be installed manually through their official sites with preprocessing steps (and anomaly generation steps) provided in the [data_prep](data_prep) folder

## Running Models
Use the `{model}_run.py` files in [model_runners](model_runners) directory to get model predictions on each dataset. Sample command: `python timesfm_run.py --dataset "data/amazon-google/y_amazon-google.csv" --save_dir "model_results/amazon-google/timesfm" --pred_length 48 --context 512 --forecast_date "2021-01-31 23:00:00" --quantiles "10,20,30,40,60,70,80,90"` (See Custom Datasets/Forecasts section on file naming scheme)

To evaluate the models (point accuracy/calibration) use `eval_run.py` which computes the various metrics used in the paper. Note: this code is not optimized and may take upwards of an hour if evaluating all models/datasets at once

## Anomaly Detection
Use [anomaly_plots.ipynb](anomaly_plots.ipynb) to visualize and evaluate model performance on detecting anomalies. 

## Custom Datasets and Forecasts
If you want to add your own Dataset it must be in the following format:

|       | ds        | y     | unique_id     |
| ----  | --------- | ----- |-------------- |
| Index | Timestamp | Value | Timeseries ID |

Save this file under `data/{dataset}/y_{dataset}.csv`

To use your own model forecasting results it must be in the following format:

|       |  unique_id        |  ds                                |  1                                               |  ...  |  k                                                                |
| ----- |  ---------        |  --                                |  -                                               |  ---  |  -                                                                |
| Index |  ID of Timeseries |  timestamp of last observed value  |  first predicted value given only observed data  |  ...  |  kth predicted value given k-1 forecasted data and rest observed  |

The code expects median predictions and [10, 20, 30, 40, 60, 70, 80, 90] quantile predictions and they should be saved as `model_results/{dataset}/{model}/median_preds.csv` and `model_results/{dataset}/{model}/quantile_{$$}_preds.csv`