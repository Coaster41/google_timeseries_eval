## Data Preparation

Use the provided notebooks with the included dataset download instructions to setup datasets for experiments.

Amazon-Google datasets are already included in [data](../data), but the raw datasets can be downloaded: (Google Local Reviews)[https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/] and (Amazon Reviews)[https://amazon-reviews-2023.github.io/]

## Generate Anomalies

Generate anomalies for any dataset using the [anomaly_generator.ipynb](anomaly_generator.ipynb) notebook. This randomly simulates anomalies by multiplying a section of the data with a strength factor. This will generate a dataset in the same format as the original dataset and saves a csv storing the dates of the simulated anomalies. 