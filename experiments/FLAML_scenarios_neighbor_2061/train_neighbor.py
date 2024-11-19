#!/usr/bin/env python
# coding: utf-8

"""
This script is used for training the FLAML and save the model based on neighbor
We assign the time budget to be 7200 s

How to run this script?

$ execcasper -A your_project -l gpu_type=v100 -l walltime=06:00:00 -l select=1:ncpus=18:mpiprocs=36:ngpus=1:mem=300GB
$ source /glade/work/zhonghua/miniconda3/bin/activate aws_urban
$ python train_neighbor.py "1" "2061" "2070"   
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xarray as xr
import gc
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from flaml import AutoML
import sys
import pickle
print("AutoML version:", AutoML.__version__)

p = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]

time_budget = 7200
estimator_list = ["lgbm", "xgboost", "rf"]

urban_LE_nc_path = "/glade/scratch/zhonghua/urban_params/urban_LE/"
parquet_save_path = "/glade/scratch/zhonghua/urban_params/urban_LE_scenarios/"

fd = {
    "label":"TREFMXAV_U",
    "CAM": ['FLNS','FSNS','PRECT','PRSN','QBOT','TREFHT','UBOT','VBOT']
}

## known gridcell (lat, lon) <-> test gridcell (lat, lon)
known_gridcell = {
    "1": {"lat":32.51309, "lon":253.75},
    "2": {"lat":40.994766, "lon":277.5},
    "3": {"lat":40.994766, "lon":247.5}
}

pred_gridcell = {
    "1": {"lat":31.57068, "lon":253.75},
    "2": {"lat":41.937172, "lon":277.5},
    "3": {"lat":42.87958, "lon":247.5}
}

## train model
train = pd.read_parquet(parquet_save_path+"train_"+p+".parquet.gzip", engine="pyarrow")
feature_ls = fd["CAM"]
automl = AutoML()
automl_settings = {
    "time_budget": time_budget,  # in seconds
    "estimator_list":estimator_list, # estimators
    "metric": 'rmse',
    "task": 'regression'
}
# fit the model
automl.fit(train[feature_ls], train[fd["label"]], **automl_settings, verbose=-1)
print(automl.model.estimator)
with open(parquet_save_path+p+"_"+start_year + "_" + end_year + "_neighbor.pkl", 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

# load the model
model_path = parquet_save_path+p+"_"+start_year + "_" + end_year + ".pkl"
pred_save_loc = parquet_save_path+p+"_"+start_year + "_" + end_year + "_neighbor_pred.parquet.gzip"

with open(model_path, 'rb') as f:
    automl = pickle.load(f)
print(automl.model.estimator)

test = pd.read_parquet(parquet_save_path+"test_"+p+".parquet.gzip", engine="pyarrow").reset_index()
test["y_pred"] = automl.predict(test[feature_ls])
test[["time","lat","lon","TREFMXAV_U","y_pred"]].to_parquet(pred_save_loc,
                                                            compression="gzip", engine="pyarrow")

# check the performance across all predictions
y = test["TREFMXAV_U"]
y_pred = test["y_pred"]
print("root mean square error:", 
      mean_squared_error(y_true=y, y_pred=y_pred, squared=False))
print("mean_absolute_error:", mean_absolute_error(y_true=y, y_pred=y_pred))
print("r2:", r2_score(y_true=y, y_pred=y_pred))
