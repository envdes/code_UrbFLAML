#!/usr/bin/env python
# coding: utf-8

"""
This script is used for training the FLAML and save the model,
based on different feature combinations
We assign the time budget to be 21,600 s

How to run this script?

$ execcasper -A your_project -l gpu_type=v100 -l walltime=06:00:00 -l select=1:ncpus=18:mpiprocs=36:ngpus=1:mem=300GB
$ source /glade/work/zhonghua/miniconda3/bin/activate aws_urban
# consier CAM and surface
$ python train_random.py "["CAM","surf"]" "2006" "2015"   
# only consider CAM
$ python train_random.py "["CAM"]" "2006" "2015"
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xarray as xr
import gc
import pickle
from flaml import AutoML
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
print("AutoML version:", AutoML.__version__)

input_feature = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]
time_budget = 21600
estimator_list = ["lgbm", "xgboost", "rf"]

urban_surf_path = "/glade/scratch/zhonghua/urban_params/urban_surface.parquet.gzip"
parquet_save_path = "/glade/scratch/zhonghua/urban_params/urban_LE_random_split/"
model_save_path = "/glade/scratch/zhonghua/urban_params/urban_LE_scenarios/"

feature_dict = {
    "label":"TREFMXAV_U",
    "CAM": ['FLNS','FSNS','PRECT','PRSN','QBOT','TREFHT','UBOT','VBOT'],
    "surf":['CANYON_HWR','EM_IMPROAD','EM_PERROAD','EM_ROOF','EM_WALL', 
            'HT_ROOF','THICK_ROOF','THICK_WALL','T_BUILDING_MAX','T_BUILDING_MIN',
            'WTLUNIT_ROOF','WTROAD_PERV','NLEV_IMPROAD','PCT_URBAN',
            'ALB_IMPROAD','ALB_PERROAD','ALB_ROOF','ALB_WALL',
            'TK_ROOF','TK_WALL','CV_ROOF','CV_WALL',
            'TK_IMPROAD_0','CV_IMPROAD_0','TK_IMPROAD_1','CV_IMPROAD_1'],
    "loc":["lat","lon"]
}

feature_cat = list(map(str,input_feature.strip('[]').split(','))) # process from input to list
feature_join = "_".join(feature_cat) # for file name
feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features
print(feature_ls)

# ====== data prep ======
# for loading CESM-LE data
def get_merge_member(start_year, end_year, parquet_save_path):
    df_tmp_ls = []
    for member_id in range(3, 34):
        member = (str(member_id).zfill(3))
        df_tmp_ls.append(pd.read_parquet(parquet_save_path + "train/" + member + "_"\
                            + start_year + "_" + end_year + ".parquet.gzip", engine="fastparquet"))
    return pd.concat(df_tmp_ls)

# load data
urban_LE = get_merge_member(start_year, end_year, parquet_save_path)
urban_surf = pd.read_parquet(urban_surf_path, engine="fastparquet").reset_index()

# ===============
# ========= remove points from training data =========

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

dd = []
for p in pred_gridcell:
    dd.append(urban_LE[(np.abs(urban_LE["lat"]-pred_gridcell[p]["lat"])<0.0001) & 
                       (np.abs(urban_LE["lon"]-pred_gridcell[p]["lon"])<0.0001)])
    
urban_LE_new = urban_LE.drop(pd.concat(dd).index).copy()

# check if we removed points successfully
dd = []
for p in pred_gridcell:
    dd.append(urban_LE_new[(np.abs(urban_LE_new["lat"]-pred_gridcell[p]["lat"])<0.0001) & 
                           (np.abs(urban_LE_new["lon"]-pred_gridcell[p]["lon"])<0.0001)])

print("check if three points are still in the dataframe")
print(dd)

print("number of removed samples:", urban_LE.shape[0] - urban_LE_new.shape[0])

# merge data
train = pd.merge(urban_LE_new, urban_surf, on = ["lat","lon"], how = "inner")
# check if we merge the data successfully
assert urban_LE_new.shape[0] == train.shape[0]

del urban_LE, urban_LE_new, urban_surf
gc.collect()
# ===============

# ====== train model ======
automl = AutoML()
automl_settings = {
    "time_budget": time_budget,  # in seconds
    "estimator_list":estimator_list, # estimators
    "metric": 'rmse',
    "task": 'regression'
}
# fit the model
automl.fit(train[feature_ls], train[feature_dict["label"]], **automl_settings, verbose=-1)
print(automl.model.estimator)

# evaluate the final model performance
y_train = train[feature_dict["label"]]
y_pred = automl.predict(train[feature_ls])
print("training rmse:", mean_squared_error(y_true=y_train, y_pred=y_pred, squared=False))
print("training r2:", r2_score(y_true=y_train, y_pred=y_pred))
print("training mean_absolute_error:", mean_absolute_error(y_true = y_train, y_pred = y_pred))

# save the model
with open(model_save_path+start_year + "_" + end_year + "_" + feature_join + ".pkl", 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)