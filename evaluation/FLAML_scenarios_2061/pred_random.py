#!/usr/bin/env python
# coding: utf-8

"""
This script is used for applying the FLAML to pred location and save the data,
based on different feature combinations

How to run this script?

$ execcasper -A your_project -l gpu_type=v100 -l walltime=06:00:00 -l select=1:ncpus=18:mpiprocs=36:ngpus=1:mem=300GB
$ source /glade/work/zhonghua/miniconda3/bin/activate aws_urban
# consier CAM and surface
$ python pred_random.py "["CAM","surf"]" "2061" "2070"   
# only consider CAM
$ python pred_random.py "["CAM"]" "2061" "2070"
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
feature_cat = list(map(str,input_feature.strip('[]').split(','))) # process from input to list
start_year = sys.argv[2]
end_year = sys.argv[3]

model_path = "/glade/scratch/zhonghua/urban_params/urban_LE_scenarios/"
data_path = "/glade/scratch/zhonghua/urban_params/urban_LE_scenarios/"
pred_path = "/glade/scratch/zhonghua/urban_params/urban_LE_scenarios/pred/"

urban_surf_path = "/glade/scratch/zhonghua/urban_params/urban_surface.parquet.gzip"
urban_surf = pd.read_parquet(urban_surf_path, engine="fastparquet").reset_index()

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

def apply_model(feature_cat, start_year, end_year, model_path, data_path, pred_path, feature_dict):
    feature_join = "_".join(feature_cat) # for file name
    feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features
    
    model_path_p = model_path + start_year + "_" + end_year + "_" + feature_join+".pkl"
    with open(model_path_p, 'rb') as f:
        automl = pickle.load(f)
        print(automl.model.estimator)
    
    for p in ["1", "2", "3"]:
        pred_path_p = pred_path + p + "_" + start_year + "_" + end_year + "_pred_" + feature_join + ".parquet.gzip"
        test_data = pd.read_parquet(data_path+"test_"+p+".parquet.gzip", engine="pyarrow").reset_index()
        # merge data
        test = pd.merge(test_data, urban_surf, on = ["lat","lon"], how = "inner")
        # check if we merge the data successfully
        assert test_data.shape[0] == test.shape[0]
        del test_data
        gc.collect()

        test["y_pred"] = automl.predict(test[feature_ls])
        test[["time","lat","lon","TREFMXAV_U","y_pred","member"]].to_parquet(pred_path_p,
                                                                             compression="gzip", engine="pyarrow")
        del test
        gc.collect()

apply_model(feature_cat, start_year, end_year, model_path, data_path, pred_path, feature_dict)