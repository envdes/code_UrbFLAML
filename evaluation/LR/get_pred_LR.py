#!/usr/bin/env python
# coding: utf-8

"""
This script is used for predicting the testing data,
based on the features and members

members:
003 009
009 015
015 021
021 027
027 034

How to run this script?

$ qsub -X -I -l select=1:ncpus=36:mpiprocs=36 -l walltime=12:00:00 -q regular -A your_project
$ source /glade/work/zhonghua/miniconda3/bin/activate aws_urban
# consier CAM and surface, and member 003 to 009
$ python get_pred.py "["CAM","surf"]" "2006" "2015" "FLAML" "003" "009"  
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
import sys
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
print("AutoML version:", AutoML.__version__)

input_feature = sys.argv[1]
start_year = sys.argv[2]
end_year = sys.argv[3]
model_name = sys.argv[4]
member_start = sys.argv[5]
member_end = sys.argv[6]

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

urban_LE_nc_path = "/glade/scratch/zhonghua/urban_params/urban_LE/"
parquet_save_path = "/glade/scratch/zhonghua/urban_params/urban_LE_random_split/"
urban_surf_path = "/glade/scratch/zhonghua/urban_params/urban_surface.parquet.gzip"
model_save_path = "/glade/scratch/zhonghua/urban_params/urban_LE_random_split/"

feature_cat = list(map(str,input_feature.strip('[]').split(','))) # process from input to list
feature_join = "_".join(feature_cat) # for file name
feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features

def get_merge_member(start_year, end_year, parquet_save_path):
    df_tmp_ls = []
    for member_id in tqdm(range(3, 34)):
        member = (str(member_id).zfill(3))
        df_tmp_ls.append(pd.read_parquet(parquet_save_path + "train/" + member + "_"\
                            + start_year + "_" + end_year + ".parquet.gzip", engine="fastparquet"))
    return pd.concat(df_tmp_ls)

def get_test_data(member, start_year, end_year, urban_LE_nc_path, parquet_save_path):
    # convert the time to datetime format
    ds_urban_LE = xr.open_dataset(urban_LE_nc_path+member+"_"+start_year+"_"+end_year+".nc")
    ds_urban_LE = ds_urban_LE.assign_coords(time = ds_urban_LE.indexes['time'].to_datetimeindex())
    df = ds_urban_LE.to_dataframe()
    
    df_train = pd.read_parquet(parquet_save_path + "train/" + member + "_"\
                               + start_year + "_" + end_year + ".parquet.gzip", engine="fastparquet")  
    
    del ds_urban_LE
    gc.collect()
    
    # remove missing value based on urban temperature
    df_final = df[~np.isnan(df["TREFMXAV_U"])].reset_index()
    df_final["member"] = member
    
    # get testing data based on the saved training data
    df_test = df_final.drop(df_train.index)
    return df_test

def get_pred(start_year, end_year, model_name, member, urban_LE_nc_path, parquet_save_path, scaler, reg):
    # ====== get data =======
    df_test = get_test_data(member, start_year, end_year, urban_LE_nc_path, parquet_save_path)
    urban_surf = pd.read_parquet(urban_surf_path, engine="fastparquet").reset_index()
    test = pd.merge(df_test, urban_surf, on = ["lat","lon"], how = "inner")
    assert df_test.shape[0] == test.shape[0] #check if we merged successfully
    del df_test, urban_surf
    gc.collect()
    # ====== pred and save=======
    print(feature_ls)

    X_test_scaler = scaler.transform(test[feature_ls])
    X_test = sm.add_constant(X_test_scaler)

    test["y_pred"] = reg.predict(X_test)
    pred_save_loc = model_save_path + "eval/" + model_name + "/" + feature_join + "_"\
               + member + "_" + start_year + "_" + end_year + ".parquet.gzip"
    print("data loc:",pred_save_loc)
    test[["time","lat","lon","TREFMXAV_U","y_pred"]].to_parquet(pred_save_loc,
                                                                compression="gzip", engine="pyarrow")

# ====== load data ======
urban_LE = get_merge_member(start_year, end_year, parquet_save_path)
urban_surf = pd.read_parquet(urban_surf_path, engine="fastparquet").reset_index()
# merge data
df = pd.merge(urban_LE, urban_surf, on = ["lat","lon"], how = "inner")
# check if we merge the data successfully
assert urban_LE.shape[0] == df.shape[0]
del urban_LE, urban_surf
gc.collect()

# ====== get model ======
X_ori = df[feature_ls]
y = df[feature_dict["label"]]

scaler = StandardScaler()
scaler.fit(X_ori)
X_scaler = scaler.transform(X_ori)

X = sm.add_constant(X_scaler)
reg = sm.OLS(y, X).fit()

for member_id in range(int(member_start), int(member_end)):
    member = (str(member_id).zfill(3))
    get_pred(start_year, end_year, model_name, member, urban_LE_nc_path, parquet_save_path, scaler, reg)