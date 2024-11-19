import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import xarray as xr
import gc
import pickle
from tqdm import tqdm
from flaml import AutoML
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
print("AutoML version:", AutoML.__version__)
from sklearn.inspection import permutation_importance



# ====== data prep ======
# for loading CESM-LE data
def get_merge_member(start_year, end_year, parquet_save_path):
    df_tmp_ls = []
    for member_id in range(3, 34):
        member = (str(member_id).zfill(3))
        df_tmp_ls.append(pd.read_parquet(parquet_save_path + "train/" + member + "_"\
                            + start_year + "_" + end_year + ".parquet.gzip", engine="fastparquet"))
    return pd.concat(df_tmp_ls)


def get_trainning_set(input_feature, start_year, end_year):

    urban_surf_path = "/mnt/eps01-rds/zheng_medal/zhonghua/UrbanClimatePaper/urban_params/urban_surface.parquet.gzip"
    parquet_save_path = "/mnt/eps01-rds/zheng_medal/zhonghua/UrbanClimatePaper/urban_params/urban_LE_random_split/"
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

    feature_cat =  input_feature # process from input to list
    feature_join = "_".join(feature_cat) # for file name
    feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features
    print(feature_ls)

    # load data
    urban_LE = get_merge_member(start_year, end_year, parquet_save_path)
    urban_surf = pd.read_parquet(urban_surf_path, engine="fastparquet").reset_index()
    # merge data
    train = pd.merge(urban_LE, urban_surf, on = ["lat","lon"], how = "inner")
    # check if we merge the data successfully
    assert urban_LE.shape[0] == train.shape[0]
    del urban_LE, urban_surf
    gc.collect()

    return train


def get_importance(model,training_X, training_y):
    scoring = ['neg_mean_squared_error']#'r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error'
    r = permutation_importance(model, training_X, training_y, n_repeats=30, random_state=0, scoring=scoring)
    return r[scoring[0]].importances_mean

def get_model_ranking_score(feature_cat, model_name, start_year, training_X, training_y):
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
    
    end_year = str(int(start_year)+9)

    scf3home = '/mnt/eps01-rds/zheng_medal/zhonghua/UrbanClimatePaper/urban_params'
    parquet_save_path = scf3home + "/urban_LE_random_split/"

    feature_join = "_".join(feature_cat) # for file name
    feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features
    model_pkl_save_path = parquet_save_path+model_name+"/"+start_year+"_"+end_year+"_"+feature_join+".pkl"
    print("model location:", model_pkl_save_path)
    
    with open(model_pkl_save_path, 'rb') as f:
        automl = pickle.load(f)
    #     print(automl.model.estimator)
        model = automl.model.estimator
        print(automl.model.estimator)
        #best_model_name = automl._best_estimator

    shap_importance = get_importance(model,training_X, training_y)

    df_feature = pd.DataFrame({"features":feature_ls,
                                   "importance":shap_importance})

    colname = model_name+"_"+start_year
    
    df_feature[colname] = df_feature["importance"].rank(ascending=True)
   
    return model_pkl_save_path, df_feature[["features",colname]]

dd = {
    "cam":['FLNS','FSNS','PRECT','PRSN','QBOT','TREFHT','UBOT','VBOT'],
    "loc":["lat","lon"],
    "morphological":['CANYON_HWR','HT_ROOF','THICK_ROOF','THICK_WALL','WTLUNIT_ROOF','WTROAD_PERV','PCT_URBAN'],
    "radiative": ['EM_IMPROAD','EM_PERROAD','EM_ROOF','EM_WALL', 'ALB_IMPROAD','ALB_PERROAD','ALB_ROOF','ALB_WALL'],
    "thermal":[
        'T_BUILDING_MAX','T_BUILDING_MIN',            
        'TK_ROOF','TK_WALL','CV_ROOF','CV_WALL',
        'TK_IMPROAD_0','CV_IMPROAD_0','TK_IMPROAD_1','CV_IMPROAD_1',
        'NLEV_IMPROAD']
}

dd_cat = {}
for key in dd:
    for feature in dd[key]:
        dd_cat[feature] = key 


feature_cat=["CAM","surf","loc"]

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
feature_ls = sum([feature_dict[k] for k in feature_cat],[]) # list of features

for start_year, end_year in zip(["2006", "2061"], ["2015", "2070"]):
    training_set = get_trainning_set(feature_cat, start_year, end_year)
    training_set = training_set.groupby(['lat','lon']).sample(frac=0.01, random_state=1).reset_index()
    training_X = training_set[feature_ls]
    training_y = training_set[feature_dict['label']]
    print(training_set.head())
    for model_name in ["FLAML", "xgboost"]: #"lgbm", 
        print(start_year)
        print(model_name)
        model_pkl_save_path, df_tmp = get_model_ranking_score(feature_cat, model_name, start_year, training_X, training_y)
        if model_name == "FLAML" and start_year == "2006":
            df = df_tmp.copy()
        else:
            df = df.merge(df_tmp, on = "features", how = "outer")
        del df_tmp
        gc.collect()

df["category"] = df["features"].map(dd_cat)
df.to_csv("./feature_importance_permutation.csv",index=False)
