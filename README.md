# Code of paper "*Leveraging Automated Machine Learning (AutoML) for Urban Climate Emulation*"

## Introduction
---

This repository is code of the manuscript "*Leveraging Automated Machine Learning (AutoML) for Urban Climate Emulation*".

The objectives of this project are:

- develop location-independent machine learning emulators for the urban daily maximum of average 2-m temperature
- apply automated machine learning (AutoML) to emulation tasks
- propose a feature importance analysis framework for the AutoML models

## Scripts
---

### 1 data processing

folder: ./data_prepare
machine: NCAR cheyenne supercomputer

| task| code | note |
| :-----| ----: | :----: |
|urban surface data preview| [0_CESM1_surface_data_EDA.ipynb](./data_prepare/0_CESM1_surface_data_EDA.ipynb)|
|urban surface data prepare| [1_urban_surface_data_prep.ipynb](./data_prepare/1_urban_surface_data_prep.ipynb)|
|forcing data processing| [2_save_CAM_CLMU_as_xr_dataset](./data_prepare/2_save_CAM_CLMU_as_xr_dataset)|
|urban surface data processing| [3_save_urban_surface_as_parquet.ipynb](./data_prepare/3_save_urban_surface_as_parquet.ipynb)|
|training and testing data processing| [4_random_split_train_test_data.ipynb](./data_prepare/4_random_split_train_test_data.ipynb)|

### 2 model training

folder: ./experiments
machine: NCAR cheyenne supercomputer

| task| folder | note |
| :-----| ----: | :----: |
|automate machine learning| [FLAML](./experiments/FLAML)| scripts for training 2006 data|
|automate machine learning| [FLAML_2061](./experiments/FLAML_2061)| scripts for training 2061 data|
|XGBoost| [xgboost](./experiments/xgboost)| scripts for training 2006 data|
|XGBoost| [xgboost_2061](./experiments/xgboost_2061)| scripts for training 2061 data|
|linear regression| [linear_regression_benchmark_2006.ipynb](./experiments/linear_regression_benchmark_2006.ipynb)| notebook for training 2006 data|
|linear regression| [linear_regression_benchmark_2061.ipynb](./experiments/linear_regression_benchmark_2061.ipynb)| notebook for training 2061 data|

### 3 model evaluation

folder: ./evaluation
machine: NCAR cheyenne supercomputer

| task| folder | note |
| :-----| ----: | :----: |
|automate machine learning| [FLAML_2006](./evaluation/FLAML_2006)| scripts for testing 2006 model using 2006 data|
|automate machine learning| [FLAML_2061](./evaluation/FLAML_2061)| scripts for testing 2061 model using 2061 data|
|automate machine learning| [FLAML_climate](./evaluation/FLAML_climate)| scripts for testing 2006 model using 2061 data|
|XGBoost| [xgboost_2006](./evaluation/xgboost_2006)| scripts for testing 2006 model using 2006 data|
|XGBoost| [xgboost_2061](./evaluation/xgboost_2061)| scripts for testing 2061 model using 2061 data|
|XGBoost| [xgboost_climate](./evaluation/xgboost_climate)| scripts for testing 2006 model using 2061 data|
|linear regression| [LR](./evaluation/LR)| scripts for testing all|

### 4 scenario experiments

machine: NCAR cheyenne supercomputer

this experiments are the location exploration part of paper, all experiments using automate machine learning for emulation, and data is 2061. The assessment used three pairs of urban grid cells from the dataset. Each pair consists of neighboring grid cells located in different regions but are close in space. The selected grid cells data is used to testing task.


| task| folder | note |
| :-----| ----: | :----: |
| data processing| [5_select_10_years_data_prep.ipynb](./data_prepare/5_select_10_years_data_prep.ipynb)|data that removed the selected grid cells|
| data processing| [5_select_10_years_data_prep.ipynb](./data_prepare/5_select_10_years_data_prep.ipynb)|data that removed the selected grid cells|
|training| [FLAML_scenarios_2061](./experiments/FLAML_scenarios_2061)| scripts for global model trained without the selected grid cell data|
|training| [FLAML_scenarios_2061](./evaluation/FLAML_scenarios_2061)| scripts for global model trained without the selected grid cell data|
|training and testing| [FLAML_scenarios_itself_subsamples_2061](./experiments/FLAML_scenarios_itself_subsamples_2061)| scripts for model trained only using the selected grid cell data|
|training and testing| [FLAML_scenarios_neighbor_2061](./experiments/FLAML_scenarios_neighbor_2061)| scripts for model trained only using the neighbor of selected grid cell|

### 5 feature importance

machine: UoM csf3 supercomputer

We developed a unified ranking score framework tailored for the AutoML tasks, combining tree-based feature importance, permutation feature importance, and SHAP values, to evaluate the relative importance of different features. This ensemble method improves the robustness of our analysis by not relying solely on a single type of feature importance evaluation.

| task| folder | note |
| :-----| ----: | :----: |
| tree importance| [tree_importance.ipynb](./feature_importance/tree_importance.ipynb)||
| permutation importance| [permutation_importance.py](./feature_importance/permutation_importance.py)| run in csf3|
| shap importance| [shap_importance.py](./feature_importance/shap_importance.py)| run in csf3|


### 6 Visualization

| task| folder | note |
| :-----| ----: | :----: |
| workflow and experiment design| [AutomL.pptx](./visualization/AutomL.pptx)|fig 1 and 2|
| urban location | [urban_location_visualization.ipynb](./visualization/urban_location_visualization.ipynb)| fig 3 and figs 1|
| model performance| [model_performance_visualization.ipynb](./visualization/model_performance_visualization.ipynb)|fig 4-5 and figs 5-6|
| location exploration | [loc_exploration.ipynb](./visualization/loc_exploration.ipynb)| fig 6|
| feature importance | [feature_importance.ipynb](./visualization/feature_importance.ipynb)| fig 7-8|
| feature distribution | [feature_dist.ipynb](./visualization/feature_dist.ipynb)| figs 2-3|
| tree-based feature importance | [figs_tree.ipynb](./visualization/figs_tree.ipynb)| figs 7-8|
| permutation feature importance | [figs_permutation.ipynb](./visualization/figs_permutation.ipynb)| figs 9-10|
| shap feature importance | [figs_shap.ipynb](./visualization/figs_shap.ipynb)| figs 11-12|

## Acknowledgments

We would like to acknowledge high-performance computing support from Cheyenne (https://doi.org/10.5065/D6RX99HX) provided by NCAR’s Computational and Information Systems Laboratory, sponsored by the National Science Foundation, the CSF (aka Danzek) is a High Performance Computing (HPC) cluster at the University of Manchester (https://ri.itservices.manchester.ac.uk/csf3/), managed by IT Services for the use of University academics, post-doctoral assistants and post-graduates to conduct academic research. 