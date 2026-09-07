
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
import sklearn.metrics, math
from sklearn.utils import shuffle
from addAttribs_binned_wm import addAttribs
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost.sklearn import XGBRegressor

def XGBoosting_main(X_parameters,Y_parameters,predict_value):
 regr=XGBRegressor(n_estimators=200,min_child_weight=2,max_depth=20,\
                   gamma=0.6,subsample=1.0,colsample_bytree=1.0,\
                   alpha=3.0,reg_lambda=8,learning_rate=0.15)
 regr.fit(X_parameters, Y_parameters)
 predict_outcome = regr.predict(predict_value)
 predictions = {}
 predictions['predicted_value'] = predict_outcome
 return predict_outcome

def ExtraTreesRegressor_main(X_parameters,Y_parameters,predict_value):
 regr = ExtraTreesRegressor(n_estimators=250,random_state=1, max_features=5)
 regr.fit(X_parameters, Y_parameters)
 predict_outcome = regr.predict(predict_value)
 predictions = {}
 predictions['predicted_value'] = predict_outcome
 return predict_outcome

model ='XG'
####################### model path#######################
PROJECT_ROOT_DIR = '/home/class/zhanglongyi/PM2.5/projection/handover'
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR,"Ensemble_Output_ssp")
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

raw_data = pd.read_csv('/home/class/zhanglongyi/PM2.5/projection/handover/op_training_data_cesm2_historical_2x25.csv',low_memory=False)  
ml_raw = raw_data.filter(['lat','lon','year','mon','burned_area','w10','t2','rh','vevatc','vevavt','vlai','vtp',\
                               'primf','primn','secdf','secdn','pastr','urban','secmb','c3per','c4ann','c3ann','c4per'])#'sm','c3nfx','secmb'])

ml_raw.drop(ml_raw.columns[ml_raw.columns.str.contains('unnamed',case=False)],
        axis=1, inplace=True)
ml_raw.burned_area=ml_raw.burned_area*1e-6#km2
ml_clean = ml_raw.reset_index(drop=True).copy()
ml_clean = ml_clean.dropna()
ml_clean = ml_clean.reset_index(drop=True)
ml_extra_attribs = addAttribs(ml_clean)

####################### log #############################
ml_log = ml_extra_attribs.copy()
ml_log = ml_log[np.isfinite(ml_log).all(1)]
ml_log = ml_log.reset_index(drop=True)
print (ml_log)

####################### model path#######################
raw_data_ml1 = pd.read_csv('/home/class/zhanglongyi/PM2.5/projection/handover/op_training_data_cesm2_ssp245_2x25.csv',low_memory=False)

ml_raw1 = raw_data_ml1.filter(['lat','lon','year','mon','w10','t2','rh','vevatc','vevavt','vlai','vtp',\
                               'primf','primn','secdf','secdn','pastr','urban','secmb','c3per','c4ann','c3ann','c4per','burned_area'])

ml_raw1.drop(ml_raw1.columns[ml_raw1.columns.str.contains('unnamed',case=False)],
        axis=1, inplace=True)
ml_clean1 = ml_raw1.reset_index(drop=True).copy()
ml_clean1 = ml_clean1.dropna()
ml_clean1 = ml_clean1.reset_index(drop=True)
ml_extra_attribs1 = addAttribs(ml_clean1)

#######################log #############################
ml_log1 = ml_extra_attribs1.copy()
ml_log1 = ml_log1[np.isfinite(ml_log1).all(1)]
ml_log1 = ml_log1.reset_index(drop=True)
print (ml_log1)

#######################   get index for external test data   #######################
ml_log_extra_attribs = ml_log.copy()
ml_log_extra_attribs1 = ml_log1.copy()
external_test = ml_log_extra_attribs1
ml_train_set = ml_log_extra_attribs
print("All data points : ", len(ml_train_set) + + len(external_test))
print("training data points : ", len(ml_train_set))
print("external validation data points: ", len(external_test))

####################### test parameter combinations ###############
rm_cols1 = ["mon","lat","lon",'secdf','secdn','pastr','vevavt','secmb','primf','primn','urban','c3per','c4ann','c3ann','c4per']#,'sm'
val_col  = ['w10','t2','rh','vevatc','vlai','vtp','year']

numCom = 7  
combs = combinations(val_col, (len(val_col)-numCom))
nidx = 5
for rm_cols2 in combs:
    rm_cols =  rm_cols1 + list(rm_cols2)
    train_features = ml_train_set.copy()
    external_features = external_test.copy()

    [train_features.pop(x) for x in rm_cols]
    [external_features.pop(x) for x in rm_cols]
        
    train_labels = train_features.pop('burned_area')
    extest_labels  = external_features.pop('burned_area')
    for col in external_features.columns:
        print(col)
        
    if len(train_features.columns)-nidx > 0:
        train_norm = train_features[train_features.columns[0:len(external_features.columns)-nidx]]
        extest_norm = external_features[external_features.columns[0:len(external_features.columns)-nidx]]        
        
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_norm)
        
        x_train_norm = min_max_scaler.transform(train_norm)
        training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
        train_features.update(training_norm_col)
        
        x_extest_norm = min_max_scaler.transform(extest_norm)
        extest_norm_col = pd.DataFrame(x_extest_norm, index=extest_norm.index, columns=extest_norm.columns)
        external_features.update(extest_norm_col)
      
        
 
    if model =='XG':
     extest_predictions = XGBoosting_main(train_features, pd.DataFrame(train_labels),external_features).flatten()
       
    # calculate statistics
    xtest_MAE = sklearn.metrics.mean_absolute_error(extest_labels,extest_predictions)
    xtest_RMSE = math.sqrt(sklearn.metrics.mean_squared_error(extest_labels,extest_predictions))
    xtest_R    = pd.Series(extest_labels).corr(pd.Series(extest_predictions))
                               
    columnsNamesArr = external_features.columns.values
    trash = (list(columnsNamesArr))
    junk = ['latlon1', 'latlon2', 'latlon3']
    predictors = [item for item in trash if item not in junk]
    hh = '+'.join(predictors)
    op1,op2=extest_labels,extest_predictions
    oplat,oplon=external_test['lat'],external_test['lon']
    opd1,opd2=external_test['year'],external_test['mon']
    op1=external_test['burned_area']
    resultall=pd.DataFrame([oplat,oplon,opd1,opd2,op1,op2])
    resultall=resultall.T;resultall.columns=['lat','lon','year','mon','GFED','prd_bd']
    
    resultall.to_csv(MODEL_PATH + '/'+str(model)+hh+'_cesm2_ssp245_2x25.csv',sep=',')

print("----------------------------------done!----------------------------------------------------")  
