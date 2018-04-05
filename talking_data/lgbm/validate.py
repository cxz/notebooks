"""


run 0:
    predictors = ['app', 'channel',  'device', 'ip',  'os',
       'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
       'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
       'count_ip_app_channel_hour', ]    
    
[10]    train's auc: 0.959527   h4's auc: 0.950359      h5's auc: 0.955799      h9's auc: 0.950592      h10's auc: 0.951147     h13's auc: 0.959001     h14's auc: 0.965224
[20]    train's auc: 0.965493   h4's auc: 0.95573       h5's auc: 0.960708      h9's auc: 0.956189      h10's auc: 0.955926     h13's auc: 0.963335     h14's auc: 0.969723
[30]    train's auc: 0.968663   h4's auc: 0.959477      h5's auc: 0.963473      h9's auc: 0.95933       h10's auc: 0.958812     h13's auc: 0.965431     h14's auc: 0.971698
[40]    train's auc: 0.970679   h4's auc: 0.960195      h5's auc: 0.964605      h9's auc: 0.961153      h10's auc: 0.960396     h13's auc: 0.967443     h14's auc: 0.97354
[50]    train's auc: 0.972083   h4's auc: 0.961635      h5's auc: 0.965603      h9's auc: 0.962678      h10's auc: 0.962149     h13's auc: 0.969177     h14's auc: 0.975069    

run 1:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device'
    ]
[10]    train's auc: 0.960376   h4's auc: 0.951176      h5's auc: 0.955174      h9's auc: 0.951835      h10's auc: 0.952416     h13's auc: 0.959729     h14's auc: 0.966331
[20]    train's auc: 0.968236   h4's auc: 0.95843       h5's auc: 0.962334      h9's auc: 0.958911      h10's auc: 0.95786      h13's auc: 0.965024     h14's auc: 0.971281
[30]    train's auc: 0.968936   h4's auc: 0.958543      h5's auc: 0.962655      h9's auc: 0.959729      h10's auc: 0.958554     h13's auc: 0.96558      h14's auc: 0.971882
[40]    train's auc: 0.970379   h4's auc: 0.959628      h5's auc: 0.963854      h9's auc: 0.96125       h10's auc: 0.9604       h13's auc: 0.967577     h14's auc: 0.973552
[50]    train's auc: 0.972173   h4's auc: 0.961355      h5's auc: 0.96519       h9's auc: 0.962771      h10's auc: 0.962334     h13's auc: 0.969326     h14's auc: 0.975364


run 2:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'mtbc_ip_device'
    ]    
[10]    train's auc: 0.960376   h4's auc: 0.951176      h5's auc: 0.955174      h9's auc: 0.951835      h10's auc: 0.952416     h13's auc: 0.959729     h14's auc: 0.966331
[20]    train's auc: 0.968236   h4's auc: 0.95843       h5's auc: 0.962334      h9's auc: 0.958911      h10's auc: 0.95786      h13's auc: 0.965024     h14's auc: 0.971281
[30]    train's auc: 0.968982   h4's auc: 0.958638      h5's auc: 0.962773      h9's auc: 0.959829      h10's auc: 0.958552     h13's auc: 0.965612     h14's auc: 0.971903
[40]    train's auc: 0.970509   h4's auc: 0.959796      h5's auc: 0.964033      h9's auc: 0.961324      h10's auc: 0.960471     h13's auc: 0.96752      h14's auc: 0.973588
[50]    train's auc: 0.972075   h4's auc: 0.96123       h5's auc: 0.965089      h9's auc: 0.962646      h10's auc: 0.962022     h13's auc: 0.969182     h14's auc: 0.975126    


run3:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device',
        'mtbc_ip_device',
    ]
[10]    train's auc: 0.958102   h4's auc: 0.946473      h5's auc: 0.950017      h9's auc: 0.949173      h10's auc: 0.950113     h13's auc: 0.958214     h14's auc: 0.965805
[20]    train's auc: 0.967429   h4's auc: 0.955952      h5's auc: 0.960304      h9's auc: 0.95774       h10's auc: 0.956678     h13's auc: 0.964073     h14's auc: 0.970948
[30]    train's auc: 0.968735   h4's auc: 0.958143      h5's auc: 0.96206       h9's auc: 0.959387      h10's auc: 0.958446     h13's auc: 0.965206     h14's auc: 0.971613
[40]    train's auc: 0.970618   h4's auc: 0.959987      h5's auc: 0.964039      h9's auc: 0.961314      h10's auc: 0.96054      h13's auc: 0.967791     h14's auc: 0.973817
[50]    train's auc: 0.972044   h4's auc: 0.96148       h5's auc: 0.965335      h9's auc: 0.962771      h10's auc: 0.962435     h13's auc: 0.96967      h14's auc: 0.975293    

run4:
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device', 'delta-ip-device-os-channel',
        'mtbc_ip_device', 'mtbc_ip_device_os_channel',
    ]

[10]    train's auc: 0.959531   h4's auc: 0.948324      h5's auc: 0.953005      h9's auc: 0.94927       h10's auc: 0.949501     h13's auc: 0.957499 h14's auc: 0.96379
[20]    train's auc: 0.966785   h4's auc: 0.955171      h5's auc: 0.959757      h9's auc: 0.956588      h10's auc: 0.955985     h13's auc: 0.963648 h14's auc: 0.970011
[30]    train's auc: 0.968445   h4's auc: 0.95791       h5's auc: 0.962198      h9's auc: 0.959082      h10's auc: 0.958242     h13's auc: 0.965146     h14's auc: 0.971263
[40]    train's auc: 0.970374   h4's auc: 0.959768      h5's auc: 0.963854      h9's auc: 0.960935      h10's auc: 0.960164     h13's auc: 0.967353     h14's auc: 0.973331
[50]    train's auc: 0.97187    h4's auc: 0.960909      h5's auc: 0.964621      h9's auc: 0.962567      h10's auc: 0.961973     h13's auc: 0.969207     h14's auc: 0.975188

run5:
    train using day==8
    predictors = [
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        'delta_ip_device'
    ]
    
[10]    train's auc: 0.965039   h4's auc: 0.953124      h5's auc: 0.956573      h9's auc: 0.95295       h10's auc: 0.953216     h13's auc: 0.961614     h14's auc: 0.967944
[20]    train's auc: 0.969389   h4's auc: 0.957767      h5's auc: 0.960935      h9's auc: 0.957844      h10's auc: 0.957219     h13's auc: 0.964847     h14's auc: 0.971389
[30]    train's auc: 0.970275   h4's auc: 0.958696      h5's auc: 0.961798      h9's auc: 0.959265      h10's auc: 0.95842      h13's auc: 0.965469     h14's auc: 0.971681
[40]    train's auc: 0.97198    h4's auc: 0.959727      h5's auc: 0.963475      h9's auc: 0.960922      h10's auc: 0.960104     h13's auc: 0.966989     h14's auc: 0.97309
[50]    train's auc: 0.97329    h4's auc: 0.96131       h5's auc: 0.964817      h9's auc: 0.962336      h10's auc: 0.961721     h13's auc: 0.968654     h14's auc: 0.974646

run6:
    train using day==8
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        #'delta_ip_device'
        'lhood_ip_day_in_test_hh',
        'lhood_app_channel_day_in_test_hh',
        'lhood_ip_hour',
        'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ]
[10]    train's auc: 0.970903   h4's auc: 0.959015      h5's auc: 0.964386      h9's auc: 0.961648      h10's auc: 0.961056     h13's auc: 0.966639     h14's auc: 0.972521
[20]    train's auc: 0.974574   h4's auc: 0.962869      h5's auc: 0.967562      h9's auc: 0.965182      h10's auc: 0.963935     h13's auc: 0.969606     h14's auc: 0.974935
[30]    train's auc: 0.975921   h4's auc: 0.964595      h5's auc: 0.96921       h9's auc: 0.966697      h10's auc: 0.96558      h13's auc: 0.970307     h14's auc: 0.975602
[40]    train's auc: 0.977953   h4's auc: 0.967292      h5's auc: 0.971751      h9's auc: 0.968905      h10's auc: 0.968073     h13's auc: 0.972673     h14's auc: 0.977673
[50]    train's auc: 0.979267   h4's auc: 0.967725      h5's auc: 0.972387      h9's auc: 0.97006       h10's auc: 0.969344     h13's auc: 0.973932     h14's auc: 0.978771
Did not meet early stopping. Best iteration is:
[50]    train's auc: 0.979267   h4's auc: 0.967725      h5's auc: 0.972387      h9's auc: 0.97006       h10's auc: 0.969344     h13's auc: 0.973932     h14's auc: 0.978771
[('lhood_ip_day_in_test_hh', 0), ('lhood_app_channel_day_in_test_hh', 0), ('lhood_ip_hour', 0), ('lhood_ip_device_hour', 1), ('hour', 6), ('count_ip_os_hour', 8), ('device', 9), ('count_ip_day_hour', 9), ('count_ip_device_hour', 9), ('count_ip_day_app_in_test_hh', 10), ('count_ip_day_device_in_test_hh', 14), ('count_ip_app_hour', 30), ('os', 33), ('count_ip_day_in_test_hh', 36), ('lhood_ip_app_hour', 38), ('lhood_ip_channel_hour', 39), ('channel', 60), ('app', 98)]
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.1,
              'num_leaves': 15, 
              'max_depth': -1, 
              #'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 250,  
              'subsample': 0.9,  # Subsample ratio of the training instance.
              'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
              'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              #'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              #'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 8,
              'verbose': 0,
              'scale_pos_weight':99.7, # because training data is extremely unbalanced 
    
"""


import os
import gc
import pickle
import logging
import datetime
import operator
from csv import DictReader
from functools import lru_cache
from collections import Counter
from datetime import datetime 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
import lightgbm as lgb

from tqdm import tqdm
import hashlib
import feather

import data2

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'

BASE_PATH = '../input'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test_v0.csv') # v0 with full rows


TMP = '/kaggle1/td-cache'
EARLY_STOP = 50
MAX_VAL_ROUNDS = 1000
MAX_ROUNDS = 650

      
def run(train_df, 
        val_dfs, 
        predictors, 
        target,
        categorical,
        max_rounds):
    
    print('train: ', len(train_df)) 
    
    use_validation = val_dfs is not None and len(val_dfs) > 0
    if use_validation:
        for val_df_name, val_df in val_dfs.items():
            print('{}: {}'.format(val_df_name, len(val_df)))

    params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.1,
              'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
              'max_depth': -1,  # -1 means no limit
              #'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 255,  
              'subsample': 0.9,  # Subsample ratio of the training instance.
              'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
              'bagging_freq': 1,
              #'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              #'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              #'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 4,
              'verbose': 0,
              'scale_pos_weight':99.7, # because training data is extremely unbalanced 
             }

    dtrain = lgb.Dataset(train_df[predictors].values, 
                         label=train_df[target].values,
                         feature_name=predictors,
                         categorical_feature=categorical)
    #del train_df
    #gc.collect()
    
    if use_validation:
        valid_sets = [dtrain]
        valid_names = ['train']
        
        for val_df_name, val_df in val_dfs.items():
            dvalid = lgb.Dataset(val_df[predictors].values,
                                 label=val_df[target].values,
                                 feature_name=predictors,
                                 categorical_feature=categorical)
            valid_sets.append(dvalid)
            valid_names.append(val_df_name)
        best_auc_metric = valid_names[0]
        early_stopping_rounds = 50
        prefix = 'cv'
    else:
        valid_sets = [dtrain]
        valid_names = ['train']
        best_auc_metric = 'train'
        early_stopping_rounds = None
        prefix = 'train'

    evals_results = {}
    
    model = lgb.train(params, 
                      dtrain, 
                      valid_sets=valid_sets, 
                      valid_names=valid_names,
                      evals_result=evals_results, 
                      num_boost_round=max_rounds,
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval=10, 
                      feval=None)
    
    best_auc = evals_results[best_auc_metric][params['metric']][model.best_iteration - 1]        
    out = '{}-{}-{}.pkl'.format(prefix, best_auc, datetime.now().strftime("%Y%m%d%H%M%S"))

    with open(out, 'wb') as f:
        pickle.dump([model, evals_results, params, predictors], f)

    print(sorted(dict(zip(model.feature_name(), model.feature_importance())).items(), 
                 key=operator.itemgetter(1)))


if __name__ == '__main__':
    trainval_df = data2.load('train')
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    
    predictors = [
        'app', 'channel',  'device', 'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 
        'count_ip_device_hour',
        #'count_ip_app_channel_hour', 
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        #'delta_ip_device'
        #'lhood_ip_day_in_test_hh',
        #'lhood_app_channel_day_in_test_hh',
        #'lhood_ip_hour',
        #'lhood_ip_device_hour',        
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ]
    
    # used to save memory only, as when building lgbm dataset we specify
    # columns to be used explicitly
    excluded = [
        'click_time',
        'ip',
        #'day',
        #'delta_ip_device', 
        #'delta_ip_app_device',
        #'delta_ip_app_device_os', 
        #'delta_ip_app_device_os_channel'
    ]
    
    for column in excluded:
        del trainval_df[column]
        
    gc.collect()
     
    # faster feedback
    train_df = trainval_df.loc[trainval_df.day == 8]    
    #train_df = trainval_df.iloc[:-VALID_ROWS]
    
    
    val_df = trainval_df.iloc[-VALID_ROWS:]
    
    val_dfs = {
        'h4':  val_df[val_df.hour ==  4],
        'h5':  val_df[val_df.hour ==  5],
        'h9':  val_df[val_df.hour ==  9],
        'h10': val_df[val_df.hour == 10],
        'h13': val_df[val_df.hour == 13],
        'h14': val_df[val_df.hour == 14]
    }
    
    run(train_df, val_dfs, predictors, target, categorical, 50)
