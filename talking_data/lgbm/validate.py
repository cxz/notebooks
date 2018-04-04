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
              'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
              'max_depth': 5,  # -1 means no limit
              'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 100,  # Number of bucketed bin for feature values
              'subsample': 0.9,  # Subsample ratio of the training instance.
              'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
              'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 8,
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
    
    best_auc = evals_results[best_auc_metric][LGB_PARAMS['metric']][m.best_iteration - 1]        
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
        'app', 'channel',  'device', 'ip',  'os',
        'hour', 'count_ip_day_in_test_hh', 'count_ip_day_hour',
        'count_ip_os_hour', 'count_ip_app_hour', 'count_ip_device_hour',
        'count_ip_app_channel_hour', 
        #'delta_ip_device',
        'mtbc_ip_device',
    ]
    
    # used to save memory only, as when building lgbm dataset we specify
    # columns to be used explicitly
    excluded = [
        'click_time',
        'day',
        'delta_ip_device', 
        #'delta_ip_app_device',
        #'delta_ip_app_device_os', 
        #'delta_ip_app_device_os_channel'
    ]
    
    for column in excluded:
        del trainval_df[column]
        
    gc.collect()
        
    train_df = trainval_df.iloc[:-VALID_ROWS]
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
