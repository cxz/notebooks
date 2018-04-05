"""

Draft.

"""


import os
import gc
import pickle
import logging
import datetime
import operator
from datetime import datetime 

import pandas as pd
import numpy as np
import lightgbm as lgb

#from sklearn.model_selection import train_test_split 

import data2

        
def run(train_df, 
        val_dfs, 
        predictors, 
        target,
        categorical,
        max_rounds,
        seed):
    
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
              'num_leaves': 31, 
              'max_depth': -1, 
              'max_bin': 255,  
              'bagging_freq': 0.8,
              'bagging_freq': 1,
              'bagging_seed': seed,
              'nthread': 8,
              'verbose': 0,
              'scale_pos_weight':100,
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


def run_train(days, iterations, seed):
    trainval_df = data2.load('train')
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    
    predictors = [
        'app', 'channel',  'device', 'os', 'hour', 
        'count_ip_day_in_test_hh', 
        'count_ip_day_hour',
        'count_ip_os_hour', 
        'count_ip_app_hour', 
        'count_ip_device_hour',
        'count_ip_day_app_in_test_hh',
        'count_ip_day_device_in_test_hh',
        'lhood_ip_channel_hour',
        'lhood_ip_app_hour',        
    ] 
    
    excluded = [
        'click_time',
        'ip'
    ]
    
    for column in excluded:
        del trainval_df[column]
        
    gc.collect()
        
    # i'm not convinced yet restricting training days is good
    train_df = trainval_df.loc[trainval_df.day.isin(days)]
    val_dfs = None
    
    run(train_df, val_dfs, predictors, target, categorical, iterations, seed)
    
import fire    
if __name__ == '__main__':
    fire.Fire(run_train)