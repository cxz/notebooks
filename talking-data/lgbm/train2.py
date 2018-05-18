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
        seed,
        param_overrides={}):
    
    print('train: ', len(train_df)) 
    
    use_validation = val_dfs is not None and len(val_dfs) > 0
    if use_validation:
        for val_df_name, val_df in val_dfs.items():
            print('{}: {}'.format(val_df_name, len(val_df)))

    default_params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.1,
              'num_leaves': 63, 
              'max_depth': 5, 
              'max_bin': 255,  
              'bagging_freq': 0.8,
              'bagging_freq': 1,
              'bagging_seed': seed,
              'nthread': 8,
              'verbose': 0,
              'scale_pos_weight':100,
             }
    
    params = {**default_params, **param_overrides}

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
    
    print('train ', datetime.now())
    print('params: ', params)
    
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
    
    print('done ', datetime.now())


def run_train(days, iterations, seed):
    trainval_df = data2.load('train')
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'binip']
        
    excluded = [
        'click_time',
        'ip',
        'day'
    ]
    
    # i'm not convinced yet restricting training days is good
    train_cond = (trainval_df.day.isin(days)) & (trainval_df.hour.isin([4,5,9,10,13,14]))
    train_df = trainval_df[train_cond]
        
    for column in excluded:
        del trainval_df[column]
                
    gc.collect()
    
    predictors = list(sorted([c for c in trainval_df.columns if c not in ['is_attributed']]))
        
    val_dfs = None
    iterations = 57
    run(train_df, val_dfs, predictors, target, categorical, iterations, seed)
    
    
import fire    
if __name__ == '__main__':
    fire.Fire(run_train)