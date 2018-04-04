"""

Draft.

validation day==9, hour==4, baseline.
[395]   train's auc: 0.984294   valid's auc: 0.963007

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

import data2

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'

BASE_PATH = '../input'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test_v0.csv') # v0 with full rows

import feather

TMP = '/kaggle1/td-cache'
EARLY_STOP = 50
MAX_VAL_ROUNDS = 1000
MAX_ROUNDS = 650

target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour']

print('loading base ', datetime.now())
trainval_df = feather.read_dataframe(os.path.join(TMP, 'train_base.feather'))
predictors = categorical

for group in [
        ['ip', 'day', 'in_test_hh'],
        ['ip', 'day', 'hour'],
        ['ip', 'os', 'hour'],
        ['ip', 'app',  'hour'],
        ['ip', 'device', 'hour'],
        ['ip', 'app', 'channel', 'hour']
]:
    column = 'count_{}'.format('_'.join(group))
    predictors.append(column)

if False:
    for group in [
            ['ip', 'device'],
            ['ip', 'app', 'device'],
            ['ip', 'app', 'device', 'os'],
            ['ip', 'app', 'device', 'os', 'channel'],
    ]:
        column = 'delta_{}'.format('_'.join(group))
        predictors.append(column)
        
def trainval_random_split(trainval_df, test_size=0.1, random_state=50):
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True)
    return train_df, val_df

def trainval_day9_hour4_split(trainval_df):
    val_idx = (trainval_df.day==9)&(trainval_df.hour==4)
    val_df = trainval_df[val_idx]
    train_df = trainval_df[~val_idx]
    train_df = train_df.iloc[109903891:] #skiprows
    return train_df, val_df
        
def run(train_df, val_df):
    print('train: ', len(train_df), 'val: ', len(val_df))
    use_validation = val_df is not None and len(val_df) > 0

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
        dvalid = lgb.Dataset(val_df[predictors].values,
                             label=val_df[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical)
        #del val_df
        #gc.collect()
        valid_sets = [dtrain, dvalid]
        valid_names = ['train', 'valid']
        best_auc_metric = 'valid'
        max_rounds = MAX_VAL_ROUNDS
        early_stopping_rounds = 50
        prefix = 'cv'
    else:
        valid_sets = [dtrain]
        valid_names = ['train']
        best_auc_metric = 'train'
        max_rounds = MAX_ROUNDS
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
    #train_df, val_df = trainval_random_split(trainval_df)
    #train_df, val_df = trainval_day9_hour4_split(trainval_df, test_size=0.1)
    train_df = trainval_df[109903891:]
    val_df = []
    print(trainval_df.info())
    
    run(train_df, val_df)
    