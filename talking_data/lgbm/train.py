"""
       LightGBM training.

"""

import os
import gc
import pickle
import logging
import datetime
import json

import pandas as pd
import numpy as np
import lightgbm as lgb
from hyperopt import hp, tpe, Trials, space_eval, fmin

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'

logging.basicConfig(level=logging.DEBUG, 
                    format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('train')

LGB_PARAMS = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 8,
    'verbose': -1
}

DEBUG = False

def train(train_df, valid_df, params, max_rounds, learning_rates=None):
    # base parameters
    lgb_params = LGB_PARAMS.copy()
    lgb_params.update(params)    
    
    target = 'is_attributed'
    excluded = [] + [target]
    features = [c for c in train_df.columns if c not in excluded]
    logger.info('features: %s' % (' '.join(features)))
    
    logger.info("preparing dtrain")
    dtrain = lgb.Dataset(train_df[features], 
                         label=train_df['is_attributed'].values)
    
    use_validation = valid_df is not None
    valid_sets = []
    valid_names = []
    
    if use_validation:
        logger.info("preparing dvalid")
        dvalid = lgb.Dataset(valid_df[features], 
                             label=valid_df['is_attributed'].values,
                             reference=dtrain)
        valid_sets = [dvalid]
        valid_names = ['valid']
        early_stopping_rounds = 30
    else:
        valid_sets = [dtrain]
        valid_names = ['train']
        early_stopping_rounds = None
                          
    evals_results = {}

    logger.info("starting train")
    logger.info("parameters: {}".format(json.dumps(lgb_params)))
    
    m = lgb.train(lgb_params, 
                  dtrain,
                  valid_sets=valid_sets,
                  valid_names=valid_names,
                  evals_result=evals_results,
                  num_boost_round=max_rounds,
                  early_stopping_rounds=early_stopping_rounds,
                  learning_rates=learning_rates,
                  verbose_eval=1)
        
    logger.info("parameters: {}".format(json.dumps(lgb_params)))
    logger.info("n_estimators : {}".format(m.best_iteration))
    if use_validation:
        best_auc = evals_results['valid'][LGB_PARAMS['metric']][m.best_iteration - 1]
        logger.info("auc : {}".format(best_auc))
    return m, evals_results, None

    
def load():    
    """ Load train + val + test df. 
    """
    with open(os.path.join(CACHE, 'train_test_base.pkl'), 'rb') as f:
        logger.info('loading base')
        df = pickle.load(f)        
        df = df.reset_index(drop=True)
        assert len(df) == TRAIN_ROWS + TEST_ROWS_V0, \
            "%d %d" %(len(df), TRAIN_ROWS + TEST_ROWS_V0)
        
    for extra in [
        'count_ip-app-os.pkl',
        'count_ip-app.pkl',
        'count_ip-channel.pkl',
        'count_ip-day-app.pkl',
        'count_ip-day-channel.pkl',
        'count_ip-day-device.pkl',
        'count_ip-day-hour.pkl',
        'count_ip-day-os.pkl',
        'count_ip-device-channel.pkl',
        'count_ip-device.pkl',
        'count_ip-os.pkl',
        'count_os-device-app.pkl',
        'count_os-device-channel.pkl',
        'delta_ip-app.pkl',
        'delta_ip-device-app.pkl',
        'delta_ip-device-channel.pkl',
        'delta_ip-device-os-app.pkl',
        'delta_ip-device-os.pkl'        
        #'delta_rev_ip-device.pkl',
        #'delta_rev_ip-device-app.pkl',
        #'delta_rev_ip-device-channel.pkl'
    ]: 
        with open(os.path.join(CACHE, extra), 'rb') as f:
            logger.info('loading %s' % extra)
            df2 = pickle.load(f)
        for c in df2.columns:
            df[c] = df2[c]
        del df2
        gc.collect()
            
    logger.info('done loading extra features.')
    
    df.drop(['click_time', 'ip', 'day'], axis=1, inplace=True)
    categorical = ['app', 'channel', 'device', 'os', 'hour' ]
    
    # lgbm recognizes automatically if the column type is category
    for c in categorical:
        df[c] = df[c].astype('category')
        
    return df

def load_splits(use_validation=False):
    df = load()
        
    logger.info(df.info())

    if DEBUG:
        valid_rows = 10000000
        train_rows = 50000000
        train_df = df.iloc[TRAIN_ROWS-train_rows-valid_rows:TRAIN_ROWS-valid_rows, :]
        valid_df = df.iloc[TRAIN_ROWS-valid_rows:TRAIN_ROWS, :]        
    
    if use_validation:
        train_df = df.iloc[:TRAIN_ROWS-VALID_ROWS, :]
        valid_df = df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS, :]
        # test_df = df.iloc[-TEST_ROWS_V0:]
        test_df = None        
        
    else:
        train_df = df.iloc[:TRAIN_ROWS, :]
        valid_df = []
        # test_df = df.iloc[-TEST_ROWS_V0:]
        test_df = None

    del df
    gc.collect()
    
    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
    return train_df, valid_df, test_df

def run_cv(params):
    train_df, valid_df, _ = load_splits()
    
    max_rounds = 1000
    m, evals_result, best_auc = train(train_df, 
                                      valid_df, 
                                      params, 
                                      max_rounds)
                                      #learning_rates=lambda it: 0.1 if it < 80 else 0.5 ** (it//80))
    
    out = 'cv-{}-{}.pkl'.format(best_auc, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if DEBUG:
        out = "debug-%s" % out
    
    with open(out, 'wb') as f:
        pickle.dump([m, evals_result, params], f)
        
    return best_auc
    
    
def hp_objective(params):
    for p in ['max_depth', 'bagging_freq', 'min_data_in_leaf']: # , 'max_bin']: #
        params[p] = int(params[p])
        
    # in lgbm, num_leaves tunned together with max_depth
    params['num_leaves'] = int(2 ** params['max_depth'])
    
    auc = run_cv(params)
    return -auc

def run_cv_single():
    """ Single run using cross validation.
    """
    params = {
        'learning_rate': 0.1,
        'num_leaves': 64, 
        'max_depth': -1,
        #'min_data_in_leaf': 1024,
        #'max_bin': 1024,
        #'min_data_in_bin': 100,
        #'bagging_fraction': 0.7,
        #'bagging_freq': 5,        
        #'feature_fraction': 0.8,
        'scale_pos_weight': 300
    }     
    run_cv(params)
    
def run_cv_single2():
    """
        [176]   valid's auc: 0.973569
        train.py 2018-03-18 00:20:23,476 INFO parameters: {"boosting_type": "gbdt", "objective": "binary", "metric": "auc", "nthread": 8, "verbose": -1, "learning_rate": 0.1, "num_leaves": 128, "max_depth": 7, "bagging_fraction": 1.0, "bagging_freq": 2, "feature_fraction": 0.8, "scale_pos_weight": 300, "categorical_column": [0, 1, 2, 3, 4]}
        
        'train_test_ip-day-hour.pkl',
        'train_test_ip-day-app.pkl',
        'train_test_ip-day-os.pkl',
        'train_test_ip-day-device.pkl',
        'train_test_ip-day-channel.pkl',
        'train_test_ip-device.pkl',
        'train_test_ip-app.pkl',
        'train_test_ip-app-os.pkl',
        'train_test_ip-device-channel.pkl',
        'delta_ip-device-os.pkl',
        'delta_ip-app-device-os.pkl',
    """
    params = {
        'learning_rate': 0.1,
        'num_leaves': 128, 
        'max_depth': 7,
        'bagging_fraction': 1.0,
        'bagging_freq': 2, 
        'feature_fraction': 0.8,
        'scale_pos_weight': 300
    }     
    run_cv(params)   
    
        
def run_hp_search():    
    """ Hyperparameters search.
    """
    
    space = {
        'learning_rate': 0.1,
        'scale_pos_weight': 300,        
        'max_depth': 7,
        #'max_depth': hp.choice('max_depth', [5, 6, 7, 8, 9]),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
        'bagging_freq': hp.choice('bagging_freq', np.arange(1, 5)),
        'feature_fraction': hp.quniform('feature_fraction', 0.7, 1.0, 0.1),
        #'max_bin': hp.choice('max_bin', [255, 511, 1023])
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 500, 5000, 500),
        #min_data_in_bin
    }
    
    best = fmin(hp_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=40, verbose=1)
    logger.info("best {}".format(best))
    best_params = space_eval(space, best)
    logger.info("best params {}".format(best_params))
    
def run_train_full():
    """
    """
    params = {
        'learning_rate': 0.1,
        'num_leaves': 128, 
        'max_depth': 7,
        'bagging_fraction': 1.0,
        'bagging_freq': 2, 
        'feature_fraction': 0.8,
        'scale_pos_weight': 300
    }     
    train_df, _, _ = load_splits(use_validation=False)
    
    rounds = 150
    m, evals_result, best_auc = train(train_df, 
                                      None, 
                                      params, 
                                      rounds)
                                          
    out = 'model-{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    
    with open(out, 'wb') as f:
        pickle.dump([m, evals_result, params], f)
        
    return best_auc

    
if __name__ == '__main__':
    #run_cv_single()
    #run_hp_search()
    #run_cv_single2()
    run_train_full()
    
    
    
                
