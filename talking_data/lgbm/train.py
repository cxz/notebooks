import os
import gc
import pickle
import logging

import pandas as pd
import lightgbm as lgb

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS = 18790469

logging.basicConfig(level=logging.DEBUG, 
                    format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('train')


def train(train_df, valid_df, params, max_rounds):
    # base parameters
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'nthread': 8,
        'verbose': -1
    }
    
    lgb_params.update(params)    
    
    features = ['app', 'channel', 'device', 'ip', 'os', 'hour']
    logger.info(features)
    
    logger.info("preparing dtrain")
    dtrain = lgb.Dataset(train_df[features], 
                         label=train_df['is_attributed'].values)
    
    logger.info("preparing dvalid")
    dvalid = lgb.Dataset(valid_df[features], 
                         label=valid_df['is_attributed'].values,
                         reference=dtrain)
                          
    evals_results = {}

    logger.info("starting train")
    m = lgb.train(lgb_params, 
                  dtrain,
                  valid_sets=[dtrain, dvalid],
                  valid_names=['train','valid'],
                  evals_result=evals_results,
                  num_boost_round=max_rounds,
                  early_stopping_rounds=100,
                  verbose_eval=10)
    
    n_estimators = m.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    #print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    
    
if __name__ == '__main__':
    with open('train_test_base.pkl', 'rb') as f:
        df = pickle.load(f)        
        assert len(df) == TRAIN_ROWS + TEST_ROWS
       
    categorical = ['app', 'channel', 'device', 'ip', 'os' ] #'hour'
    
    # lgbm recognizes automatically if the column type is category
    for c in categorical:
        df[c] = df[c].astype('category')
        
    logger.info(df.info())
    
    train_df = df.iloc[:TRAIN_ROWS-VALID_ROWS, :]
    valid_df = df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS, :]
    # test_df = df.iloc[-TEST_ROWS:]
    
    del df
    gc.collect()
    
    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
        
    # from lgbm2.py (v2) -- [30]    train's auc: 0.967871   valid's auc: 0.957921
    params = {
        'learning_rate': 0.1, 
        'num_leaves': 63, 
        'max_depth': -1,
        #'min_data_in_leaf': 1024,
        'max_bin': 1024,
        #'min_data_in_bin': 100,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 10,
        #'feature_fraction': 0.8,
        'scale_pos_weight':99
    }    

    max_rounds = 500
    train(train_df, valid_df, params, max_rounds)
                
        
    
    