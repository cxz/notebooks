import os
import gc
import pickle
import logging
import datetime

import pandas as pd
import lightgbm as lgb

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS = 18790469

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
    
    logger.info("preparing dvalid")
    dvalid = lgb.Dataset(valid_df[features], 
                         label=valid_df['is_attributed'].values,
                         reference=dtrain)
                          
    evals_results = {}

    logger.info("starting train")
    m = lgb.train(lgb_params, 
                  dtrain,
                  valid_sets=[dvalid],
                  valid_names=['valid'],
                  #valid_sets=[dtrain, dvalid],
                  #valid_names=['train','valid'],
                  evals_result=evals_results,
                  num_boost_round=max_rounds,
                  early_stopping_rounds=100,
                  learning_rates=learning_rates,
                  verbose_eval=10)
    
    print("n_estimators : ", m.best_iteration)
    #print(metrics+":", evals_results['valid'][metrics][n_estimators-1])    
    return m, evals_result

    
def load():    
    """ Load train + val + test df. 
    """
    with open('train_test_base.pkl', 'rb') as f:
        logger.info('loading base')
        df = pickle.load(f)        
        df = df.reset_index(drop=True)
        assert len(df) == TRAIN_ROWS + TEST_ROWS
        
    for extra in [
        'train_test_ip-day-hour.pkl',
        #'train_test_ip-app.pkl',
        #'train_test_ip-app-os.pkl',
        #'train_test_ip-device.pkl',
        #'train_test_ip-device-channel.pkl'
    ]: 
        with open(extra, 'rb') as f:
            logger.info('loading %s' % extra)
            df2 = pickle.load(f)
        for c in df2.columns:
            df[c] = df2[c]
        del df2
        gc.collect()
            
    logger.info('done loading extra features.')
    
    df.drop(['ip', 'day'], axis=1, inplace=True)
    categorical = ['app', 'channel', 'device', 'os', 'hour' ]
    
    # lgbm recognizes automatically if the column type is category
    for c in categorical:
        df[c] = df[c].astype('category')
        
    return df

def load_splits():
    df = load()
        
    logger.info(df.info())
    
    train_df = df.iloc[:TRAIN_ROWS-VALID_ROWS, :]
    valid_df = df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS, :]
    # test_df = df.iloc[-TEST_ROWS:]
    
    del df
    gc.collect()
    
    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
    return train_df, valid_df, None

def run_cv():
    train_df, valid_df, _ = load_splits()

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

    max_rounds = 500
    m, evals_result = train(train_df, 
                            valid_df, 
                            params, 
                            max_rounds,
                            learning_rates=lambda it: 0.1 if it < 60 else 0.5 ** (it//60))
    
    with open('cv-{}.pkl'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")), 'wb') as f:
        pickle.dump([m, evals_result, params], f)
    
    
if __name__ == '__main__':
    run_cv()
                
