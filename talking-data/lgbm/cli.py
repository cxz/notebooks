"""
    Run LightGBM using cli to avoid memory issues with python wrapper.
    
"""
import os
import logging
import gc

import pandas as pd

import data

CACHE = data.CACHE

logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('cli')

def prepare_data():
    train_df, valid_df, _ = data.load_train_val_splits(use_validation=False)
    
    target = 'is_attributed'
    excluded = [] + [target]
    features = [c for c in train_df.columns if c not in excluded]    
    
    train_fname = os.path.join(CACHE, 'lgbm_train.csv')
    if not os.path.exists(train_fname):
        logger.info("saving %s" % train_fname)
        train_df[[target] + features].to_csv(train_fname, index=False, header=None)
        
    valid_fname = os.path.join(CACHE, 'lgbm_valid.csv')
    if not os.path.exists(valid_fname):
        if len(valid_df) > 0:
            logger.info("saving %s" %valid_fname)
            valid_df[[target] + features].to_csv(valid_fname, index=False, header=None)
            
    del train_df
    del valid_df
    gc.collect()
    
    click_ids, test_df = data.load_test_split()
            
    test_fname = os.path.join(CACHE, 'lgbm_test.csv')
    if not os.path.exists(test_fname):
        logger.info("saving %s" % test_fname)
        test_df[features].to_csv(test_fname, index=False, header=None)
    
    features_fname = os.path.join(CACHE, 'lgbm_features.txt')
    pd.DataFrame(columns=[target] + features).to_csv(features_fname, index=False)
    
        
def run_cv():    
    TRAIN_CONF = """task=train 
objective=binary 
metric=auc
boosting_type=gbdt 
metric_freq=1 
num_threads=8
verbose=1
"""
    with open(os.path.join(CACHE, 'train.conf'), 'w') as f:
        f.write(TRAIN_CONF)
    
    params = {
        'learning_rate': 0.1,
        'num_leaves': 128, 
        'max_depth': 7,
        #'min_data_in_leaf': 1024,
        #'max_bin': 1024,
        #'min_data_in_bin': 100,
        #'bagging_fraction': 1.0,
        #'bagging_freq': 2, 
        #'feature_fraction': 0.8,        
        'scale_pos_weight': 300,
        #'is_unbalance': "true",
        'categorical_feature': "0,1,2,3,4",
        'num_iterations': 160,
        'early_stopping_rounds': 30,
    } 
      
    # http://lightgbm.readthedocs.io/en/latest/Parameters.html
    cmd = {        
        'config': '../cache/train.conf',
        'train_data': '../cache/lgbm_train.csv',
        #'valid_data': '../cache/lgbm_valid.csv',
        'output_model': '../cache/model.txt'
    }    
    
    params_str = ' '.join(["%s=%s" % (k, v) for k, v in params.items()])
    cmd_str = ' '.join(["%s=%s" % (k, v) for k, v in cmd.items()])
    print("./lightgbm {} {}".format(cmd_str, params_str))
    
def run_predict():    
    PREDICT_CONF = """
task=predict    
data=../cache/lgbm_test.csv
input_model=../cache/model.txt
"""    
    with open(os.path.join(CACHE, 'predict.conf'), 'w') as f:
        f.write(PREDICT_CONF)
    
    cmd = {
        'config': '../cache/predict.conf',
    }
    
    cmd_str = ' '.join(["%s=%s" % (k, v) for k, v in cmd.items()])
    print('./lightgbm {}'.format(cmd_str))

if __name__ == '__main__':
    prepare_data()
    #run_cv()
    #logger.info('done')