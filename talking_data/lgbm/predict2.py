"""
    Predict & prepare submission.
"""

import os
import pickle
import logging
import gc
from datetime import datetime

import pandas as pd
import lightgbm as lgb

import data2

logging.basicConfig(level=logging.DEBUG, 
                    format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('test')

def load_model(model_fname):
    with open(model_fname, 'rb') as f:
        m, _, _, _ = pickle.load(f)
    return m

def run(model_fname, out_csv, num_iteration=-1):
    test_df = data2.load('test')
    click_ids = data2.load_click_ids()    
            
    m = load_model(model_fname) 
    predictors = m.feature_name()
    
    print('test_df: ', len(test_df))
    print('predictors: ', predictors)
    
    if num_iteration != -1 and num_iteration != m.best_iteration - 1:
        print('best iter: {}, specified: {}', m.best_iteration, num_iteration)
    
    preds = m.predict(test_df[predictors], num_iteration=num_iteration)

    # generated using map_clickid.ipynb
    mapping = pd.read_csv('../cache/test_mapping.csv')
    print('len before: ', len(mapping))

    mapping = mapping.drop_duplicates(subset=['click_id'])
    print('len after duplicates removed: ', len(mapping))

    mapping = mapping.set_index(['click_id_v0'])
    print(mapping.head(10))
    
    preds_df = pd.DataFrame(preds, columns=['is_attributed'])
    preds_df['click_id_v0'] = click_ids
    print(preds_df[preds_df.click_id_v0==21290878])
    
    preds_df = preds_df.set_index(['click_id_v0'])
    preds_df = mapping.join(preds_df, how='left')
    
    subm = pd.read_csv('../input/test.csv', usecols=['click_id'])
    
    preds_df = preds_df.reset_index().set_index(['click_id'])
    subm = subm.set_index(['click_id'])
    subm = subm.join(preds_df, how='left')

    subm = subm.reset_index()
    subm[['click_id', 'is_attributed']].to_csv(out_csv, index=False)
    print('saved ', out_csv)

import fire
if __name__ == '__main__':
    #num_iteration = -1
    #out_csv = '../tmp/subm{}.csv'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
    #model = 'cv-x-20180403203602.pkl'
    #run(model, out_csv, num_iteration)
    fire.Fire(run)
    