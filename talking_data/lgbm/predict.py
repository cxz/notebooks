"""
    Predict & prepare submission.
"""

import os
import pickle
import logging
import gc

import pandas as pd
import lightgbm as lgb

import data

logging.basicConfig(level=logging.DEBUG, 
                    format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('test')

def load_model():
    with open('model-20180322031037.pkl', 'rb') as f:
        m, _, _ = pickle.load(f)        
    return m

def run(out_csv, num_iteration=-1):
    click_ids, test_df = data.load_test_split()
    
    target = 'is_attributed'
    excluded = [] + [target]
    features = [c for c in test_df.columns if c not in excluded]

    dtest = test_df[features]

    m = load_model()
    preds = m.predict(dtest, num_iteration=num_iteration)

    # generated using map_clickid.ipynb
    mapping = pd.read_csv('../cache/test_mapping.csv')
    print('len before: ', len(mapping))

    mapping = mapping.drop_duplicates(subset=['click_id'])
    print('len after duplicates removed: ', len(mapping))

    mapping = mapping.set_index(['click_id_v0'])
    
    preds_df = pd.DataFrame(preds, columns=['is_attributed'])
    preds_df['click_id_v0'] = click_ids
    
    preds_df = preds_df.set_index(['click_id_v0'])
    preds_df = mapping.join(preds_df, how='left')
    
    subm = pd.read_csv('../input/test.csv', usecols=['click_id'])
    
    preds_df = preds_df.reset_index().set_index(['click_id'])
    subm = subm.set_index(['click_id'])
    subm = subm.join(preds_df, how='left')

    subm = subm.reset_index()
    subm[['click_id', 'is_attributed']].to_csv(out_csv, index=False)

if __name__ == '__main__':
    num_iteration = 180
    out_csv = '../tmp/submxx180.csv'
    run(out_csv, num_iteration)