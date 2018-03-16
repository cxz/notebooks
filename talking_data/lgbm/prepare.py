""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging

import pandas as pd
import numpy as np

BASE_PATH = '../input'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test.csv')

DTYPES = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

import logging
logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('prepare')


def prepare_base(out_fname):
    logger.info('load train')
    train_df = pd.read_csv(TRAIN_CSV,  dtype=DTYPES, parse_dates=['click_time'])
    
    logger.info('load test')
    test_df = pd.read_csv(TEST_CSV, dtype=DTYPES, parse_dates=['click_time'])    

    df = train_df.append(test_df)

    del test_df
    del train_df
    gc.collect()

    logger.info('hour/day')
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    df['day'] = df.click_time.dt.day.astype('uint8')
    
    # recast after merge
    df['is_attributed'] = df.is_attributed.fillna(0).astype('uint8')
    
    print('len valid:', len(df[df.day==9]))

    df.drop(['click_time', 'click_id', 'attributed_time'], axis=1, inplace=True)
    
    logger.info(df.info())

    with open(out_fname, 'wb') as f:
        pickle.dump(df, f)
        
    logger.info('done.')
        
def prepare_count(df, group, out_fname):
    # column used to retrieve the count
    helper = 'is_attributed'
    
    logger.info('building {}'.format(out_fname))
    column_name = 'count-{}'.format('-'.join(group))
    gp = df[group + [helper]].groupby(by=group)[[helper]].count().reset_index().rename(index=str, columns={helper: column_name})
    gp[[column_name]] = gp[[column_name]].astype('uint16')
    
    logger.info('merge ')
    df = df.merge(gp, on=group, how='left')
    
    logger.info('saving {}'.format(out_fname))
    out_df = df[[column_name]]
    with open(out_fname, 'wb') as f:
        pickle.dump(out_df, f)
    

def run():
    base_pkl = 'train_test_base.pkl'
    if not os.path.exists(base_pkl):
        prepare_base(base_pkl)
    
    with open(base_pkl, 'rb') as f:
        df = pickle.load(f)
        logger.info('loaded')
        
    for group in [
            ['ip', 'app'],
            ['ip', 'app', 'os'],            
            ['ip', 'day', 'hour'],
            ['ip', 'day', 'app'],
            ['ip', 'day', 'os'],
            ['ip', 'day', 'device'],
            ['ip', 'day', 'channel'],
            ['ip', 'device'],
            ['ip', 'device', 'channel']
        ]:    
        feature_name = 'train_test_{}'.format('-'.join(group))
        out_fname = "%s.pkl" % feature_name
        if not os.path.exists(out_fname):
            prepare_count(df, group, out_fname)
    
if __name__ == '__main__':
    run()
