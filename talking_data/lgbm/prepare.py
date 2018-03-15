""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging

import pandas as pd

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

    df.drop(['click_time', 'click_id', 'attributed_time', 'day'], axis=1, inplace=True)
    
    logger.info(df.info())

    with open(out_fname, 'wb') as f:
        pickle.dump(df, f)
        
    logger.info('done.')
        

if __name__ == '__main__':
    prepare_base('train_test_base.pkl')
