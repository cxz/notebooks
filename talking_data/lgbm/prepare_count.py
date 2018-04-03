import os
import gc
import pickle
import logging
import datetime
from csv import DictReader
from functools import lru_cache
from collections import Counter
from datetime import datetime 

import pandas as pd
import numpy as np

from tqdm import tqdm
import hashlib


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

def prepare_count(df, group, out_column, dtype):    
    helper = 'is_attributed'
    gp = df[group + [helper]].groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
    out = df[group].merge(gp, on=group, how='left')    
    out = out[[out_column]].astype(dtype)
    gc.collect()
    return out  

def run():
    print("loading ", datetime.now())
    train_df = feather.read_dataframe(os.path.join(TMP, 'train_base.feather'))
    print("done. ", datetime.now())
    
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]

    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    
    for group in [
            ['ip', 'day', 'in_test_hh'],
            ['ip', 'day', 'hour'],
            ['ip', 'os', 'hour'],
            ['ip', 'app',  'hour'],
            ['ip', 'device', 'hour'],
            ['ip', 'app', 'channel', 'hour']
    ]:
        out_column = 'count_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, 'train_{}.feather'.format(out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        out = prepare_count(train_df, group, out_column, np.uint32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
    
    
if __name__ == '__main__':
    run()