""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging
from datetime import datetime
from csv import DictReader
from functools import lru_cache
from collections import Counter

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

import logging
logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('prepare')

MISSING = -1


def datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.uint32)

def prepare_click_time_delta(df, group, column_name, dtype):
    """ Click_time difference between consecute grouped rows.
    """
    
    logger.info('building {}'.format(column_name))
    
    df['t'] = datetime_to_deltas(df.click_time)    
            
    # store last click by given group
    last_click = {}
    
    out = []
    
    #assuming input csv are ordered by click_time.
    for row in tqdm(zip(*([df['t']] + [df[g] for g in group])), total=len(df)): 
        t = row[0]
        k = tuple(row[1:])
        prev = last_click.get(k, -1)
        curr = t
        if prev != -1:
            delta = curr - prev # in seconds
            out.append(delta)
        else:
            out.append(MISSING)
        last_click[k] = curr
        
    diff = pd.DataFrame(out, columns=[column_name]).astype(dtype)
    return diff
        
    
def run():
    print("loading ", datetime.now())
    train_df = feather.read_dataframe(os.path.join(TMP, 'train_base.feather'))
    print("done. ", datetime.now())

    for group in [
            ['ip', 'device'],
            ['ip', 'app', 'device'],
            ['ip', 'app', 'device', 'os'],
            ['ip', 'app', 'device', 'os', 'channel'],
    ]:
        out_column = 'delta_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, 'train_{}.feather'.format(out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        out = prepare_click_time_delta(train_df, group, out_column, np.int16)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
    
    
if __name__ == '__main__':
    run()
