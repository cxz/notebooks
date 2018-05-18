""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging
from datetime import datetime
import logging
from csv import DictReader
from functools import lru_cache
from collections import Counter

import pandas as pd
import numpy as np
import feather

from tqdm import tqdm

TMP = '/kaggle1/td-cache'

logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('prepare')

BASE_PATH = '../input'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test_v0.csv') # v0 with full rows
MISSING = -1
        
def iter_all(paths):
    for path in paths:
        for t, row in tqdm(enumerate(DictReader(open(path)))):
            yield row        

@lru_cache(maxsize=None)    
def parse_ts(txt):
    fmt = "%Y-%m-%d %H:%M:%S"    
    return datetime.strptime(txt, fmt)    

def _prepare_click_time_delta(df, kind, group, out_fname, dtype):
    """ Delta since last click_time .         
    """
    if kind != 'train':
        raise # not ready for test.
    
    logger.info('building {}'.format(out_fname))
    column_name = 'delta_{}'.format('_'.join(group))
            
    # store last click by given group
    last_click = {}
    
    out = []
    
    #assuming input csv are ordered by click_time.
    for row in iter_all([TRAIN_CSV]):
        k = tuple([row[g] for g in group])
        prev = last_click.get(k, MISSING)
        curr = parse_ts(row['click_time'])
        if prev != -1:
            delta = curr - prev
            delta_seconds = delta.total_seconds()
            out.append(delta_seconds)
        else:
            out.append(MISSING)
        last_click[k] = curr
        
    diff = pd.DataFrame(out, columns=[column_name])    
    diff.loc[diff[column_name] > 0, column_name] = np.log1p(diff.loc[diff[column_name] > 0, column_name])
    diff = diff.astype(dtype)
    return diff
    
    
def process(df, kind):

    for group in [
            ['ip', 'device'],
            ['ip', 'app', 'device'],            
            ['ip', 'device', 'os', 'channel'],
            #['ip', 'device', 'os', 'app'],
            #['app', 'channel']
    ]:
        out_column = 'delta_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        # using float instead of uint because log1p and -1 for missing.
        out = _prepare_click_time_delta(df, kind, group, out_column, np.float32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
    
    
if __name__ == '__main__':
    # sanity check
    for kind in ['train', 'test']:
        df = feather.read_dataframe(os.path.join(TMP, '{}_delta_ip_device.feather'.format(kind)))
        print(df.info())
        print(df.delta_ip_device.describe())
