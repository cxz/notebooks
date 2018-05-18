""" IP frequencies.
"""

import os
import gc
import pickle
import logging

from functools import lru_cache
from collections import Counter
from datetime import datetime 

import pandas as pd
import numpy as np
import feather

from data2 import load_base
from util import *
from dataset import *

# important to group by day as well, so counts between
# train & test are equivalent
GROUP = ['ip', 'day']
BINS = 255
DTYPE = np.uint8

def _build_bins():
    bin_fname = os.path.join(CACHE, 'binip.pkl')
    if os.path.exists(bin_fname):
        with open(bin_fname, 'rb') as f:
            bins = pickle.load(f)
            return bins
        
    # build full ips from train + test_v0
    df = load_base('train')
    df = df.append(load_base('test_v0'))
    assert len(df) == TEST_ROWS_V0 + TRAIN_ROWS
        
    group = GROUP
    helper = 'is_attributed'
    out_column = 'binip'
    
    gp = df[group + [helper]].groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
        
    count = df[group].merge(gp, on=group, how='left')
    _, bins = pd.qcut(count[out_column], BINS, retbins=True, labels=False, duplicates='drop')
    
    info("==== bins:")
    info(bins)
    
    with open(bin_fname, 'wb') as f:
        pickle.dump(bins, f)
        
    return bins
        

def _prepare_binip(bins, df, out_column, kind):
    group = GROUP
    helper = 'is_attributed'
    gp = df[group + [helper]] \
        .groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
        
    count = df[group].merge(gp, on=group, how='left')
    info(count[out_column].describe())
    info("nans: %d" % len(count[count[out_column].isnull()]))
    assert 0 == len(count[count[out_column].isnull()])
    
    out = pd.cut(count[out_column], bins, labels=False)
    info(out.describe())
    info("nans: %d" % len(out[out.isnull()]))
    
    # nans happen here for unique ips i.e. where count is 1. 
    # merge then with first bin
    out = out.fillna(0)
    out = out.astype(DTYPE).to_frame() 

    del count
    del gp
    gc.collect()
    return out  

def process(kind):
    out_column = 'binip'
    out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
    if os.path.exists(out_fname):
        info('%s exists, skippping.' % out_fname)
        return
    
    info("preparing bins")
    bins = _build_bins()
    gc.collect()
    
    info("reading df")
    df = load_base(kind)
    
    info('preparing %s' % out_column)
    out = _prepare_binip(bins, df, out_column, kind)

    feather.write_dataframe(out, out_fname)
    print('wrote ', out_fname)

    del out
    del bins
    del df
    gc.collect()

    print('done ', datetime.now())

if __name__ == '__main__':
    process('train')
    process('test')
    process('test_v0')