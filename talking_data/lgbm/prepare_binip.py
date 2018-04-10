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

def _prepare_count(source_df, target_df, out_column):
    helper = 'is_attributed'
    group = ['ip']
    gp = source_df[group + [helper]].groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
    count = target_df[group].merge(gp, on=group, how='left')
    count[out_column] = count[out_column].fillna(0)
    out = pd.cut(count[out_column], 255, labels=False).astype(np.uint8).to_frame()        
    gc.collect()
    return out  

def process(kind):
    df = load_base('train') # calculate bins based on train
    
    source_cond = (df.day == 9)  # both train/test
    source_df = df[source_cond]
    target_df = load_base('test') if kind == 'test' else df
    
    out_column = 'binip'
    out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
    if os.path.exists(out_fname):
        info('%s exists, skippping.' % out_fname)
        return

    info('preparing %s' % out_column)
    out = _prepare_count(source_df, target_df, out_column)

    feather.write_dataframe(out, out_fname)
    print('wrote ', out_fname)

    del out
    gc.collect()

    print('done ', datetime.now())

if __name__ == '__main__':
    process('train')