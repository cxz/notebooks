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

def _prepare_count(kind, source_df, target_df, group, out_column, dtype):    
    helper = 'is_attributed'
    
    group_fname = 'count_group_{}'.format('_'.join(group))
    group_fpath = os.path.join(CACHE, group_fname)
    
    if kind == 'train':
        group_object = source_df[group + [helper]].groupby(group)

        group_sizes = group_object.size()
        log_group = 100000 # 1000 views -> 60% confidence, 100 views -> 40% confidence 

        info(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
            group, out_column, 
            group_sizes.max(), 
            np.round(group_sizes.mean(), 2),
            np.round(group_sizes.median(), 2),
            group_sizes.min()
        ))

        # Aggregation function
        def rate_calculation(x):
            """Calculate the attributed rate. Scale by confidence"""
            rate = x.sum() / float(x.count())
            conf = np.min([1, np.log(x.count()) / log_group])
            return rate * conf

        gp = group_object[helper] \
            .apply(rate_calculation) \
            .reset_index() \
            .rename(index=str, columns={helper: out_column})
                    
        with open(group_fpath, 'wb') as f:
            pickle.dump(gp, f)
            
    else:        
        with open(group_fpath, 'rb') as f:
            gp = pickle.load(f)
                
    # Perform the merge
    out = target_df[group].merge(gp, on=group, how='left')
    out = out[[out_column]].astype(dtype)
    gc.collect()
    return out

 
def process(kind):    
    """
        these ones did not improve val score.
        #['ip', 'day', 'in_test_hh'],
        #['ip', 'day', 'app', 'in_test_hh'],
        #['ip', 'day', 'device', 'in_test_hh'],
        #['app', 'day', 'in_test_hh'],
        #['channel', 'day', 'in_test_hh'],
        #['binip', 'os', 'hour']
    """
    for group in [
        #['ip', 'day', 'hour'],
        ['ip', 'os', 'hour'],
        ['ip', 'app',  'hour'],            
        ['ip', 'device', 'hour'],
        ['ip', 'app', 'channel', 'hour'],
        #['ip', 'day', 'app', 'hour'],
        ]:   
        
        out_column = 'count_{}'.format('_'.join(group))
        out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            info("%s exists, skipping." % out_fname)
            continue
        
        df = load_base(kind)

        source_df = df
        source_cond = df.hour.isin([4,5,9,10,13,14])
        if kind == 'train':
            source_cond = source_cond & df.day.isin([9])
        source_df = df[source_cond]
        target_df = df
        
        info('source: %d, target: %d' %(len(source_df), len(target_df)))

        info('preparing %s %s' % (out_column, datetime.now()))
        out = _prepare_count(kind, source_df, target_df, group, out_column, np.float32)

        info(out.info())
        info(out[out_column].describe())

        feather.write_dataframe(out, out_fname)
        info('wrote %s' % out_fname)

        del out
        del source_df
        del target_df
        gc.collect()

        info('done %s' % datetime.now())


if __name__ == '__main__':
    # process('train')
    process('test')
    