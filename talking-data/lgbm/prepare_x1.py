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

        
def _prepare_x1(kind, source_df, target_df, group, out_column, dtype):    
    out = target_df[group]  

    group_fname = 'x1_group_{}'.format('_'.join(group))
    group_fpath = os.path.join(CACHE, group_fname)
    
    if kind == 'train':
        # Perform the groupby
        group_object = source_df.groupby(group)

        # Group sizes    
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
        
        gp = group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename(index=str, columns={'is_attributed': out_column})[group + [out_column]]
        
        with open(group_fpath, 'wb') as f:
            pickle.dump(gp, f)
            
        del group_object
        del group_sizes        
        gc.collect()        
        
    else:
        with open(group_fpath, 'rb') as f:
            gp = pickle.load(f)
    
    # Perform the merge
    out = out.merge(gp, on=group, how='left')
    
    out = out[[out_column]].astype(dtype)    
    gc.collect()
    return out
        
            
def process(kind):
    
    for group in [
        # single features
        ['ip'], 
        
        #['app'], ['device'], ['os'], ['channel'],

        # pairs
        ['app', 'channel'],
        ['app', 'os'],
        ['app', 'device'],
        
        #triple
        ['app', 'channel', 'hour'],
        ['app', 'os', 'hour']
          
    ]:
        out_column = "x1_{}".format('_'.join(group))
        out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue
            
        info('loading train base')
        df = load_base('train')
        
        #controlled leak ^-^
        #source_cond = (df.day.isin([8, 9])) & df.hour.isin([4,5,9,10,13,14])
        source_cond = (df.day.isin([9])) & df.hour.isin([4,5,9,10,13,14])
        source_df = df[source_cond]                
        
        target_df = df if kind == 'train' else load_base('test')
        info('source: %d, target: %d' %(len(source_df), len(target_df)))        

        print('preparing ', out_column, datetime.now())
        out = _prepare_x1(kind, source_df, target_df, group, out_column, np.float32)
        
        info(out[out_column].describe())

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        del df
        del source_df
        del target_df
        gc.collect()

        print('done ', datetime.now())

if __name__ == '__main__':    
    # process('train')
    process('test')