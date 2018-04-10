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

def _prepare_x1_properly(df, group, out_column, dtype):    
    out = df[group + ['day']]  
    
    # try to reduce leakage by splitting calculation 
    # into folds (one for each day)
    for day in range(6, 10):
        print('day ', day)

        # exclude this day
        day_group_df = df[~(df.day==day)]
        day_column = 'out_{}'.format(day)
        
        # Find frequency of is_attributed for each unique value in column
        freqs = {}

        # Perform the groupby
        group_object = day_group_df.groupby(group)

        # Group sizes    
        group_sizes = group_object.size()
        log_group = 100000 # 1000 views -> 60% confidence, 100 views -> 40% confidence 
        print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
            group, day_column, 
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

        # Perform the merge
        merged = out.merge(
            group_object['is_attributed']. \
                apply(rate_calculation). \
                reset_index(). \
                rename( 
                    index=str,
                    columns={'is_attributed': day_column}
                )[group + [day_column]],
            on=group, how='left'
        )

        out.loc[out.day==day, out_column] = merged.loc[merged.day==day, day_column]
        
        del day_group_df
        del merged
        del group_object
        del group_sizes        
        gc.collect()        
    
    out = out[[out_column]].astype(dtype)    
    gc.collect()
    return out
        
def _prepare_x1(source_df, target_df, group, out_column, dtype):    
    out = target_df[group + ['day']]  

    # Find frequency of is_attributed for each unique value in column
    freqs = {}

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

    # Perform the merge
    out = out.merge(
        group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename( 
                index=str,
                columns={'is_attributed': out_column}
            )[group + [out_column]],
        on=group, how='left'
    )
    
    del group_object
    del group_sizes        
    gc.collect()        
    
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
          
    ]:
        info('loading base')
        df = load_base(kind)

        #controlled leak ^-^
        #source_cond = (df.day.isin([8, 9])) & df.hour.isin([4,5,9,10,13,14])
        source_cond = (df.day.isin([9])) & df.hour.isin([4,5,9,10,13,14])
        source_df = df[source_cond]
        target_df = df if kind == 'train' else load_base('test')
        info('source: %d, target: %d' %(len(source_df), len(target_df)))
        
        out_column = "x1_{}".format('_'.join(group))
        out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        out = _prepare_x1(source_df, target_df, group, out_column, np.float32)
        
        info(out[out_column].describe())

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())

if __name__ == '__main__':    
    process('train')