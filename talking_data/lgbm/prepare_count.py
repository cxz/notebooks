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

from sklearn.preprocessing import RobustScaler
    
from data2 import load_base
from util import *
from dataset import *

def _prepare_count(source_df, target_df, group, out_column, dtype):    
    helper = 'is_attributed'
    gp = source_df[group + [helper]].groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
    out = target_df[group].merge(gp, on=group, how='left')    
    out = out[[out_column]].astype(dtype)
    gc.collect()
    return out  

def _fit_save_scaler(df, column):
    scaler = RobustScaler()
        
    scaler.fit(df[column].fillna(0).values.reshape(-1, 1))
    
    scaler_fname = os.path.join(CACHE, "scaler_%s.pkl" % column)
    with open(scaler_fname, 'wb') as f:
        pickle.dump(scaler, f)
        
def _scale(df, column):
    scaler_fname = os.path.join(CACHE, "scaler_%s.pkl" % column)
    with open(scaler_fname, 'rb') as f:
        scaler = pickle.load(f)
            
    df[column] = scaler.transform(df[column].fillna(0).values.reshape(-1, 1))
    
def process(kind):    
    for group in [
        #['ip', 'day', 'in_test_hh'],
        ['ip', 'day', 'hour'],
        ['ip', 'os', 'hour'],
        ['ip', 'app',  'hour'],            
        ['ip', 'device', 'hour'],
        ['ip', 'app', 'channel', 'hour'],
        #['ip', 'day', 'app', 'in_test_hh'],
        ['ip', 'day', 'app', 'hour'],
        #['ip', 'day', 'device', 'in_test_hh'],
        #['app', 'day', 'in_test_hh'],
        #['channel', 'day', 'in_test_hh'],
        #['binip', 'os', 'hour']
        ]:   
        info('loading base')
        df = load_base(kind)

        source_cond = (df.day.isin([8, 9])) & df.hour.isin([4,5,9,10,13,14])
        source_df = df[source_cond] if kind == 'train' else df
        target_df = df
        info('source: %d, target: %d' %(len(source_df), len(target_df)))

        out_column = 'count_{}'.format('_'.join(group))
        out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            info("%s exists, skipping." % out_fname)
            continue

        info('preparing %s %s' % (out_column, datetime.now()))
        out = _prepare_count(source_df, target_df, group, out_column, np.float32)

        info(out.info())
        info(out[out_column].describe())

        #if kind == 'train':
        #    info("fitting scaler")
        #    _fit_save_scaler(out, out_column)

        info(out.info())

        #info("applying scaler")
        #_scale(out, out_column)
        #out[out_column] = out[out_column].astype(np.float32)

        info(out[out_column].describe())

        feather.write_dataframe(out, out_fname)
        info('wrote %s' % out_fname)

        del out
        gc.collect()

        info('done %s' % datetime.now())

if __name__ == '__main__':
    process('train')    
    