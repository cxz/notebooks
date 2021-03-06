""" Load base csvs and save as dataframes in feather.
"""

import os
import gc
import pickle
import logging
import datetime

import pandas as pd
import numpy as np


import feather

from util import info
from dataset import *

def process(kind):
    """ Convert original csv into feather.
    
    Also split click_time into hour & day and drop click_id & attributed_time.
    
    """
    out_fname = os.path.join(CACHE, '{}_base.feather'.format(kind))
    if os.path.exists(out_fname):
        return
    
    fname = os.path.join(BASE_PATH, '{}.csv'.format(kind))
    info('loading %s' % fname)
        
    df = pd.read_csv(fname, dtype=DTYPES, parse_dates=['click_time'])
    
    info('hour/day')
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    df['day'] = df.click_time.dt.day.astype('uint8')
    
    if kind == 'train':
        df['is_attributed'] = df.is_attributed.astype('uint8')
    else:
        df['is_attributed'] = 0
        
    # recast
    df.is_attributed = df.is_attributed.fillna(0).astype('uint8')
    
    # delete uneeded columns, attributed_time exists only in train
    # and click_id only in test
    to_drop = [x for x in ['click_id', 'attributed_time'] if x in df.columns]
    df.drop(to_drop, axis=1, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    info(df.info())
    
    info("writing %s" % out_fname)
    feather.write_dataframe(df, out_fname)
             
    info('done.')


if __name__ == '__main__':
    process('train')
    process('test_v0')
    process('test')