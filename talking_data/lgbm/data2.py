import os
import glob
import gc
import pickle
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
import feather

from sklearn.preprocessing import RobustScaler

from dataset import *
from util import *

def load_base(kind):
    base_fname = os.path.join(CACHE, '{}_base.feather'.format(kind))
    info('loading base %s' % base_fname)
    df = feather.read_dataframe(base_fname)
    return df

def load(kind):
    df = load_base(kind)
    
    fnames = []
    fnames.extend(glob.glob(os.path.join(CACHE, '%s_count*.feather' % kind)))
    fnames.extend(glob.glob(os.path.join(CACHE, '%s_x1_*.feather' % kind)))
    fnames.extend(glob.glob(os.path.join(CACHE, '%s_x2_*.feather' % kind)))
    fnames.extend(glob.glob(os.path.join(CACHE, '%s_binip.feather' % kind)))
    #fnames.extend(glob.glob(os.path.join(CACHE, '%s_x3_*.feather' % kind)))
    fnames.extend(glob.glob(os.path.join(CACHE, '%s_x4_*.feather' % kind)))
        
    for fname in fnames:
        fpath = os.path.join(CACHE, fname)

        info('loading %s %s' %(fname, datetime.now()))
        extra_df = feather.read_dataframe(fname)
        for c in extra_df.columns:
            df[c] = extra_df[c]
        
    info('loaded. %s' % datetime.now()) 
    return df

def load_click_ids():
    click_ids = pd.read_csv('../input/test_v0.csv', usecols=['click_id']).click_id.values
    return click_ids

if __name__ == '__main__':
    df = load('train')
    print(df.info())