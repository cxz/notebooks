""" Convert to Feather
"""

import os
import gc
import pickle
import logging
import datetime
from csv import DictReader
from functools import lru_cache
from collections import Counter
from datetime import datetime 

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

def convert_to_feather():    
    """ Load train + val + test df. 
    """
    with open(os.path.join(CACHE, 'train_test_base.pkl'), 'rb') as f:

        df = pickle.load(f)        
        df = df.reset_index(drop=True)
        assert len(df) == TRAIN_ROWS + TEST_ROWS_V0, \
            "%d %d" %(len(df), TRAIN_ROWS + TEST_ROWS_V0)
        
    ips = df[['ip']].values
    click_times = df[['click_time']].values
            
    # lgbm recognizes automatically if the column type is category
    #categorical = ['app', 'channel', 'device', 'os', 'hour']
    #for c in categorical:
    #    df[c] = df[c].astype('category')
    
    train_df = df.iloc[:TRAIN_ROWS]
    test_df = df.iloc[TRAIN_ROWS:]
    feather.write_dataframe(train_df, os.path.join(TMP, 'train_base.feather'))
    feather.write_dataframe(test_df, os.path.join(TMP, 'test_base.feather'))
    
convert_to_feather()  