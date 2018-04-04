import os
import gc
import pickle
import logging
from datetime import datetime
import json

import pandas as pd
import numpy as np
import feather

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'
TMP = '/kaggle1/td-cache'

BASE_PATH = '../input'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test_v0.csv') # v0 with full rows


logger = logging.getLogger('data2')


def load(kind):
    base_fname = os.path.join(TMP, '{}_base.feather'.format(kind))
    print('loading base ', base_fname, datetime.now())
    df = feather.read_dataframe(base_fname)

    for group in [
            ['ip', 'day', 'in_test_hh'],
            ['ip', 'day', 'hour'],
            ['ip', 'os', 'hour'],
            ['ip', 'app',  'hour'],
            ['ip', 'device', 'hour'],
            ['ip', 'app', 'channel', 'hour']
    ]:
        column = 'count_{}'.format('_'.join(group))
        fname = os.path.join(TMP, '{}_{}.feather'.format(kind, column))

        print('loading ', fname, datetime.now())
        extra_df = feather.read_dataframe(fname)
        df[column] = extra_df[column]

    for group in [
            ['ip', 'device'],
            ['ip', 'app', 'device'],
            ['ip', 'app', 'device', 'os'],
            ['ip', 'app', 'device', 'os', 'channel'],
    ]:
        column = 'delta_{}'.format('_'.join(group))
        fname = os.path.join(TMP, '{}_{}.feather'.format(kind, column))

        print('loading ', fname, datetime.now())
        extra_df = feather.read_dataframe(fname)
        df[column] = extra_df[column]

    print('loaded. ', datetime.now()) 
    return df

def load_click_ids():
    click_ids = pd.read_csv('../input/test_v0.csv', usecols=['click_id']).click_id.values
    return click_ids