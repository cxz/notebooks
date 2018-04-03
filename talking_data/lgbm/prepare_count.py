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

from tqdm import tqdm

TMP = '/kaggle1/td-cache'

def _prepare_count(df, group, out_column, dtype):    
    helper = 'is_attributed'
    gp = df[group + [helper]].groupby(by=group)[helper] \
        .count() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
    out = df[group].merge(gp, on=group, how='left')    
    out = out[[out_column]].astype(dtype)
    gc.collect()
    return out  

def process(df, kind):
    
    for group in [
            ['ip', 'day', 'in_test_hh'],
            ['ip', 'day', 'hour'],
            ['ip', 'os', 'hour'],
            ['ip', 'app',  'hour'],
            ['ip', 'device', 'hour'],
            ['ip', 'app', 'channel', 'hour']
    ]:
        out_column = 'count_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        out = _prepare_count(df, group, out_column, np.uint32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
