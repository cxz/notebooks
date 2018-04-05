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

def _prepare_lhood(df, group, out_column, dtype):    
    helper = 'is_attributed'
    gp = df[group + [helper]].groupby(by=group)[helper] \
        .mean() \
        .reset_index() \
        .rename(index=str, columns={helper: out_column})
    out = df[group].merge(gp, on=group, how='left')    
    out = out[[out_column]].astype(dtype)
    gc.collect()
    return out  

def process(df, kind):
    if kind == 'test':
        # TODO: do not forget test set cannot be used to build this feature
        raise
    
    for group in [
            #['ip', 'day', 'in_test_hh'],
            #['app', 'channel', 'day', 'in_test_hh'],
            #['ip', 'hour'],
            #['ip', 'device', 'hour'],
            ['ip', 'channel', 'hour'],
            ['ip', 'app', 'hour'],
    ]:
        out_column = 'lhood_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        out = _prepare_lhood(df, group, out_column, np.uint32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
