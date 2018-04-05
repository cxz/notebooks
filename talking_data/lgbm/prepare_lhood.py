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

def process(df, kind):    
    for group in [
            #['ip', 'day', 'in_test_hh'],
            #['app', 'channel', 'day', 'in_test_hh'],
            #['ip', 'hour'],
            #['ip', 'device', 'hour'],
            ['ip', 'channel', 'hour'],
            ['ip', 'app', 'hour'],
            ['channel', 'hour'],
            ['app', 'hour'],
            ['app', 'channel', 'hour'],
            ['os', 'hour']
    ]:
        out_column = 'lhood_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        helper = 'is_attributed'
        
        # calculate using train
        if kind == 'train':
            group_df = df
            
        else:
            group_df = feather.read_dataframe(os.path.join(TMP, '{}_base.feather'.format('train')))
            group_df = group_df[group + [helper, 'day']]
            gc.collect()
            
        # placeholder for final output        
        out = df[group + ['day']]                        

        #
        #
        #
        # TODO: handling ips not in common between train & test sets
        #
        #
        
        # try to reduce leakage by splitting calculation 
        # into folds (one for each day)
        for day in range(6, 10):
            print('day ', day)
            
            # exclude this day
            day_group_df = group_df[~(group_df.day==day)]
            
            day_column = 'out_{}'.format(day)
            gp = day_group_df[group + [helper]].groupby(by=group)[helper] \
                .mean() \
                .reset_index() \
                .rename(index=str, columns={helper: day_column})
            
            gc.collect()

            # merge result for this day
            out = out.merge(gp, on=group, how='left') 
            out.loc[out.day==day, out_column] = out.loc[out.day==day, day_column]
            
            #cleanup
            del out[day_column]             
            del gp
            del day_group_df
            gc.collect()
                            
        # binning
        bins = [0, .25, .5, .75, 1.]
        labels = [0, 1, 2, 3]
        out[out_column] = pd.cut(out[out_column], bins=bins, labels=labels).astype(np.uint8)
        out = out.loc[:, [out_column]]
        
        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
