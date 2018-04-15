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
          
def process(kind):
    
    GROUPBY_AGGREGATIONS = [
        #{'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
        {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
        #{'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
        #{'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
        {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
        #{'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
        {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 

        #{'groupby': ['app'], 
        # 'select': 'ip', 
        # 'agg': lambda x: float(len(x)) / len(x.unique()), 
        # 'agg_name': 'AvgViewPerDistinct'
        #},
        
        {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
        #{'groupby': ['channel'], 'select': 'app', 'agg': 'count'}
    ]
        
    # Apply all the groupby transformations
    for spec in GROUPBY_AGGREGATIONS:
        # Name of the aggregation we're applying
        agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']

        new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])        
        out_column = "x2_{}".format(new_feature)
        out_fname = os.path.join(CACHE, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue
        
        df = load_base('train')
        
        source_cond = (df.day.isin([9])) & df.hour.isin([4,5,9,10,13,14])
        source_df = df[source_cond]
        
        target_df = df if kind == 'train' else load_base('test')
        info('source: %d, target: %d' %(len(source_df), len(target_df)))        
        

        # Info
        print("Grouping by {}, and aggregating {} with {}".format(
            spec['groupby'], spec['select'], agg_name
        ))
        
        # Unique list of features to select
        all_features = list(set(spec['groupby'] + [spec['select']]))

        # ...
        group_fname = 'x2_{}'.format('_'.join(new_feature))
        group_fpath = os.path.join(CACHE, group_fname)
            
        if kind == 'train':
            print('preparing ', out_column, datetime.now())

            # Perform the groupby
            gp = source_df[all_features]. \
                groupby(spec['groupby'])[spec['select']]. \
                agg(spec['agg']). \
                reset_index(). \
                rename(index=str, columns={spec['select']: out_column})
                    
            with open(group_fpath, 'wb') as f:
                pickle.dump(gp, f)
        else:
            with open(group_fpath, 'rb') as f:
                gp = pickle.load(f)            

        out = target_df.merge(gp, on=spec['groupby'], how='left')
        out = out[[out_column]].astype(np.float32)
        
        info(out[out_column].describe())

        feather.write_dataframe(out, out_fname)
        info('wrote %s' % out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
        
if __name__ == '__main__':
    # process('train')
    process('test')
