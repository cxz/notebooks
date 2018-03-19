""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging
import datetime
from csv import DictReader
from functools import lru_cache

import pandas as pd
import numpy as np

from tqdm import tqdm

BASE_PATH = '../input'
CACHE = '../cache'
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV = os.path.join(BASE_PATH, 'test_v0.csv') # v0 with full rows

DTYPES = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

import logging
logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('prepare')


def prepare_base(out_fname):
    logger.info('load train')
    train_df = pd.read_csv(TRAIN_CSV,  dtype=DTYPES, parse_dates=['click_time'])
    
    logger.info('load test')
    test_df = pd.read_csv(TEST_CSV, dtype=DTYPES, parse_dates=['click_time'])    

    df = train_df.append(test_df)

    del test_df
    del train_df
    gc.collect()

    logger.info('hour/day')
    df['hour'] = df.click_time.dt.hour.astype('uint8')
    df['day'] = df.click_time.dt.day.astype('uint8')
    
    # recast after merge
    df['is_attributed'] = df.is_attributed.fillna(0).astype('uint8')
    
    print('len valid:', len(df[df.day==9]))

    df.drop(['click_id', 'attributed_time'], axis=1, inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    logger.info(df.info())

    logger.info("writing %s" % out_fname)
    with open(out_fname, 'wb') as f:
        pickle.dump(df, f)
        
    logger.info('done.')
        
def prepare_count(df, group, out_fname):
    # column used to retrieve the count
    helper = 'is_attributed'
    
    logger.info('building {}'.format(out_fname))
    column_name = 'count-{}'.format('-'.join(group))
    gp = df[group + [helper]].groupby(by=group)[[helper]].count().reset_index().rename(index=str, columns={helper: column_name})
    gp[column_name] = np.log1p(gp[column_name]).astype('float16')
    
    logger.info('merge ')
    df = df.merge(gp, on=group, how='left')
    
    logger.info('saving {}'.format(out_fname))
    out_df = df[[column_name]]
    out_df.reset_index(drop=True, inplace=True) # free index storage
    
    with open(out_fname, 'wb') as f:
        pickle.dump(out_df, f)
        
    del df
    del out_df
    gc.collect()
    
def iter_all(paths):
    for path in paths:
        for t, row in tqdm(enumerate(DictReader(open(path)))):
            yield row        

@lru_cache(maxsize=None)    
def parse_ts(txt):
    fmt = "%Y-%m-%d %H:%M:%S"    
    return datetime.datetime.strptime(txt, fmt)    

def prepare_click_time_delta(df, group, out_fname):
    """ Delta since last click_time .
         
    TODO: prepare.py:149: RuntimeWarning: invalid value encountered in log1p
    
    Using pandas takes too long.
    diff = df.groupby(group) \
        .click_time \
        .apply(lambda x: x.sort_values().diff().dt.total_seconds()) \
        .fillna(-1) \
        .sort_index() \
        .reset_index(level=1, drop=True) \
        .reset_index(name='click_time_diff')
    
    """
    logger.info('building {}'.format(out_fname))
    column_name = 'delta-{}'.format('-'.join(group))
            
    # store last click by given group
    last_click = {}
    
    out = []
    
    
    # cannot use -1 as missing as we apply log1p.
    missing = 86400 * 3 
    
    #assuming input csv are ordered by click_time.
    for row in iter_all([TRAIN_CSV, TEST_CSV]):
        k = tuple([row[g] for g in group])
        prev = last_click.get(k, -1)
        curr = parse_ts(row['click_time'])
        if prev != -1:
            delta = curr - prev
            delta_seconds = delta.total_seconds()
            out.append(delta_seconds)
        else:
            out.append(missing)
        last_click[k] = curr
        
    diff = pd.DataFrame(out, columns=[column_name])    
    diff[column_name] = np.log1p(diff[column_name]).astype(np.float16)
        
    #logger.info('merging ')
    #df = df.merge(gp, on=group, how='left')
    
    logger.info('saving {}'.format(out_fname))
    out_df = diff
    with open(out_fname, 'wb') as f:
        pickle.dump(out_df, f)
    
def _cache_reversed(df, out):
    if os.path.exists(out):
        return
    df2 = df.iloc[::-1]
    df2.to_csv(out, index=False)
    
def prepare_click_time_delta_rev(df, group, out_fname):
    """ Delta to next click_time.
    """
    logger.info('building {}'.format(out_fname))
    column_name = 'delta-rev-{}'.format('-'.join(group))
            
    # store last click by given group
    last_click = {}
    
    out = []
    
    logger.info("reversing")
    reversed_csv = 'reverse.csv'
    _cache_reversed(df, reversed_csv)
    
    logger.info("starting")
    
    #assuming input csv are ordered by click_time, we just reverse the ordering
    for row in iter_all([reversed_csv]):
        k = tuple([row[g] for g in group])
        prev = last_click.get(k, -1)
        curr = parse_ts(row['click_time'])
        if prev != -1:
            delta = curr - prev
            out.append(delta.total_seconds())            
        else:
            out.append(-1)
        last_click[k] = curr
        
    diff = pd.DataFrame(out, columns=[column_name])    
    diff = diff.astype(np.float16)
    diff = diff.iloc[::-1] # reverse again to get correct ordering
        
    logger.info('saving {}'.format(out_fname))
    out_df = diff
    with open(out_fname, 'wb') as f:
        pickle.dump(out_df, f)
        
    
def run():
    base_pkl = os.path.join(CACHE, 'train_test_base.pkl')
    if not os.path.exists(base_pkl):
        prepare_base(base_pkl)
    
    #with open(base_pkl, 'rb') as f:
    #    df = pickle.load(f)
    #    logger.info('loaded')
        
    for group in [
            ['ip', 'device'],
            ['ip', 'day', 'os'],
            ['ip', 'app'],
            ['ip', 'day', 'hour'],
            ['ip', 'day', 'device'],
            ['ip', 'app', 'os'],            
            ['ip', 'day', 'app'],
            ['ip', 'day', 'channel'],
            ['ip', 'device', 'channel'],
            ['ip', 'os'],
            ['ip', 'channel'],
            ['os', 'device', 'channel'],
            ['os', 'device', 'app']
        ]:                    
        feature_name = 'count_{}'.format('-'.join(group))
        out_fname = os.path.join(CACHE, "%s.pkl" % feature_name)
        
        if os.path.exists(out_fname):
            continue
            
        with open(base_pkl, 'rb') as f:
            df = pickle.load(f)
            logger.info('loaded')
            
        prepare_count(df, group, out_fname)
            
            
        del df
        gc.collect()
                
    for group in [
            #['ip', 'device', 'os'],
            ['ip', 'device', 'os', 'app'],
            #['ip', 'device', 'channel'],
            #'ip', 'device', 'app'],
            #['ip', 'app']
        ]:
        feature_name = 'delta_{}'.format('-'.join(group))
        out_fname = os.path.join(CACHE, '%s.pkl' % feature_name)
        
        if os.path.exists(out_fname):
            continue
            
        with open(base_pkl, 'rb') as f:
            df = pickle.load(f)
            logger.info('loaded')
            
        prepare_click_time_delta(df, group, out_fname)
            
    for group in [
            #['ip', 'device'],
            #['ip', 'device', 'app'],
            #['ip', 'device', 'channel'],
        ]:
        feature_name = 'delta_rev_{}'.format('-'.join(group))
        out_fname = os.path.join(CACHE, '%s.pkl' % feature_name)
        if not os.path.exists(out_fname):
            prepare_click_time_delta_rev(df, group, out_fname)            
    
if __name__ == '__main__':
    run()
