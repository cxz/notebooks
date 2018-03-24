import os
import gc
import pickle
import logging
import datetime
import json

import pandas as pd
import numpy as np

TRAIN_ROWS = 184903890
VALID_ROWS = 53016937 # rows in train.csv with day == 2017-11-09             
TEST_ROWS_V0 = 57537505
TEST_ROWS = 18790469
CACHE = '../cache'

logger = logging.getLogger('')

def load():    
    """ Load train + val + test df. 
    """
    with open(os.path.join(CACHE, 'train_test_base.pkl'), 'rb') as f:
        logger.info('loading base')
        df = pickle.load(f)        
        df = df.reset_index(drop=True)
        assert len(df) == TRAIN_ROWS + TEST_ROWS_V0, \
            "%d %d" %(len(df), TRAIN_ROWS + TEST_ROWS_V0)
        
    for extra in [
        'count_ip-app-os.pkl',
        'count_ip-app.pkl',
        'count_ip-channel.pkl',
        'count_ip-day-app.pkl',
        'count_ip-day-channel.pkl',
        'count_ip-day-device.pkl',
        'count_ip-day-hour.pkl',
        'count_ip-day-os.pkl',
        'count_ip-hour-device.pkl',
        #'count_ip-hour-channel.pkl',
        'count_ip-hour-app.pkl',
        'count_ip-hour-os.pkl',
        'count_ip-device-channel.pkl',
        'count_ip-device.pkl',
        'count_ip-os.pkl',
        'count_os-device-app.pkl',
        'count_os-device-channel.pkl',
        #'delta_ip-app.pkl',
        #'delta_ip-device-app.pkl',
        'delta_ip-device-channel.pkl',
        'delta_ip-device-os-app.pkl',
        #'delta_ip-device-os.pkl',
        #'delta_rev_ip-device.pkl',
        #'delta_rev_ip-device-app.pkl',
        #'delta_rev_ip-device-channel.pkl'
        'count_hour-app.pkl',
        'count_hour-device.pkl',
        'count_hour-channel.pkl',
        #'cps_app.pkl',
        #'cps_app-channel.pkl',
    ]: 
        with open(os.path.join(CACHE, extra), 'rb') as f:
            logger.info('loading %s' % extra)
            df2 = pickle.load(f)
        for c in df2.columns:
            df[c] = df2[c]
        del df2
        gc.collect()
        
    #with open('../cache/ip_mapping.pkl', 'rb') as f:
    #    mapped, _, _, _ = pickle.load(f)
    #    mapped = np.array(mapped)
            
    logger.info('done loading extra features.')
    ips = df[['ip']].values
    
    df.drop(['click_time', 'ip', 'day'], axis=1, inplace=True)
    categorical = ['app', 'channel', 'device', 'os', 'hour']
    
    # lgbm recognizes automatically if the column type is category
    for c in categorical:
        df[c] = df[c].astype('category')
        
    return df, ips


def load_sample():
    df, _ = load()
        
    logger.info(df.info())

    valid_rows = 10000000
    train_rows = 50000000
    train_df = df.iloc[TRAIN_ROWS-train_rows-valid_rows:TRAIN_ROWS-valid_rows, :]
    valid_df = df.iloc[TRAIN_ROWS-valid_rows:TRAIN_ROWS, :]        

    del df
    gc.collect()
    
    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
    return train_df, valid_df, test_df


def load_train_val_splits(use_validation=True):
    df, _ = load()
        
    logger.info(df.info())

    if use_validation:
        train_df = df.iloc[:TRAIN_ROWS-VALID_ROWS, :]
        valid_df = df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS, :]
        # test_df = df.iloc[-TEST_ROWS_V0:]
        test_df = None        
        
    else:
        train_df = df.iloc[:TRAIN_ROWS, :]
        valid_df = []
        test_df = None

    del df
    gc.collect()
    
    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
    return train_df, valid_df, test_df

def load_train_val_splits2(common=False):
    """ No common ips between train & val sets
    """
    df, ips = load()        
    logger.info(df.info())
    
    with open('../cache/ip_mapping.pkl', 'rb') as f:
        mapped, _, _, _ = pickle.load(f)
        mapped = np.array(mapped)
        
    #assert len(ips) == len(mapped), 
    #    'len(ips)==%d, len(mapped)==%d' %(len(ips), len(mapped))

    # reserve ips for val set
    ips = np.array(ips)
    val_ips = ips[mapped==3]  # ips in common between trani & test
    #val_ips = ips[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS]
    
    
    #val_ips = ips[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS]
    unique_val_ips = np.unique(val_ips)
    logger.info("unique val ips: %d" % len(unique_val_ips))
    
    sample_size = int(len(unique_val_ips) * 0.5)
    val_ips = np.random.choice(unique_val_ips, size=(sample_size,), replace=False)
    logger.info("selected val ips: %d" % len(val_ips))
        
    # exclude reserved ips from train set
    val_idx = np.isin(ips[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS], val_ips)
    if common:
        train_idx = np.isin(ips[:TRAIN_ROWS-VALID_ROWS], val_ips)        
    else:
        train_idx = ~np.isin(ips[:TRAIN_ROWS-VALID_ROWS], val_ips)
        
    train_df = df.iloc[:TRAIN_ROWS-VALID_ROWS, :]
    train_df = train_df.iloc[train_idx.ravel(), :]

    valid_df = df.iloc[TRAIN_ROWS-VALID_ROWS:TRAIN_ROWS, :]
    valid_df = valid_df.iloc[val_idx.ravel(), :]

    logger.info("train: %d, valid: %d" % (len(train_df), len(valid_df)))
    return train_df, valid_df, None
    

def load_click_ids():
    click_ids = pd.read_csv('../input/test_v0.csv', usecols=['click_id']).click_id.values
    return click_ids

def load_test_split():
    df, _ = load()
        
    logger.info(df.info())

    test_df = df.iloc[-TEST_ROWS_V0:]
    click_ids = load_click_ids()

    del df
    gc.collect()
    
    logger.info("test: %d" % len(test_df))
    return click_ids, test_df