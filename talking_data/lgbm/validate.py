"""
    Validation playground.


run14: added x4 with all days except 8 ((df.day.isin([6, 7, 9])) & df.hour.isin([4,5,9,10,13,14]))

[85]    train's auc: 0.981486   h4's auc: 0.969668      h5's auc: 0.973027      h9's auc: 0.971214      h10's auc: 0.971095     h13's auc: 0.977906     h14's auc: 0.982335
[('x2_app_count_channel', 0), ('x2_ip_app_count_channel', 0), ('x2_ip_app_os_var_hour', 2), ('x2_ip_app_channel_mean_hour', 4), ('x2_ip_app_os_count_channel', 6), ('count_ip_app_channel_hour', 10), ('count_ip_app_hour', 10), ('x4_binip_hour', 10), ('x1_app_os_hour', 16), ('x4_binip_app_channel_hour', 17), ('count_ip_os_hour', 18), ('x1_app_channel_hour', 18), ('x4_binip_app_os', 18), ('device', 19), ('x1_app_channel', 21), ('x1_ip', 22), ('x1_app_device', 23), ('count_ip_device_hour', 27), ('x4_binip_app_os_hour', 31), ('hour', 35), ('x1_app_os', 46), ('count_ip_day_app_hour', 54), ('x4_binip_app_channel', 58), ('app', 84), ('count_ip_day_hour', 102), ('binip', 119), ('os', 132), ('channel', 288)]


run 15: 
[63]    train's auc: 0.978018   h4's auc: 0.968397      h5's auc: 0.973441      h9's auc: 0.972321      h10's auc: 0.971292     h13's auc: 0.978662     h14's auc: 0.982767
[100]   train's auc: 0.980078   h4's auc: 0.969854      h5's auc: 0.974957      h9's auc: 0.969743      h10's auc: 0.971118     h13's auc: 0.97577      h14's auc: 0.982264
[('count_ip_app_channel_hour', 0), ('count_ip_day_app_hour', 0), ('count_ip_day_hour', 0), ('x2_ip_app_channel_mean_hour', 1), ('count_ip_app_hour', 2), ('x2_app_count_channel', 2), ('x2_ip_app_os_var_hour', 2), ('count_ip_os_hour', 3), ('x4_binip_hour', 8), ('x2_ip_app_os_count_channel', 10), ('x1_app_os_hour', 12), ('device', 14), ('x1_app_device', 18), ('x2_ip_app_count_channel', 18), ('count_ip_device_hour', 20), ('x1_app_channel_hour', 23), ('x4_binip_app_os_hour', 24), ('hour', 25), ('x4_binip_app_channel_hour', 25), ('x4_binip_app_os', 26), ('x1_app_channel', 28), ('x1_app_os', 43), ('x1_ip', 46), ('app', 54), ('x4_binip_app_channel', 79), ('binip', 82), ('os', 96), ('channel', 221)]

run16:
[70]    train's auc: 0.980282   h4's auc: 0.971428      h5's auc: 0.975672      h9's auc: 0.974172      h10's auc: 0.971081     h13's auc: 0.978723     h14's auc: 0.983168
params = { 'num_leaves': 63, 'max_depth': 5, 'max_bin': 255}
[57]    train's auc: 0.978996   h4's auc: 0.971117      h5's auc: 0.975262      h9's auc: 0.973577      h10's auc: 0.972468     h13's auc: 0.980032     h14's auc: 0.98388
[('count_ip_app_channel_hour', 3), ('count_ip_app_hour', 6), ('count_ip_os_hour', 6), ('x2_ip_app_os_var_hour', 9), ('x2_app_count_channel', 10), ('x4_binip_hour', 12), ('x2_ip_app_channel_mean_hour', 14), ('x2_ip_app_os_count_channel', 20), ('x1_app_os_hour', 21), ('x1_app_device', 30), ('count_ip_device_hour', 32), ('x4_binip_app_channel_hour', 35), ('device', 36), ('hour', 39), ('x2_ip_app_count_channel', 39), ('x1_app_channel_hour', 43), ('x4_binip_app_os_hour', 44), ('x1_app_channel', 54), ('x4_binip_app_os', 57), ('x1_app_os', 73), ('x1_ip', 89), ('x4_binip_app_channel', 105), ('app', 127), ('binip', 218), ('os', 243), ('channel', 402)]

"""


import os
import gc
import pickle
import logging
import datetime
import operator
from datetime import datetime 

import pandas as pd
import numpy as np
import lightgbm as lgb

import data2
import train2 

from dataset import *
from util import info

SEED = 0

from sklearn.preprocessing import RobustScaler 

if __name__ == '__main__':
    trainval_df = data2.load('train')
    
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'binip']

    # faster feedback
    train_cond = (trainval_df.day == 8) & (trainval_df.hour.isin([4,5,9,10,13,14]))
    train_df = trainval_df[train_cond] 
    #train_df = trainval_df.iloc[:-VALID_ROWS]
        
    #info('shuffling train')
    #train_df = train_df.iloc[np.random.permutation(len(train_df))]
        
    # used to save memory only, as when building lgbm dataset we specify
    # columns to be used explicitly
    excluded = [
        'click_time',
        'ip',
        'day'
    ]
    
    for column in excluded:
        del trainval_df[column]
        
    #apply scaler to x1
    
    predictors = list(sorted([c for c in trainval_df.columns if c not in ['is_attributed']]))
    
    gc.collect()
             
    val_df = trainval_df.iloc[-VALID_ROWS:]
    
    val_dfs = {
        'h4':  val_df[val_df.hour ==  4],
        'h5':  val_df[val_df.hour ==  5],
        'h9':  val_df[val_df.hour ==  9],
        'h10': val_df[val_df.hour == 10],
        'h13': val_df[val_df.hour == 13],
        'h14': val_df[val_df.hour == 14]
    }
    
    params = {
        'num_leaves': 63                                                                                                                                                                            ,
        'max_depth': 5,
        'max_bin': 255        
    }
    
    info('predictors: %s' % ','.join(predictors))
    
    iterations = 500
    train2.run(train_df, val_dfs, 
        predictors, target, categorical, 
        iterations, SEED, param_overrides=params)
    
    
