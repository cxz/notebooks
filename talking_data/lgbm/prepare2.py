""" 
    Build and cache train/test Dataframes.
"""

import os
from datetime import datetime

import pandas as pd
import feather


TMP = '/kaggle1/td-cache'

import prepare_count
import prepare_delta


def prepare(kind):
    print("loading ", datetime.now())
    df = feather.read_dataframe(os.path.join(TMP, '{}_base.feather'.format(kind)))
    print("done. ", datetime.now())
    
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]

    df['in_test_hh'] = (   3 
                         - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    
    prepare_count.process(df, kind)
    prepare_delta.process(df, kind)
        

if __name__ == '__main__':
    prepare('train')
    # prepare('test')