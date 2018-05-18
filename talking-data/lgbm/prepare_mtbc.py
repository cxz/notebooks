""" 

    Build and cache input.


"""

import os
import gc
import pickle
import logging
from datetime import datetime
import logging


import pandas as pd
import numpy as np
import feather

from tqdm import tqdm

TMP = '/kaggle1/td-cache'

logging.basicConfig(level=logging.DEBUG, format='%(pathname)s %(asctime)s %(levelname)s %(message)s',)
logger = logging.getLogger('prepare')

MISSING = -1


def _prepare_mtbc(df, group, column_name, delta_column, dtype):
    """ Mean deltas
    """
    
    logger.info('building {}'.format(column_name))
    
    df = df.loc[df[delta_column] != -1, :]
    df[delta_column] = np.expm1(df[delta_column])
    
    gp = df[group + [delta_column]].groupby(by=group)[delta_column] \
        .mean() \
        .reset_index() \
        .rename(index=str, columns={delta_column: column_name})
    out = df[group].merge(gp, on=group, how='left')
    out[column_name] = out[column_name].fillna(-1)
    out.loc[out[column_name] > 0, column_name] = np.log1p(out.loc[out[column_name] > 0, column_name])
    out = out[[column_name]].astype(dtype)
    gc.collect()
    return out
    
def process(df, kind):

    for group in [
            #['ip', 'device'],
            ['ip', 'device', 'os', 'channel'],
            #['ip', 'app', 'device'],
            #['ip', 'app', 'device', 'os'],
            #['ip', 'app', 'device', 'os', 'channel'],
    ]:
        delta_column = 'delta_{}'.format('_'.join(group))
        delta_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, delta_column))
        if not os.path.exists(delta_fname):
            raise
        delta = feather.read_dataframe(delta_fname)
        if delta_column not in delta.columns:
            print(delta.columns)
        df[delta_column] = delta[delta_column]
        
        out_column = 'mtbc_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        # using float instead of uint because log1p and -1 for missing.
        out = _prepare_mtbc(df, group, out_column, delta_column, np.float32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
    
    
if __name__ == '__main__':
    # sanity check
    for kind in ['train', 'test']:
        df = feather.read_dataframe(os.path.join(TMP, '{}_mtbc_ip_device.feather'.format(kind)))
        print('---')
        print(df.head(10))
        print('---')
        print(df.tail(10))
        print('---')
              
        print(df.mtbc_ip_device.describe())
