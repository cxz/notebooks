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


def _datetime_to_deltas(series, delta=np.timedelta64(1, 's')):
    t0 = series.min()
    return ((series-t0)/delta).astype(np.uint32)

def _prepare_click_time_delta(df, group, column_name, dtype):
    """ Click_time difference between consecutive grouped rows.
    """
    
    logger.info('building {}'.format(column_name))
    
    df['t'] = _datetime_to_deltas(df.click_time)    
            
    # store last click by given group
    last_click = {}
    
    out = []
    
    #assuming input csv is ordered by click_time.
    for row in tqdm(zip(*([df['t']] + [df[g] for g in group])), total=len(df)): 
        t = row[0]
        k = tuple(row[1:])
        prev = last_click.get(k, -1)
        curr = t
        if prev != -1:
            delta = curr - prev # in seconds
            out.append(delta)
        else:
            out.append(MISSING)
        last_click[k] = curr
        
    diff = pd.DataFrame(out, columns=[column_name])
    diff.loc[diff[column_name] > 0, column_name] = np.log1p(diff.loc[diff[column_name] > 0, column_name])
    diff = diff.astype(dtype)
    return diff
        
    
def process(df, kind):

    for group in [
            ['ip', 'device'],
            #['ip', 'app', 'device'],
            #['ip', 'app', 'device', 'os'],
            #['ip', 'app', 'device', 'os', 'channel'],
    ]:
        out_column = 'delta_{}'.format('_'.join(group))
        out_fname = os.path.join(TMP, '{}_{}.feather'.format(kind, out_column))
        if os.path.exists(out_fname):
            continue

        print('preparing ', out_column, datetime.now())
        # using float instead of uint because log1p and -1 for missing.
        out = _prepare_click_time_delta(df, group, out_column, np.float32)

        feather.write_dataframe(out, out_fname)
        print('wrote ', out_fname)

        del out
        gc.collect()

        print('done ', datetime.now())
    
    
if __name__ == '__main__':
    df = feather.read_dataframe(os.path.join(TMP, 'test_delta_ip_device.feather'))
    print(df.info())
    print(df.delta_ip_device.describe())
