""" 
    Build and cache train/test Dataframes.
"""

import os
from datetime import datetime

import pandas as pd

from util import info

import prepare_base
import prepare_binip
import prepare_binip_lhood
import prepare_count
import prepare_delta
import prepare_mtbc
import prepare_lhood
import prepare_count_delta
import prepare_x1
import prepare_x2

def prepare(kind):
    prepare_base.process(kind)
    # prepare_binip(kind)
    prepare_count.process(kind)
    # prepare_x1.process(df, kind)
    # prepare_x2.process(df, kind)

if __name__ == '__main__':
    prepare('train')
    #prepare('test')