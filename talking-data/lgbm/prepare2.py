""" 
    Build and cache train/test Dataframes.
"""

import os
from datetime import datetime

import pandas as pd

from util import info

import prepare_base
import prepare_binip
import prepare_count
import prepare_x1
import prepare_x2
import prepare_x4

def prepare(kind):
    prepare_base.process(kind)
    prepare_binip.process(kind)
    prepare_count.process(kind)
    prepare_x1.process(kind)
    prepare_x2.process(kind)
    prepare_x4.process(kind)

if __name__ == '__main__':
    # prepare('train')
    prepare('test')