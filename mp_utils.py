#!/usr/bin/env python
# coding=utf-8

'''
Global multiprocessing utilization, setting global pool for both multiprocessing and torch.multiprocessing
**This package should be imported before everything!**
'''

import sys as _sys
if 'tensorflow' in _sys.modules:
    raise ValueError('Wrong mp_utils file imported as this one only interacts with pytorch programs')
if 'torch' in _sys.modules:
    raise ValueError('mp_utils should be imported before torch')

PROCESS_NUM = 8
import multiprocessing as _tmp
# import torch.multiprocessing as _tmp
# import multiprocessing.dummy as _tmp
# try:
    # _tmp.set_start_method('spawn')
# except:
    # pass
_torch_pool = _tmp.Pool(PROCESS_NUM)

def get_torch_pool():
    return _torch_pool
    # return _tmp.Pool(PROCESS_NUM)

