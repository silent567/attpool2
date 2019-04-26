#!/usr/bin/env python
# coding=utf-8

import numpy as np
from pygfl.easy import solve_gfl

def numpy_gfusedlasso(z,edge,lam=None):
    z_fused = solve_gfl(z.astype(np.float64),edge.astype('int'),lam=float(lam))
    return z_fused.astype(z.dtype)
