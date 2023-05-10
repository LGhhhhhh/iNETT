# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:12:50 2019

@author: admin
"""

import numpy as np
def wgn_fun(x, per):
    return x*(1+per*np.random.randn(np.shape(x)[0], np.shape(x)[1]))