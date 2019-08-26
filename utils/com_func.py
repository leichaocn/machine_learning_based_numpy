# -*- coding: utf-8 -*-
#!/usr/bin/env python
# File : com_func.py
# Date : 2019/8/13
# Author: leichao
# Email : leichaocn@163.com

"""简述功能.

详细描述.
"""

__filename__ = "com_func.py"
__date__ = 2019 / 8 / 13
__author__ = "leichao"
__email__ = "leichaocn@163.com"

import os
import sys

import pandas as pd
import numpy as np

def show(_obj):
    print('#'*30)
    if isinstance(_obj,np.ndarray):
        print('是np数组，形状为',_obj.shape)
        if _obj.shape[0]>=3:
            print('其head(3)数据为', _obj[:3])
        else:
            print('其全部数据为', _obj)

    if isinstance(_obj,list):
        the_len=len(_obj)
        print('是python数组，长度为',the_len)
        if the_len>=3:
            print('其头3个数据为', _obj[:3])
        else:
            print('其全部数据为', _obj)
