from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from os.path import join
from datetime import datetime

import time
import numpy as np


__all__ = ['modelName', 'toTime', 'r', 'accCal']


def modelName(m):
    return '{}.{}'.format(m.__module__, m.__class__.__name__)


def toTime(s, p):
    return datetime.strptime(s, p)


def r(s, path, mode='a', rt=True):
    if rt:
        s = time.strftime("%Y:%m:%d %H:%M:%S") + '\t' + s
    print (s)
    with open(path, mode) as f:
        f.write(s+'\n')


def accCal(pred, ground, l):
    assert l >= 0, 'L must >= 0.'
    pred = np.asarray(pred[:-l or None])
    ground = np.asarray(ground[:-l or None])
    return sum(abs(pred-ground))/len(pred)
