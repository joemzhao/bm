from __future__ import print_function
from __future__ import absolute_import

from os.path import join
from datetime import datetime

import time


__all__ = ['modelName', 'toTime', 'r']


def modelName(m):
    return '{}.{}'.format(m.__module__, m.__class__.__name__)


def toTime(s, p):
    return datetime.strptime(s, p)


def r(s, path, mode, rt=True):
    if rt:
        s = time.strftime("%Y:%m:%d %H:%M:%S") + '\t' + s
    print (s)
    with open(path, mode) as f:
        f.write(s+'\n')
