from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from datetime import datetime

import time
import numpy as np

__all__ = ['getSents', 'r']


def getSents(scoreDicts):
    ret = {'positive': 0, 'negative': 0, 'neutral': 0}
    for _dict in scoreDicts:
        if _dict['compound'] >= 0.5:
            ret['positive'] += 1
        elif _dict['compound'] <= -0.5:
            ret['negative'] += 1
        else:
            ret['neutral'] += 1
    assert sum(ret.values()) == len(scoreDicts)
    return ret, {k: v / len(scoreDicts) for k, v in ret.iteritems()}


def r(s, path, mode='a', rt=False):
    if not isinstance(s, basestring):
        s = str(s)
    if rt:
        s = time.strftime("%Y:%m:%d %H:%M:%S") + '\t' + s
    print (s)
    with open(path, mode) as f:
        f.write(s+'\n')
