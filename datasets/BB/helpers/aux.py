from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['getSents']


def getSents(scoreDicts):
    series = []
    ret = {'positive': 0, 'negative': 0, 'neutral': 0}
    for _dict in scoreDicts:
        series.append(_dict['compound'])
        if _dict['compound'] >= 0.5:
            ret['positive'] += 1
        elif _dict['compound'] <= -0.5:
            ret['negative'] += 1
        else:
            ret['neutral'] += 1
    assert sum(ret.values()) == len(scoreDicts)
    return series, {k: v / len(scoreDicts) for k, v in ret.iteritems()}
