from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helpers.aux import *

import re
import numpy as np
import matplotlib.pyplot as plt

""" For Old Testament data only ..."""

def prep(path):
    x = []
    _x = []
    tmp = ''
    with open(path, 'r') as f:
        lines = f.readlines()
        for idx, subline in enumerate(lines):
            tmp += subline
            if subline != '\r\n':
                continue
            else:
                _x.append(tmp)
                tmp = ''
    for idx, line in enumerate(_x):
        if idx % 10000 == 0:
            print ('Processed {} lines...'.format(idx))
        line = line.strip()
        line = re.sub(r'\d*:', '', line)
        line = re.sub(r'\b\d+\b', '', line)
        line = re.sub('\r\n', ' ', line)
        x.append(line)
    del tmp, _x
    return x


def generatePair(verbose=False, window=20):
    sid = SentimentIntensityAnalyzer()
    x = prep('./engraw.txt')
    y = []
    cs = []
    for s in x:
        ss = sid.polarity_scores(s)
        y.append(ss)
        if verbose:
            print ('---')
            print(s)
            for k in sorted(ss):
                print('{}: {}, '.format(k, ss[k]), end='')
            print()
    series, stats = getSents(y)
    rollmean = np.convolve(series, np.ones(window)/window, mode='same')
    print (stats)
    plt.plot(rollmean)
    plt.plot(np.zeros(len(rollmean)))
    plt.show()


if __name__ == '__main__':
    generatePair()
