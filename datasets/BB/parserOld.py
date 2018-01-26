from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helpers.aux import *
from parserNew import prepNew

import re
import numpy as np
import matplotlib.pyplot as plt

""" For Old Testament data only ..."""

def prepOld(path):
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


if __name__ == '__main__':
    pass
