from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from nltk import word_tokenize

import re
import os

__all__ = ['mrProcess']


def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    return string.strip().lower()


def mrProcess(pathP, pathN, crossValid=4):
    sents = []
    label = []
    with open(pathP, 'r') as f:
        for line in f:
            line = word_tokenize(clean_string(line))
            sents.append(line)
            label.append(1)
    with open(pathN, 'r') as f:
        for line in f:
            line = word_tokenize(clean_string(line))
            sents.append(line)
            label.append(0)
    print ('-' * 31)
    print('Finished preprocessing of MR...')
    return sents, label
