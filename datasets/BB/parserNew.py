from __future__ import absolute_import
from __future__ import print_function

from six.moves import xrange
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from helpers.aux import *
from sys import exit
from os.path import join

import os


def getSentScores(x, xs, names, verbose=True):
    logger = './log.txt'
    sid = SentimentIntensityAnalyzer()
    y = []
    for _id in x.keys():
        s = x[_id]
        ss = sid.polarity_scores(s)
        y.append(ss)
        if verbose:
            print ('\n---')
            print(s)
            print ('Key: {}'.format(_id))
            for k in sorted(ss):
                print('{}: {}, '.format(k, ss[k]), end='')
            print()
    counts, stats = getSents(y)
    exit()
    r('--- Results of concatenated translations: ---', logger, 'w')
    r('Counts: ', logger)
    r(counts, logger)
    r('Percentage: ', logger)
    r(stats, logger)
    r('\n--- Individual translations: ---', logger)
    for idx, t in enumerate(xs):
        y = []
        for s in t.values():
            ss = sid.polarity_scores(s)
            y.append(ss)
        _, stats = getSents(y)
        r('\nResults of translation {}: '.format(names[idx]), logger)
        r(stats, logger)


def prepNew(dataset):
    direc = join(os.getcwd(), dataset)
    files = os.listdir(direc)
    nameIdx = xrange(len(files))
    nameDict = dict(zip(nameIdx, files))
    concatPairs = {}
    singlePairs = []
    for idx, trans in enumerate(files):
        print ('Processed {} translations...'.format(idx+1))
        print ('Translation: {}'.format(trans))
        tmp = {}
        with open(join(direc, trans), 'r') as f:
            vs = f.readlines()[10:]
            for v in vs:
                key, verse = v.split('\t')
                if int(key) < 40001001 or int(key) > 66022021:
                    continue
                else:
                    verse = verse[:-1] + '\n' # remove last \n
                    try:
                        concatPairs[int(key)] += verse
                    except KeyError:
                        concatPairs[int(key)] = verse
                    tmp[int(key)] = verse
        singlePairs.append(tmp)
    print ('* Finish producing verse dictionary...\n')
    return concatPairs, singlePairs, nameDict


if __name__ == '__main__':
    x, xs, names = prepNew('bibleeng')
    getSentScores(x, xs, names)
