from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from .iterHelpers import *

import abc
import random

__all__ = ['baseSentIter']


class baseSentIter(object):
    def __init__(self, batchSize, sents, label):
        self.b = batchSize
        self.x = sents
        self.y = label
        self.epoch = 0
        self.ptr = 0
        self.vocab = {}
        self.reVocab = None
        self.wd2id = lambda wds: [self.vocab[wd] for wd in wds]
        self.id2wd = lambda ids: [self.reVocab[id_] for id_ in ids]
        self.pad = lambda wds, ml: wds + ['<pad>'] * (ml - len(wds))
        self._doShuffle()

    @abc.abstractmethod
    def nextBatch(self):
        pass

    @abc.abstractmethod
    def buildVocab(self):
        pass

    def _doShuffle(self):
        zipped = list(zip(self.x, self.y))
        random.shuffle(zipped)
        self.x[:], self.y[:] = zip(*zipped)

    def _reset(self):
        self.ptr = 0
        self.epoch += 1
        self._doShuffle()
