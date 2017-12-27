from __future__ import absolute_import
from __future__ import print_function

import abc
import random


__all__ = ['baseTrainIter', 'baseEvalIter']


class _baseIter(object):
    def __init__(self, bSize, sents, label):
        self.b = bSize
        self.x = sents
        self.y = label
        self.ptr = 0

    @abc.abstractmethod
    def nextBatch(self):
        pass


class baseTrainIter(_baseIter):
    def __init__(self, bSize, sents, label):
        super(baseTrainIter, self).__init__(bSize, sents, label)
        self.epoch = 0
        self.vocab = {}
        self.reVocab = None
        self.wd2id = lambda wds: [self.vocab[wd] for wd in wds]
        self.id2wd = lambda ids: [self.reVocab[id_] for id_ in ids]
        self.pad = lambda wds, ml: wds + ['<pad>'] * (ml - len(wds))
        self._doShuffle()

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


class baseEvalIter(_baseIter):
    def __init__(self, bSize, sents, label, vocab):
        super(baseEvalIter, self).__init__(bSize, sents, label)
        self.wd2id = lambda wds: [vocab[wd] for wd in wds]
        self.pad = lambda wds, ml: wds + ['<pad>'] * (ml - len(wds))
