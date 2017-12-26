from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from utils.iterators import *
from utils.iterHelpers import *
from overrides import overrides

import itertools
import numpy as np


__all__ = ['mrLoader', 'convMrLoader']


def getOneHot(y):
    return [[0, 1] if i == 0 else [1, 0] for i in y]


class mrLoader(baseSentIter):
    def __init__(self,
                 sents,
                 label,
                 batchSize=64,
                 sparseEmb=True):
        super(mrLoader, self).__init__(batchSize, sents, label)
        self.buildVocab(sparseEmb)

    def buildVocab(self, sparseEmb=True):
        aux = ['<s>', '</s>', '<unk>', '<pad>']
        tmp = aux + list(itertools.chain.from_iterable(self.x))
        for idx, wd in enumerate(tmp):
            if wd in self.vocab:
                continue
            else:
                self.vocab[wd] = len(self.vocab)
        self.reVocab = {v: k for k, v in self.vocab.iteritems()}
        del aux, tmp
        print('* Created vocabulary with type sparseEmb is ' + str(sparseEmb))

    def nextBatch(self):
        return self._nextBatch()

    def _nextBatch(self):
        if (self.ptr+self.b) < len(self.x):
            xRet = self.x[self.ptr:self.ptr+self.b]
            yRet = self.y[self.ptr:self.ptr+self.b]
        else:
            print ('Epoch %d finished!' % self.epoch)
            xRet = self.x[self.ptr:-1]
            yRet = self.y[self.ptr:-1]
            _sample = np.random.randint(0, len(self.x)-1, self.b-len(xRet))
            xRet.extend([self.x[i] for i in _sample])
            yRet.extend([self.y[i] for i in _sample])
            self._reset()
        ml = max([len(x) for x in xRet])
        xRet = [self.wd2id(self.pad(x, ml)) for x in xRet]
        self.ptr += self.b
        assert len(xRet) == self.b
        assert len(yRet) == self.b
        return np.asarray(xRet), np.asarray(getOneHot(yRet))


class convMrLoader(mrLoader):
    def __init__(self,
                 sents,
                 label,
                 maxSeqLen=100,
                 batchSize=64,
                 sparseEmb=True):
        super(convMrLoader, self).__init__(sents, label, batchSize, sparseEmb)
        self.msl = maxSeqLen
        self._push = lambda x: np.concatenate((x,
            self.vocab['<pad>']*np.ones((self.b, self.msl-x.shape[1]))), axis=1)
        self._pop = lambda x: x[:, :self.msl]

    @overrides
    def nextBatch(self):
        _x, yRet = self._nextBatch()
        xRet = self._push(_x) if _x.shape[1] <= self.msl else self._pop(_x)
        assert xRet.shape[1] == self.msl
        return xRet, yRet


if __name__ == '__main__':
    pathP = 'datasets/MR/rt-polarity.pos'
    pathN = 'datasets/MR/rt-polarity.neg'
    sents, label = mrProcess(pathP, pathN)
    mrLoader = convMrLoader(sents, label)
    while True:
        a, b = mrLoader.nextBatch()
        print (a)
        exit()
