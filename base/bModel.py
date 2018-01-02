from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit

import numpy as np
import tensorflow as tf
import abc


class _base(object):
    def __init__(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

    @abc.abstractmethod
    def buildGraph(self):
        pass


    def getOps(self, lr=0.001, clip=1., opt='adam'):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(self.loss)
        with tf.variable_scope('grad_clip/'):
            clip_grads = [(tf.clip_by_norm(grad, clip), v) for grad, v in grads]
        train_op = opt.apply_gradients(clip_grads)
        return train_op


class rcuModel(_base):
    def __init__(self, hSize, ct, dropout, layers, init, mode):
        self.defaultCell = lambda: self._getCell(
            hSize, ct, dropout, layers, init, mode)

    def _getCell(self, hSize, ct, dropout, layers, init, mode):
        if ct == 'lstm':
            c = tf.contrib.rnn.LSTMCell(
                num_units=hSize, state_is_tuple=True, initializer=init)
        elif ct == 'gru':
            c = tf.contrib.rnn.GRUCell(num_units=hSize)
        else:
            raise Exception('Required cell type not supported!')
        cs = [c] * layers
        c = tf.contrib.rnn.MultiRNNCell(cells=cs, state_is_tuple=True)
        if mode == 'train' and dropout > 0:
            c = tf.contrib.rnn.DropoutWrapper(cell=c, dtype=tf.float32,
                input_keep_prob=1.-dropout, output_keep_prob=1.-dropout)
        return c


class convModel(_base):
    def __init__(self, bSize, seqLen, filterSizes, numFilters, l2):
        self.b = bSize
        self.sl = seqLen
        self.fts = filterSizes
        self.nfs = numFilters
        self.l2 = l2
