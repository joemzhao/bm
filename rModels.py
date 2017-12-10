from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from bModel import rnnModel

import numpy as np
import tensorflow as tf

class naiveRecurrentClassifier(rnnModel):
    """ A naive recurrent nn classifier for sentiment analysis. The last hiddent
    state, which encodes the context of the sentence untill time step t, is used
    to pass through dense layers to do classification.
    """
    def __init__(self,
                 emb,
                 bSize=128
                 hSize=128,
                 ct='lstm',
                 nt='uni'
                 dropout=.2,
                 layers=1,
                 init=None,
                 mode='train'):
        super(naiveRecurrentClassifier, self).__init__(
            hSize, ct, dropout, layers, init, mode)
        self.b = bSize
        self.h = hSize
        self.inps = tf.placeholder(
            shape=[bSize, None], dtype=tf.int32, name='encoderInps')
        self.tats = tf.placeholder(
            shape=[bSize, 2], dtype=tf.int32, name='targets')
        self.inpts_emb = tf.nn.embedding_lookup(emb, self.inps)
        self.buildGraph(nt, ct, layers, dropout, init, mode)

    def buildGraph(self, nt, ct, layers, dropout, init, mode):
        if nt == 'uni':
            c = self.defaultCell()
            with tf.variable_scope('encoding/'):
                res = tf.nn.dynamic_rnn(cell=c, inputs=self.inps,
                    dtype=tf.float32, time_major=False)
        elif nt == 'bi':
            if layers == 1:
                cf = self.defaultCell()
                cb = self.defaultCell()
                with tf.variable_scope('encoding/'):
                    res = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cf, cell_bw=cb, inputs=self.inps, dtype=tf.float32)
            else:
                cfs = self.getCell(self.h, ct, dropout, layers, init, mode)
                cbs = self.getCell(self.h, ct, dropout, layers, init, mode)
                with tf.variable_scope('encoding/'):
                    res = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=cfs, cells_bw=cbs, inputs=self.inps, dtype=tf.float32)
        outputs = res[0]
