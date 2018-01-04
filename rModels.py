from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from six.moves import xrange
from base.bModel import rcuModel

import numpy as np
import tensorflow as tf


__all__ = ['naiveRecurrentClassifier']


class naiveRecurrentClassifier(rcuModel):
    """ A naive recurrent nn classifier for sentiment analysis. The last hiddent
    state, which encodes the context of the sentence untill time step t, is used
    to pass through dense layers to do classification.
    """
    def __init__(self,
                 emb,
                 bSize=64,
                 hSize=128,
                 dropout=.2,
                 layers=2,
                 ct='lstm',
                 nt='bi',
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
        with tf.device('/cpu:0'):
            self.inpsEmb = tf.nn.embedding_lookup(emb, self.inps)
        self.logits = self.buildGraph(nt, ct, layers, dropout, init, mode)
        self.pred = tf.argmax(self.logits, 1)
        self.loss = self._computeLoss(self.logits)

    def buildGraph(self, nt, ct, layers, dropout, init, mode):
        """ For classifier the final state of a cell is discarded.
        Toggle time_major=True if want more efficiency.
        """
        if nt == 'uni':
            c = self.defaultCell()
            with tf.variable_scope('encoding/'):
                outputs, _ = tf.nn.dynamic_rnn(cell=c,
                    inputs=self.inpsEmb, dtype=tf.float32, time_major=False)
        elif nt == 'bi':
            if layers == 1:
                cf = self.defaultCell()
                cb = self.defaultCell()
                with tf.variable_scope('encoding/'):
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cf,
                        cell_bw=cb, inputs=self.inpsEmb, dtype=tf.float32)
                    outputs = tf.concat(list(outputs), -1)
                self.h = 2 * self.h
            else:
                cfs = self._getCell(self.h, ct, dropout, layers, init, mode)
                cbs = self._getCell(self.h, ct, dropout, layers, init, mode)
                with tf.variable_scope('encoding/'):
                    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    cells_fw=cfs, cells_bw=cbs, inputs=self.inpsEmb, dtype=tf.float32)
                self.h = 2 * self.h
        with tf.variable_scope('attention/'):
            W = tf.get_variable('W', [1, self.h], dtype=tf.float32)
            Alpha = tf.nn.softmax(tf.reshape(
                tf.transpose(tf.squeeze(
                    tf.matmul(W, tf.transpose(tf.reshape(outputs, [-1, self.h]))),
                [0])), tf.gather(tf.shape(outputs), [0, 1])), dim=1)
            wh = tf.reduce_sum(tf.tile(
                tf.expand_dims(Alpha, -1), [1, 1, self.h]) * outputs, axis=1)
        wh = tf.layers.dense(inputs=wh, units=self.h/2, activation=tf.nn.tanh)
        logits = tf.layers.dense(inputs=wh, units=2, activation=None)
        return logits

    def _computeLoss(self, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.tats))
