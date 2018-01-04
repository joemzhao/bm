from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from six.moves import xrange
from base.bModel import convModel

import numpy as np
import tensorflow as tf


__all__ = ['baseConvClassifier']


class baseConvClassifier(convModel):
    def __init__(self,
                 emb,
                 bSize=64,
                 mType='hybrid',
                 seqLen=100,
                 l2=0.1,
                 dropout=0.1,
                 numClass=2,
                 filterSizes=[2, 3, 4],
                 numFilters=64):
        super(baseConvClassifier, self).__init__(
            bSize, seqLen, filterSizes, numFilters, l2)
        self.inps = tf.placeholder(
            shape=[bSize, seqLen], dtype=tf.int32, name='inputSents')
        self.tats = tf.placeholder(
            shape=[bSize, numClass], dtype=tf.int32, name='targets')
        with tf.device('/cpu:0'):
            inpsEmb = tf.nn.embedding_lookup(emb.emb, self.inps)
        inpsEmb = tf.expand_dims(inpsEmb, -1)
        if mType == 'static':
            _c = 1
        elif mType == 'hybrid':
            assert emb.trainEmb == False
            _c = 2
            with tf.variable_scope('embeddings/'):
                dEmb = tf.Variable(initial_value=emb.emb, name='dembW', trainable=True)
            with tf.device('/cpu:0'):
                dynamicInpsEmb = tf.nn.embedding_lookup(dEmb, self.inps)
            dynamicInpsEmb = tf.expand_dims(dynamicInpsEmb, -1)
            inpsEmb = tf.concat([inpsEmb, dynamicInpsEmb], -1)
        else:
            raise Exception('Incorrect input number of channel. Require 1 or 2.')
        self.inpsEmb = inpsEmb
        self.buildGraph(_c, emb, dropout, numClass)

    def buildGraph(self, _c, emb, dropout, numClass):
        es = emb.embSize
        poolRes = []
        for i , fs in enumerate(self.fts):
            with tf.variable_scope('conv-maxpool-%s' % str(fs)):
                _shape = [fs, es, _c, self.nfs]
                W = tf.Variable(tf.truncated_normal(_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.nfs]), name='b')
                conv = tf.nn.conv2d(input=self.inpsEmb,
                    filter=W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                _pooled = tf.nn.max_pool(h, ksize=[1, self.sl-fs+1, 1, 1],
                    strides=[1, 1, 1, 1], padding='VALID', name="pool")
                poolRes.append(_pooled)
        _f = tf.reshape(tf.concat(poolRes, -1), [-1, len(self.fts)*self.nfs])
        _f = tf.nn.dropout(_f, 1.-dropout)
        l2_loss = tf.constant(0.0)
        with tf.variable_scope('projection/'):
            pW = tf.get_variable('W', shape=[len(self.fts)*self.nfs, numClass],
                initializer=tf.contrib.layers.xavier_initializer())
            pb = tf.Variable(tf.constant(0.1, shape=[numClass]), name='b')
        l2_loss += tf.nn.l2_loss(pW)
        l2_loss += tf.nn.l2_loss(pb)

        with tf.variable_scope('loss-computation/'):
            self.logits = tf.nn.xw_plus_b(_f, pW, pb)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.tats)) + self.l2*l2_loss
        self.pred = tf.argmax(self.logits, 1)
