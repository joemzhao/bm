from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from six.moves import xrange
from bModel import convModel

import numpy as np
import tensorflow as tf


__all__ = ['baseConvClassifier']


class baseConvClassifier(convModel):
    def __init__(self,
                 emb,
                 bSize=64,
                 mType='hybrid',
                 seqLen=100,
                 numClass=2,
                 filterSizes=[2, 3, 4],
                 numFilters=64,
                 l2=0.1,
                 dropout=0.1):
        super(baseConvClassifier, self).__init__(
            bSize, seqLen, filterSizes, numFilters, l2)
        self.inps = tf.placeholder(
            shape=[bSize, seqLen], dtype=tf.int32, name='inputSents')
        with tf.device('/cpu:0'):
            self.inpsEmb = tf.nn.embedding_lookup(emb.emb, self.inps)
        self.tats = tf.placeholder(
            shape=[bSize, numClass], dtype=tf.int32, name='targets')
        self.buildGraph(mType, emb, dropout, numClass)

    def buildGraph(self, mType, emb, dropout, numClass):
        es = emb.embSize
        self.inpsEmb = tf.expand_dims(self.inpsEmb, -1)
        if mType == 'static':
            _c = 1
        elif mType == 'hybrid':
            assert emb.trainEmb == False
            dynamicEmb = tf.Variable(initial_value=self.inpsEmb, trainable=True)
            self.inpsEmb = tf.concat([self.inpsEmb, dynamicEmb], axis=-1)
            _c = 2
        else:
            raise Exception('Incorrect input number of channel. Require 1 or 2.')
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
            self.pred = tf.argmax(self.logits, 1)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.tats)) + self.l2*l2_loss

        with tf.variable_scope('evaluation/'):
            corr = tf.equal(self.pred, tf.argmax(self.tats, 1))
            self.acc = tf.reduce_mean(tf.cast(corr, tf.float32))

    def getOps(self, lr=0.001, clip=1., opt='adam'):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(self.loss)
        with tf.variable_scope('grad_clip'):
            clip_grads = [(tf.clip_by_norm(grad, clip), v) for grad, v in grads]
        train_op = opt.apply_gradients(clip_grads)
        return train_op
