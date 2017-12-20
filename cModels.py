from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from six.moves import xrange
from bModel import convModel

import numpy as np
import tensorflow as tf


class baseConvClassifier(convModel):
    def __init__(self,
                 bSize
                 emb,
                 mType='static',
                 seqLen=50,
                 numClass=2,
                 filterSizes=[2, 3, 4],
                 numFilters=128,
                 l2=0.1):
        super(baseConvClassifier, self).__init__(
            bSize, seqLen, numClass, filterSizes, numFilters, l2)
        self.inps = tf.placeholder(
            shape=[bSize, seqLen], dtype=tf.int32, name='inputSents')
        with tf.device('/cpu:0'):
            self.inpsEmb = tf.nn.embedding_lookup(emb, self.inps)
        self.tats = tf.placeholder(
            shape=[bSize, numClass], dtype=tf.int32, name='targets')
        self.buildGraph(mType)

    def buildGraph(self, mType):
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

        
