from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import deepdish as dd
import numpy as np
import tensorflow as tf

class embLoader(object):
    def __init__(self, embSize, embType, inVocab, trainEmb=False):
        notShown = 0
        if embType == 'glove':
            tmp = dd.io.load('embs/glove/embLookup50.sc')
            self.emb = []
            for idx in xrange(len(inVocab)):
                wd = inVocab[idx]
                try:
                    vec = tmp[wd]
                except KeyError:
                    notShown += 1
                    vec = np.random.uniform(
                        low=-1., high=1., size=(self.embSize))
                self.emb.append(vec)
            del tmp
            with tf.variable_scope('embeddings/'):
                self.emb = tf.Variable(initial_value=np.asarray(self.emb),
                    name='embW', trainable=trainEmb, dtype=tf.float32)
        elif embType == 'random':
            with tf.variable_scope('embeddings/'):
                self.emb = tf.get_variable(name='embW', dtype=tf.float32,
                shape=[len(inVocab), self.embSize], trainable=trainEmb,
                initializer=tf.truncated_normal_initializer())
        else:
            raise Exception('No supported embedding type!')

        print ('-' * 31)
        print ('Embedding type ' + str(embType) + ' ' + str(embSize))
        print ('Finish obtain glove embedding ' + str(embSize))
        print ('Not shown ' + str(notShown))
        print ('-' * 31)
