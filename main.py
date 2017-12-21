from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit

from utils.iterHelpers import *
from rModels import *
from cModels import *
from embLoader import *
from dataLoaders import *

import numpy as np
import tensorflow as tf

def master(sents, label):
    data = mrLoader(sents, label)
    emb = embLoader(50, 'glove', data.reVocab)
    model = baseConvClassifier(emb)
    train_op = model.getOps()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    while True:
        x, y = data.nextBatch()
        feed_dict = {model.inps: x, model.tats:y}
        _, batchLoss = sess.run([train_op, model.loss], feed_dict=feed_dict)
        print (batchLoss)



if __name__ == '__main__':
    pathP = 'datasets/MR/rt-polarity.pos'
    pathN = 'datasets/MR/rt-polarity.neg'
    sents, label = mrProcess(pathP, pathN)
    master(sents, label)
