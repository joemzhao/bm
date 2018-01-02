from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from os.path import join
from collections import namedtuple

from utils.iterHelpers import *
from rModels import *
from cModels import *
from embLoader import *
from dataLoaders import *
from parameters import getArgs

import numpy as np
import tensorflow as tf

paths = namedtuple('paths', 'root data emb saved')


def master(args, trainLoader, evalLoader, emb, model):
    """ Given model and data, controlling training procedure """
    print ('* Finish defining model. Start training...\n')
    train_op = model.getOps()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    while True:
        x, y = trainLoader.nextBatch()
        feed_dict = {model.inps: x, model.tats:y}
        _, batchLoss = sess.run([train_op, model.loss], feed_dict=feed_dict)
        print (batchLoss)


def main(args, paths):
    """ Obtaining model, embedding and dataloaders based on specifiaction """
    dPath = join(paths.data, args.DATA_TYPE)
    trainS, trainL = mrProcess(join(dPath, 'train.pos'), join(dPath, 'train.neg'))
    testS, testL = mrProcess(join(dPath, 'test.pos'), join(dPath, 'test.neg'))

    if args.MODEL_TYPE == 'CONV':
        trainLoader = convMrTrainLoader(
            trainS, trainL, args.MSL, args.BATCH_SIZE, sparseEmb=True)
        evalLoader = convMrEvalLoader(
            testS, testL, args.MSL, args.BATCH_SIZE, trainLoader.vocab)
        emb = embLoader(
            args.EMB_SIZE, args.EMB_TYPE, trainLoader.reVocab)
        model = baseConvClassifier(
            emb, args.BATCH_SIZE, args.CONV_TYPE, args.MSL, args.L2, args.DROP_OUT)
    elif args.MODEL_TYPE == 'RCU':
        trainLoader = mrTrainLoader(
            trainS, trainL, args.BATCH_SIZE, sparseEmb=True)
        evalLoader = mrEvalLoader(
            testS, testL, args.BATCH_SIZE, trainLoader.vocab)
        emb = embLoader(
            args.EMB_SIZE, args.EMB_TYPE, trainLoader.reVocab)
        model = naiveRecurrentClassifier(
            emb.emb, args.BATCH_SIZE, args.RNN_SIZE, args.DROP_OUT, args.RNN_LAYERS)
    else:
        raise Exception('Not supported yet!')
        exit()
    master(args, trainLoader, evalLoader, emb, model)


if __name__ == '__main__':
    ROOT_DIR = '/Users/mzhao/Desktop/major/bm'
    ALL_DATA = join(ROOT_DIR, 'datasets/')
    ALL_EMB = join(ROOT_DIR, 'embs/')
    SAVE_MODEL = join(ROOT_DIR, 'saved_models/')
    main(getArgs(), paths(ROOT_DIR, ALL_DATA, ALL_EMB, SAVE_MODEL))
