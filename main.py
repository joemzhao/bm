from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from os.path import join
from collections import namedtuple

from utils.iterHelpers import *
from utils.aux import *
from helpers import *
from rModels import *
from cModels import *
from embLoader import *
from dataLoaders import *
from parameters import getArgs

import time
import numpy as np
import tensorflow as tf

paths = namedtuple('paths', 'root data emb saved logger')
tf.set_random_seed(3)
np.random.seed(3)


def evaluate(paths, sess, evalLoader, model, epochNum):
    _l = -1
    predict = []
    groundt = []
    while _l < 0:
        x, y, _l = evalLoader.nextBatch()
        feed_dict = {model.inps: x}
        pred = sess.run(model.pred, feed_dict=feed_dict)
        predict.extend(pred.tolist())
        groundt.extend(y)
    acc = accCal(predict, groundt, _l)
    r('Evaluating accuracy: {}'.format(acc), paths.logger)


def master(args, paths, trainLoader, evalLoader, model):
    """ Given model and data, controlling training procedure """
    r('-'*25, paths.logger, rt=False)
    r('Finish defining model. Start training...', paths.logger)
    train_op = model.getOps()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epochLosses = []
    epochLoss = 0.
    stepNum = 0
    epochNum = 0
    while epochNum < args.MAX_EPOCH:
        stepNum += 1
        x, y = trainLoader.nextBatch()
        feed_dict = {model.inps: x, model.tats:y}
        _, batchLoss = sess.run([train_op, model.loss], feed_dict=feed_dict)
        epochLoss += batchLoss
        if epochNum != trainLoader.epoch:
            epochNum += 1
            epochLosses.append(epochLoss)
            r('Done epoch {}, loss: {}'.format(epochNum, epochLoss), paths.logger)
            epochLoss = 0.
            if epochNum % args.EVAL_EVERY == 0:
                r('$'*25, paths.logger)
                saveModel(sess, paths, model, epochNum)
                r('$'*25, paths.logger)
                evaluate(paths, sess, evalLoader, model, epochNum)


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
    r('Start logging...', paths.logger, 'w')
    r('-'*25, paths.logger, rt=False)
    for k, v in vars(args).iteritems():
        r(str(k) + ':  ' + str(v), paths.logger, rt=False)
    checkPath(paths)
    master(args, paths, trainLoader, evalLoader, model)


if __name__ == '__main__':
    ROOT_DIR = '/Users/mzhao/Desktop/master/bm'
    ALL_DATA = join(ROOT_DIR, 'datasets/')
    ALL_EMB = join(ROOT_DIR, 'embs/')
    SAVE_MODEL = join(ROOT_DIR, 'saved_model/')
    LOGGER = join(ROOT_DIR, 'logger.txt')
    main(getArgs(), paths(ROOT_DIR, ALL_DATA, ALL_EMB, SAVE_MODEL, LOGGER))
