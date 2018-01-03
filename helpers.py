from __future__ import absolute_import
from __future__ import print_function

from sys import exit
from six.moves import cPickle
from os.path import join
from utils.aux import *

import shutil
import os


__all__ = ['saveModel', 'checkPath']


def saveModel(sess, paths, model, epochNum):
    route = join(paths.saved, 'epoch'+str(epochNum))
    try:
        os.makedirs(route)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    model.saver.save(sess, join(route, '_model.ckpt'))
    r('Model saved, epoch {}'.format(epochNum), paths.logger)


def checkPath(paths):
    print ('-'*25)
    if os.path.exists(paths.saved):
        print ('Starting a new expriment, removing following checkpoins...')
        for f in os.listdir(paths.saved):
            print (f)
            toremove = join(paths.saved, f)
            try:
                if os.path.isfile(toremove):
                    os.unlink(toremove)
                elif os.path.isdir(toremove):
                    shutil.rmtree(toremove)
            except Exception as e:
                print ('Exception when removing saved_model: ' + e)
    else:
        print ('Creating saved_models folder...')
