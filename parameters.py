from __future__ import print_function
from __future__ import absolute_import

from os.path import join

import argparse


def getArgs():
	""" Specifying:
			Model parameters
			Training settings
			Directores of pretrained embeddings, data and saved model
	"""
	parser = argparse.ArgumentParser()

	""" Model type """
	parser.add_argument('--MODEL_TYPE', type=str, default='RCU')
	parser.add_argument('--CONV_TYPE', type=str, default='hybrid')

	""" Model parameters """
	parser.add_argument('--MSL', type=int, default=50)
	parser.add_argument('--RNN_LAYERS', type=int, default=1)
	parser.add_argument('--RNN_SIZE', type=int, default=64)
	parser.add_argument('--EMB_SIZE', type=int, default=50)
	parser.add_argument('--BATCH_SIZE', type=int, default=64)
	parser.add_argument('--MAX_VOCAB', type=int, default=50000)
	parser.add_argument('--OPT', type=str, default='adam')
	parser.add_argument('--LR', type=float, default=1e-3)
	parser.add_argument('--L2', type=float, default=.1)
	parser.add_argument('--DROP_OUT', type=float, default=.1)
	parser.add_argument('--GRAD_CLIP', type=float, default=5.)

	""" Training settings """
	parser.add_argument('--EVAL_EVERY', type=int, default=100)
	parser.add_argument('--SAVE_EVERY', type=int, default=5)
	parser.add_argument('--MAX_NUM_KEEP', type=int, default=100)

	""" PATH """
	parser.add_argument('--DATA_TYPE', type=str, default='MR')
	parser.add_argument('--EMB_TYPE', type=str, default='glove')

	args = parser.parse_args()
	print (50 * '~')
	print ('Using following settings in this training:')
	for k, v in vars(args).iteritems():
		print ('{} ---------> {}'.format(k, v))
	print (50 * '~')
	return args


if __name__ == '__main__':
	getArgs()
