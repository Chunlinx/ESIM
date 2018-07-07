import sys
import os
import time

import numpy as np
import theano
import theano.tensor as tensor

import argparse
import logging

from lib import *
from model import *

def test(options):
	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s")
	hdlr = logging.FileHandler(os.path.join(options.folder, options.file_log), mode = "w")
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)

	logger.info("python %s" %(" ".join(sys.argv)))

	#################################################################################
	start_time = time.time()

	msg = "Loading dicts from %s..." %(options.file_dic)
	display(msg)
	word2idx, label2idx = dicfold(options.file_dic)

	msg = "Loading data from %s..." %(options.file_test)
	display(msg)
	test_x, test_y, test_z = datafold(options.file_test)

	end_time = time.time()

	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg)

	options.size_vocab = len(word2idx)
	options.num_class = len(label2idx)

	msg = "#inst in test %d" %(len(test_x))
	display(msg)
	msg = "#word vocab: %d" %(options.size_vocab)
	display(msg)
	msg = "#label: %d" %(options.num_class)
	display(msg)

	msg = "=" * 30 + "Hyperparameter:" + "=" * 30
	display(msg)
	for attr, value in sorted(vars(options).items(), key = lambda x: x[0]):
		msg = "{}={}".format(attr.upper(), value)
		display(msg)

	#################################################################################
	#msg = "=" * 30 + "model:" + "=" * 30
	#display(msg)
	if not os.path.exists(os.path.join(options.folder, options.reload_model)):
		msg = "Exception: The pretrained model doesn't exist."
		display(msg)
		sys.exit(0)

	model = Model(word2idx,
				  options.file_emb,
				  options.size_vocab,
				  options.dim_emb,
				  options.dim_proj,
				  options.num_class,
				  options.seed,
				  options.dropout_rate,
				  options.folder,
				  options.reload_model,
				  options.decay_c,
				  options.optimizer,
				  options.clip_c)

	msg = "=" * 30 + "Variable:" + "=" * 30
	display(msg)
	for k, v in model.params.items():
		msg = "name: {}, shape: {}".format(k, v.shape)
		display(msg)

	model.build_model()

	msg = "=" * 30 + "Compiling:" + "=" * 30
	display(msg)
	f_cost, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred = model.build_optimizer()

	#################################################################################
	msg = "=" * 30 + "Evaluating:" + "=" * 30
	display(msg)

	tst_data_iterator = DataIterator([test_x, test_y, test_z])

	model.use_noise.set_value(0.)
	
	start_time = time.time()
	tst_acc = evaluate(f_pred, os.path.join(options.folder, options.file_pred), tst_data_iterator, options)
	end_time = time.time()

	msg = "Test accuracy: %f" %tst_acc
	display(msg)

	msg = "Evaluating time: %f seconds" %(end_time - start_time)
	display(msg)

	msg = "Finished"
	display(msg)

def main(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument("--folder", help = "the dir of model", default = "workshop")
	parser.add_argument("--file_dic", help = "the file of vocabulary", default = "../data/test_dic.pkl")
	parser.add_argument("--file_test", help = "the file of testing data", default = "../data/test.pkl")
	parser.add_argument("--file_pred", help = "the file of pred data", default = "pred.txt")
	parser.add_argument("--file_log", help = "the log file", default = "test.log")
	parser.add_argument("--file_emb", help = "the file of pretrained embedding", default = "")
	parser.add_argument("--reload_model", help = "the pretrained model", default = "model.npz")
	
	parser.add_argument("--size_vocab", help = "the size of vocabulary", default = 10000, type = int)
	parser.add_argument("--dim_emb", help = "the dimension of the word embedding", default = 300, type = int)
	parser.add_argument("--dim_proj", help = "the dimension of the LSTM hidden", default = 300, type = int)
	parser.add_argument("--num_class", help = "the number of labels", default = 3, type = int)
	parser.add_argument("--seed", help = "random seed", default = 1234, type = int)

	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adam")
	parser.add_argument("--dropout_rate", help = "dropout rate", default = 0.5, type = float)
	parser.add_argument("--decay_c", help = "decay rate", default = 0.0, type = float)
	parser.add_argument("--clip_c", help = "grad clip", default = 10.0, type = float)
	parser.add_argument("--batch_size", help = "batch size", default = 32, type = int)
	parser.add_argument("--maxlen", help = "max length of sentence", default = None, type = int)

	options = parser.parse_args(argv)
	test(options)

if "__main__" == __name__:
	main(sys.argv[1:])