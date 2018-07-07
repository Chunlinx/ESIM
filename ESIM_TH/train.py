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

def train(options):
	if not os.path.exists(options.folder):
		os.mkdir(options.folder)

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

	msg = "Loading data from %s..." %(options.file_train)
	display(msg)
	train_x, train_y, train_z = datafold(options.file_train)

	msg = "Loading data from %s..." %(options.file_dev)
	display(msg)
	dev_x, dev_y, dev_z = datafold(options.file_dev)

	msg = "Loading data from %s..." %(options.file_test)
	display(msg)
	test_x, test_y, test_z = datafold(options.file_test)

	end_time = time.time()

	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg)

	options.size_vocab = len(word2idx)
	options.num_class = len(label2idx)

	if options.devFreq == -1:
		options.devFreq = (len(train_x) + options.batch_size - 1) / options.batch_size

	msg = "#inst in train: %d" %(len(train_x))
	display(msg)
	msg = "#inst in dev %d" %(len(dev_x))
	display(msg)
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
	msg = "=" * 30 + "Optimizing:" + "=" * 30
	display(msg)

	tra_data_iterator = DataIterator([train_x, train_y, train_z])
	tra_data_iterator_beta = DataIterator([train_x, train_y, train_z])
	dev_data_iterator = DataIterator([dev_x, dev_y, dev_z])
	tst_data_iterator = DataIterator([test_x, test_y, test_z])

	wait_counter = 0
	bad_counter = 0	
	estop = False

	dev_acc_record = []
	tst_acc_record = []
	lr_change_list = []

	best_acc = 0.0
	best_epoch = 0

	n_updates = 0
	
	start_time = time.time()

	while True:
		x, x_mask, y, y_mask, z = tra_data_iterator.next_batch(options.batch_size, max_len = options.maxlen, max_epoch = options.nepochs, is_train = True, shuffle = True)

		if x is None:
			break

		n_updates += 1

		model.use_noise.set_value(1.)

		disp_start = time.time()

		cost = f_cost(x, y, x_mask, y_mask, z)
		f_update(options.lr)

		disp_end = time.time()

		if numpy.isnan(cost) or numpy.isinf(cost):
			msg = "NaN detected"
			display(msg)
			sys.exit(0)

		if numpy.mod(n_updates, options.dispFreq) == 0:
			msg = "Epoch: %d, Update: %d, Cost: %f, Grad: %f, Time: %.2f sec" %(tra_data_iterator.epoch, n_updates, cost, f_grad_norm(x, y, x_mask, y_mask, z), (disp_end-disp_start))
			display(msg)

		if numpy.mod(n_updates, options.devFreq) == 0:
			msg = "=" * 30 + "Evaluating" + "=" * 30
			display(msg)
			model.use_noise.set_value(0.)
			dev_acc = evaluate(f_pred, os.path.join(options.folder, "current_dev_result"), dev_data_iterator, options)
			dev_data_iterator.reset()
			tst_acc = evaluate(f_pred, os.path.join(options.folder, "current_test_result"), tst_data_iterator, options)
			tst_data_iterator.reset()

			msg = "dev accuracy: %f" %dev_acc
			display(msg)
			msg = "test accuracy: %f" %tst_acc
			display(msg)
			msg = "lrate: %f" %options.lr
			display(msg)

			dev_acc_record.append(dev_acc)
			tst_acc_record.append(tst_acc)

			if dev_acc > best_acc:
				best_acc = dev_acc
				best_p = unzip(model.tparams)
				best_epoch = tra_data_iterator.epoch
				wait_counter = 0

				msg = "Saving model..."
				display(msg)

				numpy.savez(os.path.join(options.folder, options.saveto), **best_p)
				pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
				
				msg = "Model has been saved to {}".format(os.path.join(options.folder, options.saveto))
				display(msg)
			else:
				wait_counter += 1

			if wait_counter >= options.wait_N:
				msg = "wait_counter max, need to half the lr"
				display(msg)
				bad_counter += 1
				wait_counter = 0
				msg = "bad_counter: %d" %bad_counter
				display(msg)
				options.lr *= 0.5
				lr_change_list.append(tra_data_iterator.epoch)
				msg = "lrate change to: %f" %(options.lr)
				display(msg)
				zipp(best_p, model.tparams)

			if bad_counter > options.patience:
				msg = "Early Stop!"
				display(msg)
				estop = True
				break
		if estop:
			break
	end_time = time.time()
	msg = "Optimizing time: %f seconds" %(end_time - start_time)
	display(msg)

	with open(os.path.join(options.folder, "record.csv"), "w") as f:
		f.write(str(best_epoch) + '\n')
		f.write(','.join(map(str,lr_change_list)) + '\n')
		f.write(','.join(map(str,dev_acc_record)) + '\n')
		f.write(','.join(map(str,tst_acc_record)) + '\n')

	if best_p is not None:
		zipp(best_p, model.tparams)

	model.use_noise.set_value(0.)
	
	msg = "=" * 80
	display(msg)
	msg = "Final Result"
	display(msg)
	msg = "=" * 80
	display(msg)

	tra_acc = evaluate(f_pred, os.path.join(options.folder, "train_result"), tra_data_iterator_beta, options)
	tra_data_iterator_beta.reset()
	dev_acc = evaluate(f_pred, os.path.join(options.folder, "dev_result"), dev_data_iterator, options)
	dev_data_iterator.reset()
	tst_acc = evaluate(f_pred, os.path.join(options.folder, "test_result"), tst_data_iterator, options)
	tst_data_iterator.reset()
	msg = "Train accuracy: %f" %tra_acc
	display(msg)
	msg = "Valid accuracy: %f" %dev_acc
	display(msg)
	msg = "Test accuracy: %f" %tst_acc
	display(msg)
	msg = "best epoch: %d" %best_epoch
	display(msg)

	if best_p is None:
		best_p = unzip(model.tparams)
	
	numpy.savez(os.path.join(options.folder, options.saveto), **best_p)
	pkl.dump(options, open('%s.pkl' %os.path.join(options.folder, options.saveto), 'wb'))
	msg = "Finished"
	display(msg)

def main(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument("--folder", help = "the dir of model", default = "workshop")
	parser.add_argument("--file_dic", help = "the file of vocabulary", default = "../data/test_dic.pkl")
	parser.add_argument("--file_train", help = "the file of training data", default = "../data/train.pkl")
	parser.add_argument("--file_dev", help = "the file of valid data", default = "../data/valid.pkl")
	parser.add_argument("--file_test", help = "the file of testing data", default = "../data/test.pkl")
	parser.add_argument("--file_emb", help = "the file of embedding", default = "")
	parser.add_argument("--file_log", help = "the log file", default = "train.log")
	parser.add_argument("--reload_model", help = "the pretrained model", default = "")
	parser.add_argument("--saveto", help = "the file to save the parameter", default = "model")
	
	parser.add_argument("--size_vocab", help = "the size of vocabulary", default = 10000, type = int)
	parser.add_argument("--dim_emb", help = "the dimension of the word embedding", default = 300, type = int)
	parser.add_argument("--dim_proj", help = "the dimension of the LSTM hidden", default = 300, type = int)
	parser.add_argument("--num_class", help = "the number of labels", default = 3, type = int)
	parser.add_argument("--seed", help = "random seed", default = 1234, type = int)

	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adam")
	parser.add_argument("--lr", help = "learning rate", default = 0.0004, type = float)
	parser.add_argument("--dropout_rate", help = "dropout rate", default = 0.5, type = float)
	parser.add_argument("--decay_c", help = "decay rate", default = 0.0, type = float)
	parser.add_argument("--clip_c", help = "grad clip", default = 10.0, type = float)
	parser.add_argument("--nepochs", help = "the max epoch", default = 5000, type = int)
	parser.add_argument("--batch_size", help = "batch size", default = 32, type = int)
	parser.add_argument("--dispFreq", help = "the frequence of display", default = 100, type = int)
	parser.add_argument("--devFreq", help = "the frequence of evaluation", default = -1, type = int)
	parser.add_argument("--wait_N", help = "use to early stop", default = 1, type = int)
	parser.add_argument("--patience", help = "use to early stop", default = 7, type = int)
	parser.add_argument("--maxlen", help = "max length of sentence", default = None, type = int)

	options = parser.parse_args(argv)
	train(options)

if "__main__" == __name__:
	main(sys.argv[1:])