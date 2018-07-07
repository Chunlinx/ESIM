import sys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

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
	msg = "=" * 30 + "model:" + "=" * 30
	display(msg)

	model = Model(word2idx,
				  options.file_emb,
				  options.size_vocab,
				  options.dim_emb,
				  options.dim_proj,
				  options.num_class,
				  options.dropout_rate).cuda()

	if options.reload_model != "":
		model.load_state_dict(torch.load(os.path.join(options.folder, "checkpoints", options.reload_model)))

	parameters = filter(lambda param: param.requires_grad, model.parameters())
	optimizer = optimizer_wrapper(options.optimizer, options.lr, parameters)

	msg = "\n{}".format(model)
	display(msg)
	
	#################################################################################
	checkpoint_dir = os.path.join(options.folder, "checkpoints")
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)
	best_path = os.path.join(checkpoint_dir, options.saveto)

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
		x, x_len, y, y_len, z = tra_data_iterator.next_batch(options.batch_size, max_len = options.maxlen, max_epoch = options.nepochs, is_train = True, shuffle = True)

		if x is None:
			break

		disp_start = time.time()

		model.train()

		n_updates += 1

		optimizer.zero_grad()
		logit = model(x, x_len, y, y_len)
		loss = F.cross_entropy(logit, z)
		loss.backward()

		if options.clip_c != 0:
			total_norm = torch.nn.utils.clip_grad_norm_(parameters, options.clip_c)

		optimizer.step()

		disp_end = time.time()

		if np.isnan(loss.cpu().item()) or np.isinf(loss.cpu().item()):
			msg = "NaN detected"
			display(msg)
			sys.exit(0)

		if np.mod(n_updates, options.dispFreq) == 0:
			msg = "Epoch: %d, Update: %d, Loss: %f, Time: %.2f sec" %(tra_data_iterator.epoch, n_updates, loss.cpu().item(), disp_end - disp_start)
			display(msg)

		if np.mod(n_updates, options.devFreq) == 0:
			msg = "=" * 30 + "Evaluating" + "=" * 30
			display(msg)
			model.eval()
			dev_acc = evaluate(model, os.path.join(options.folder, "current_dev_result"), dev_data_iterator, options)
			dev_data_iterator.reset()
			tst_acc = evaluate(model, os.path.join(options.folder, "current_test_result"), tst_data_iterator, options)
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
				best_epoch = tra_data_iterator.epoch
				wait_counter = 0

				msg = "Saving model..."
				display(msg)
				torch.save(model.state_dict(), best_path)
				msg = "Model checkpoint has been saved to {}".format(best_path)
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
				optimizer = optimizer_wrapper(options.optimizer, options.lr, parameters)
				lr_change_list.append(tra_data_iterator.epoch)
				msg = "lrate change to: %f" %(options.lr)
				display(msg)
				model.load_state_dict(torch.load(best_path))

			if bad_counter > options.patience:
				msg = "Early Stop!"
				display(msg)
				estop = True

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

	if os.path.exists(best_path):
		model.load_state_dict(torch.load(best_path))

	msg = "=" * 80
	display(msg)
	msg = "Final Result"
	display(msg)
	msg = "=" * 80
	display(msg)

	tra_acc = evaluate(model, os.path.join(options.folder, "train_result"), tra_data_iterator_beta, options)
	tra_data_iterator_beta.reset()
	dev_acc = evaluate(model, os.path.join(options.folder, "dev_result"), dev_data_iterator, options)
	dev_data_iterator.reset()
	tst_acc = evaluate(model, os.path.join(options.folder, "test_result"), tst_data_iterator, options)
	tst_data_iterator.reset()
	msg = "Train accuracy: %f" %tra_acc
	display(msg)
	msg = "Valid accuracy: %f" %dev_acc
	display(msg)
	msg = "Test accuracy: %f" %tst_acc
	display(msg)
	msg = "best epoch: %d" %best_epoch
	display(msg)

	if not os.path.exists(best_path):
		torch.save(model.state_dict(), best_path)
	pkl.dump(options, open("{}.pkl".format(os.path.join(options.folder, options.saveto)), "wb"))
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

	parser.add_argument("--optimizer", help = "optimization algorithm", default = "adam")
	parser.add_argument("--lr", help = "learning rate", default = 0.0004, type = float)
	parser.add_argument("--dropout_rate", help = "dropout rate", default = 0.5, type = float)
	parser.add_argument("--l2_reg_lambda", help = "decay rate", default = 0.0, type = float)
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