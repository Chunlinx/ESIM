import os
import sys
import time

import numpy as np
import tensorflow as tf

import logging

from lib import *
from model import *

tf.flags.DEFINE_string("folder", "workshop", "workshop folder")
tf.flags.DEFINE_string("file_dic", "../data/test_dic.pkl", "the file of vocabulary")
tf.flags.DEFINE_string("file_train", "../data/train.pkl", "the file of trainging data")
tf.flags.DEFINE_string("file_dev", "../data/dev.pkl", "the file of valid data")
tf.flags.DEFINE_string("file_test", "../data/test.pkl", "the file of test data")
tf.flags.DEFINE_string("file_emb", "", "the file of pretrained embedding")
tf.flags.DEFINE_string("file_log", "train.log", "the log file")
tf.flags.DEFINE_string("reload_model", "", "the pretrained model")
tf.flags.DEFINE_string("saveto", "model", "the file to save the parameter")

tf.flags.DEFINE_integer("size_vocab", 10000, "the size of the vocabulary")
tf.flags.DEFINE_integer("dim_emb", 300, "the dimension of the word embedding")
tf.flags.DEFINE_integer("dim_proj", 300, "the dimension of the MLP layers")
tf.flags.DEFINE_integer("num_class", 3, "the number of labels")

tf.flags.DEFINE_string("optimizer", "adam", "optimization algorithm")
tf.flags.DEFINE_float("lr", 0.0004, "learning rate")
tf.flags.DEFINE_float("dropout_rate", 0.5, "dropout rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "decay rate")
tf.flags.DEFINE_float("clip_c", 10, "grad clip")
tf.flags.DEFINE_integer("nepochs", 5000, "the max epoch")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_integer("dispFreq", 100, "the frequence of display")
tf.flags.DEFINE_integer("devFreq", -1, "the frequence of evaluation")
tf.flags.DEFINE_integer("patience", 7, "use to early stop")
tf.flags.DEFINE_integer("wait_N", 1, "use to early stop")
tf.flags.DEFINE_integer("maxlen", None, "max length of sentence")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def main(argv):
	if not os.path.exists(FLAGS.folder):
		os.mkdir(FLAGS.folder)

	logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter("%(asctime)s: %(name)s: %(levelname)s: %(message)s")
	hdlr = logging.FileHandler(os.path.join(FLAGS.folder, FLAGS.file_log), mode = "w")
	hdlr.setFormatter(formatter)
	logger.addHandler(hdlr)

	logger.info("python %s" %(" ".join(sys.argv)))

	#################################################################################
	start_time = time.time()

	msg = "Loading dicts from %s..." %(FLAGS.file_dic)
	display(msg)
	word2idx, label2idx = dicfold(FLAGS.file_dic)

	msg = "Loading data from %s..." %(FLAGS.file_train)
	display(msg)
	train_x, train_y, train_z = datafold(FLAGS.file_train)

	msg = "Loading data from %s..." %(FLAGS.file_dev)
	display(msg)
	dev_x, dev_y, dev_z = datafold(FLAGS.file_dev)

	msg = "Loading data from %s..." %(FLAGS.file_test)
	display(msg)
	test_x, test_y, test_z = datafold(FLAGS.file_test)

	end_time = time.time()

	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg)

	FLAGS.size_vocab = len(word2idx)
	FLAGS.num_class = len(label2idx)

	if FLAGS.devFreq == -1:
		FLAGS.devFreq = (len(train_x) + FLAGS.batch_size - 1) / FLAGS.batch_size

	msg = "#inst in train: %d" %(len(train_x))
	display(msg)
	msg = "#inst in dev %d" %(len(dev_x))
	display(msg)
	msg = "#inst in test %d" %(len(test_x))
	display(msg)
	msg = "#word vocab: %d" %(FLAGS.size_vocab)
	display(msg)
	msg = "#label: %d" %(FLAGS.num_class)
	display(msg)

	msg = "=" * 30 + "Hyperparameter:" + "=" * 30
	display(msg)
	for attr, value in sorted(FLAGS.__flags.items(),reverse=True):
		msg = "{}={}".format(attr.upper(), value)
		display(msg)

	#################################################################################
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement,
									  log_device_placement = FLAGS.log_device_placement)
		sess = tf.Session(config = session_conf)
		with sess.as_default():
			model = Model(word2idx,
						  FLAGS.file_emb,
						  FLAGS.size_vocab,
						  FLAGS.dim_emb,
						  FLAGS.dim_proj,
						  FLAGS.num_class,
						  FLAGS.l2_reg_lambda,
						  FLAGS.optimizer,
						  FLAGS.clip_c)
			model()

			tvs = tf.trainable_variables()
			msg = "=" * 30 + "Variable:" + "=" * 30
			display(msg)
			for v in tvs:
				msg = "name: {}, shape: {}".format(v.name, v.shape)
				display(msg)

			#################################################################################
			checkpoint_dir = os.path.join(FLAGS.folder, "checkpoints")
			if not os.path.exists(checkpoint_dir):
				os.mkdir(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables())

			#################################################################################
			msg = "=" * 30 + "Optimizing:" + "=" * 30
			display(msg)

			tra_data_iterator = DataIterator([train_x, train_y, train_z])
			tra_data_iterator_beta = DataIterator([train_x, train_y, train_z])
			dev_data_iterator = DataIterator([dev_x, dev_y, dev_z])
			tst_data_iterator = DataIterator([test_x, test_y, test_z])

			if FLAGS.reload_model:
				saver.restore(sess, os.path.join(checkpoint_dir, FLAGS.reload_model))
			else:
				sess.run(tf.global_variables_initializer())

			wait_counter = 0
			bad_counter = 0	
			estop = False

			dev_acc_record = []
			tst_acc_record = []
			lr_change_list = []

			best_acc = 0.0
			best_epoch = 0
			best_path = ""

			start_time = time.time()

			while True:
				x_, x_len_, y_, y_len_, z_ = tra_data_iterator.next_batch(FLAGS.batch_size, max_len = FLAGS.maxlen, max_epoch = FLAGS.nepochs, is_train = True, shuffle = True)

				if x_ is None:
					break

				disp_start = time.time()

				step, loss = model.train_step(sess, x_, x_len_, y_, y_len_, z_, FLAGS.dropout_rate, FLAGS.lr)

				disp_end = time.time()

				if np.isnan(loss) or np.isinf(loss):
					msg = "NaN detected"
					display(msg)
					sys.exit(0)

				if np.mod(step, FLAGS.dispFreq) == 0:
					msg = "Epoch: %d, Update: %d, Loss: %f, Time: %.2f sec" %(tra_data_iterator.epoch, step, loss, disp_end - disp_start)
					display(msg)

				if np.mod(step, FLAGS.devFreq) == 0:
					msg = "=" * 30 + "Evaluating" + "=" * 30
					display(msg)
					dev_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, "current_dev_result"), dev_data_iterator, FLAGS)
					dev_data_iterator.reset()
					tst_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, "current_test_result"), tst_data_iterator, FLAGS)
					tst_data_iterator.reset()

					msg = "dev accuracy: %f" %dev_acc
					display(msg)
					msg = "test accuracy: %f" %tst_acc
					display(msg)
					msg = "lrate: %f" %FLAGS.lr
					display(msg)

					dev_acc_record.append(dev_acc)
					tst_acc_record.append(tst_acc)

					if dev_acc > best_acc:
						best_acc = dev_acc
						best_epoch = tra_data_iterator.epoch
						wait_counter = 0

						msg = "Saving model..."
						display(msg)
						best_path = saver.save(sess, os.path.join(checkpoint_dir, FLAGS.saveto))
						msg = "Model checkpoint has been saved to {}".format(best_path)
						display(msg)
					else:
						wait_counter += 1

					if wait_counter >= FLAGS.wait_N:
						msg = "wait_counter max, need to half the lr"
						display(msg)
						bad_counter += 1
						wait_counter = 0
						msg = "bad_counter: %d" %bad_counter
						display(msg)
						FLAGS.lr *= 0.5
						lr_change_list.append(tra_data_iterator.epoch)
						msg = "lrate change to: %f" %(FLAGS.lr)
						display(msg)
						saver.restore(sess, best_path)

					if bad_counter > FLAGS.patience:
						msg = "Early Stop!"
						display(msg)
						estop = True

				if estop:
					break

			end_time = time.time()
			msg = "Optimizing time: %f seconds" %(end_time - start_time)
			display(msg)

			with open(os.path.join(FLAGS.folder, "record.csv"), "w") as f:
				f.write(str(best_epoch) + '\n')
				f.write(','.join(map(str,lr_change_list)) + '\n')
				f.write(','.join(map(str,dev_acc_record)) + '\n')
				f.write(','.join(map(str,tst_acc_record)) + '\n')

			if best_path:
				saver.restore(sess, best_path)

			msg = "=" * 80
			display(msg)
			msg = "Final Result"
			display(msg)
			msg = "=" * 80
			display(msg)

			tra_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, "train_result"), tra_data_iterator_beta, FLAGS)
			tra_data_iterator_beta.reset()
			dev_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, "dev_result"), dev_data_iterator, FLAGS)
			dev_data_iterator.reset()
			tst_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, "test_result"), tst_data_iterator, FLAGS)
			tst_data_iterator.reset()
			msg = "Train accuracy: %f" %tra_acc
			display(msg)
			msg = "Valid accuracy: %f" %dev_acc
			display(msg)
			msg = "Test accuracy: %f" %tst_acc
			display(msg)
			msg = "best epoch: %d" %best_epoch
			display(msg)

			if best_path == "":
				saver.save(sess, os.path.join(checkpoint_dir, FLAGS.saveto))
			pkl.dump(FLAGS, open("{}.pkl".format(os.path.join(FLAGS.folder, FLAGS.saveto)), "wb"))
			msg = "Finished"
			display(msg)
			
if "__main__" == __name__:
	tf.app.run()