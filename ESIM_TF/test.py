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
tf.flags.DEFINE_string("file_test", "../data/test.pkl", "the file of testing data")
tf.flags.DEFINE_string("file_pred", "pred.txt", "the file of pred data")
tf.flags.DEFINE_string("file_log", "test.log", "the log file")
tf.flags.DEFINE_string("file_emb", "", "the file of pretrained embedding")
tf.flags.DEFINE_string("reload_model", "model", "the pretrained model")

tf.flags.DEFINE_integer("size_vocab", 10000, "the size of the vocabulary")
tf.flags.DEFINE_integer("dim_emb", 300, "the dimension of the word embedding")
tf.flags.DEFINE_integer("dim_proj", 300, "the dimension of the MLP layers")
tf.flags.DEFINE_integer("num_class", 3, "the number of labels")

tf.flags.DEFINE_string("optimizer", "adam", "optimization algorithm")
tf.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "decay rate")
tf.flags.DEFINE_float("clip_c", 10, "grad clip")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

def main(argv):
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

	msg = "Loading data from %s..." %(FLAGS.file_test)
	display(msg)
	test_x, test_y, test_z = datafold(FLAGS.file_test)

	end_time = time.time()

	msg = "Loading data time: %f seconds" %(end_time - start_time)
	display(msg)

	FLAGS.size_vocab = len(word2idx)
	FLAGS.num_class = len(label2idx)

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
				msg = "Exception: The target folder does not exist."
				display(msg)
				sys.exit(0)
			saver = tf.train.Saver(tf.global_variables())
			#saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, "{}.meta".format(FLAGS.reload_model)))
			saver.restore(sess, os.path.join(checkpoint_dir, FLAGS.reload_model))

			#################################################################################
			msg = "=" * 30 + "Evaluating:" + "=" * 30
			display(msg)

			tst_data_iterator = DataIterator([test_x, test_y, test_z])

			start_time = time.time()
			tst_acc = evaluate(sess, model.test_step, os.path.join(FLAGS.folder, FLAGS.file_pred), tst_data_iterator, FLAGS)
			end_time = time.time()
			
			msg = "Test accuracy: %f" %tst_acc
			display(msg)

			msg = "Evaluating time: %f seconds" %(end_time - start_time)
			display(msg)

			msg = "Finished"
			display(msg)

if "__main__" == __name__:
	tf.app.run()