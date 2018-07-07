import sys
import numpy
import cPickle as pkl

import logging

logger = logging.getLogger()

def display(msg):
	print(msg)
	logger.info(msg)

def datafold(textFile):
	x, y, z = pkl.load(open(textFile, "rb"))
	return x, y, z

def dicfold(textFile):
	word2idx, label2idx = pkl.load(open(textFile, "rb"))
	return word2idx, label2idx

def getA(answers, results):
	total = 0.
	right = 0.
	for i in range(len(answers)):
		if answers[i] == results[i]:
			right += 1
		total += 1
	return right / total

def evaluate(sess, test_step, pred_file, data_iterator, FLAGS):
	labels = []
	preds = []

	while True:
		x_, x_len_, y_, y_len_, z_ = data_iterator.next_batch(FLAGS.batch_size)

		if x_ is None:
			break
		else:
			labels.append(z_)
			preds.append(test_step(sess, x_, x_len_, y_, y_len_, z_))

	labels = numpy.concatenate(labels)
	preds = numpy.concatenate(preds)
	
	data_acc = getA(labels, preds)

	with open(pred_file, "w") as f:
		for pred in preds:
			f.write("%d\n" %pred)

	return data_acc