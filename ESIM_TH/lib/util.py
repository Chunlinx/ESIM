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

def evaluate(f_pred, pred_file, data_iterator, options):
	labels = []
	preds = []

	while True:
		x, x_mask, y, y_mask, z = data_iterator.next_batch(options.batch_size)

		if x is None:
			break
		else:
			labels.append(z)
			preds.append(f_pred(x, y, x_mask, y_mask))

	labels = numpy.concatenate(labels)
	preds = numpy.concatenate(preds)
	
	data_acc = getA(labels, preds)

	with open(pred_file, "w") as f:
		for pred in preds:
			f.write("%d\n" %pred)

	return data_acc