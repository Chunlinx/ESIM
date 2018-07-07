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

def str2list(num_str):
	num_list = [int(num.strip()) for num in num_str.split(" ")]
	return num_list

def getA(answers, results):
	total = 0.
	right = 0.
	for i in range(len(answers)):
		if answers[i] == results[i]:
			right += 1
		total += 1
	return right / total

def evaluate(model, pred_file, data_iterator, options):
	labels = []
	preds = []

	scores = []
	while True:
		x, x_len, y, y_len, z = data_iterator.next_batch(options.batch_size)

		if x is None:
			break
		else:
			labels.append(z.cpu().numpy())
			scores.append(model(x, x_len, y, y_len).detach().cpu().numpy())

	labels = numpy.concatenate(labels, axis = 0)
	scores = numpy.concatenate(scores, axis = 0)
	preds = scores.argmax(axis = 1)
	
	data_acc = getA(labels, preds)

	with open(pred_file, "w") as f:
		for pred in preds:
			f.write("%d\n" %pred)

	return data_acc