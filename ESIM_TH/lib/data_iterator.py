import numpy as np
import theano

import cPickle as pkl

class DataIterator():
	def __init__(self, data, sort = True, num_fragment = 100):
		self.data = data
		self.total = len(data[0])
		self.sort = sort
		self.num_fragment = num_fragment
		self.size = self.total / num_fragment
		self.idx_list = np.arange(self.total)
		self.cursor = 0
		self.epoch = 0

	def shuffle(self):
		if self.sort:
			lens = [max(len(x), len(y)) for x, y in zip(self.data[0], self.data[1])]
			idx2lens = {}
			for k, v in enumerate(lens):
				idx2lens[k] = v
			idx_list = np.array([k for k, v in sorted(idx2lens.iteritems(), key = lambda x: x[1])])
			
			fragments = []
			for i in range(self.num_fragment - 1):
				fragment = np.array(idx_list[i * self.size: (i + 1) * self.size])
				np.random.shuffle(fragment)
				fragments.append(fragment)
			fragment = np.array(idx_list[(self.num_fragment - 1) * self.size:])
			np.random.shuffle(fragment)
			fragments.append(fragment)
			
			self.idx_list = np.concatenate(fragments)
		else:
			self.idx_list = np.arange(self.total)
			np.random.shuffle(self.idx_list)

	def next_batch(self, batch_size, max_len = None, max_epoch = 1, is_train = False, shuffle = False):
		if self.cursor >= self.total:
			if is_train:
				self.epoch += 1
				if self.epoch >= max_epoch:
					return None, None, None, None, None
				if shuffle:
					self.shuffle()
				batch_index = self.idx_list[: batch_size]
				self.cursor = batch_size
			else:
				batch_index = []
		else:
			if self.cursor == 0:
				if shuffle:
					self.shuffle()
				batch_index = self.idx_list[: batch_size]
				self.cursor = batch_size
			else:
				batch_index = self.idx_list[self.cursor: self.cursor + batch_size]
				self.cursor += batch_size
		
		if len(batch_index) < 1:
			return None, None, None, None, None
		else:
			x = [self.data[0][idx] for idx in batch_index]
			y = [self.data[1][idx] for idx in batch_index]
			z = [self.data[2][idx] for idx in batch_index]
			
			x_, x_mask_ = self.prepare_data(x, max_len)
			y_, y_mask_ = self.prepare_data(y, max_len)
			z_ = np.array(z).astype("int32")

			return x_, x_mask_, y_, y_mask_, z_

	def prepare_data(self, seqs, maxlen = None):
		lengths = [len(s) for s in seqs]

		if maxlen:
			new_seqs = []
			new_lengths = []
			for l, s in zip(lengths, seqs):
				if l < maxlen:
					new_seqs.append(s)
					new_lengths.append(l)
				else:
					new_seqs.append(s[: maxlen])
					new_lengths.append(maxlen)
			seqs = new_seqs
			lengths = new_lengths
			
		if len(lengths) < 1:
			return None, None

		max_len = max(lengths)
		n_samples = len(seqs)
		
		x = np.zeros((max_len, n_samples)).astype("int32")
		x_mask = np.zeros((max_len, n_samples)).astype(theano.config.floatX)
		
		for idx, s in enumerate(seqs):
			x[:lengths[idx], idx] = s
			x_mask[:lengths[idx], idx] = 1.
				
		return x, x_mask

	def reset(self):
		self.cursor = 0
		self.epoch = 0
