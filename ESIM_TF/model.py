import sys
sys.path.append(".")
from lib import *
import re

import numpy as np
import tensorflow as tf

class Model(object):
	def __init__(self,
				 word2idx,
				 file_emb,
				 size_vocab,
				 dim_emb,
				 dim_proj,
				 num_class,
				 l2_reg_lambda,
				 optimizer,
				 clip_c):
		self.word2idx = word2idx
		self.file_emb = file_emb
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb
		self.dim_proj = dim_proj
		self.num_class = num_class
		self.l2_reg_lambda = l2_reg_lambda
		self.optimizer = optimizer
		self.clip_c = clip_c

	def _input(self, name = "Wemb"):
		init_embedding = 0.01 * np.random.randn(self.size_vocab, self.dim_emb).astype(np.float32)
		
		if self.file_emb != "":
			init_embedding = self.load_emb(init_embedding)

		Wemb = tf.Variable(initial_value = init_embedding, name = name)

		return Wemb

	def _encoder(self, inp, inp_len, name = "encoding"):
		fw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim_proj)
		bw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim_proj)

		(forward_output_inp, backward_output_inp), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
																					   bw_cell,
																					   inp,
																					   dtype = tf.float32,
																					   sequence_length = inp_len,
																					   scope = name)
		proj_inp = tf.concat(values = [forward_output_inp, backward_output_inp], axis = -1)
		
		return proj_inp

	def _matching(self, inp_x, inp_y, inp_x_len, inp_y_len, name = "matching"):
		weight = tf.matmul(inp_x, inp_y, transpose_b = True)
		weight_x = tf.exp(weight - tf.reduce_max(weight, axis = 1, keep_dims = True))
		weight_y = tf.exp(weight - tf.reduce_max(weight, axis = 2, keep_dims = True))

		weight_x = weight_x * self.len2mask(inp_x_len)[:, :, None]
		weight_y = weight_y * self.len2mask(inp_y_len)[:, None, :]

		alpha = weight_x / tf.reduce_sum(weight_x, axis = 1, keep_dims = True)
		beta = weight_y / tf.reduce_sum(weight_y, axis = 2, keep_dims = True)

		inp_y_att = tf.reduce_sum(tf.expand_dims(inp_x, axis = 2) * tf.expand_dims(alpha, axis = -1), axis = 1)
		inp_x_att = tf.reduce_sum(tf.expand_dims(inp_y, axis = 1) * tf.expand_dims(beta, axis = -1), axis = 2)

		inp_x_cat = tf.concat([inp_x, inp_x_att, inp_x - inp_x_att, inp_x * inp_x_att], axis = -1)
		inp_y_cat = tf.concat([inp_y, inp_y_att, inp_y - inp_y_att, inp_y * inp_y_att], axis = -1)

		fusion_mlp_x = tf.layers.dense(inputs = inp_x_cat,
									   units = self.dim_proj,
									   activation = tf.nn.relu,
									   kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
									   name = "attention_fusion_mlp")
		fusion_mlp_y = tf.layers.dense(inputs = inp_y_cat,
									   units = self.dim_proj,
									   activation = tf.nn.relu,
									   kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01),
									   name = "attention_fusion_mlp",
									   reuse = True)

		fusion_mlp_x = tf.nn.dropout(fusion_mlp_x, self.dropout_rate)
		fusion_mlp_y = tf.nn.dropout(fusion_mlp_y, self.dropout_rate)

		fw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim_proj)
		bw_cell = tf.contrib.rnn.BasicLSTMCell(self.dim_proj)

		(forward_output_inp, backward_output_inp), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
																					   bw_cell,
																					   fusion_mlp_x,
																					   dtype = tf.float32,
																					   sequence_length = inp_x_len,
																					   scope = "attention_fusion_lstm")
		fusion_lstm_x = tf.concat([forward_output_inp, backward_output_inp], axis = -1)

		(forward_output_inp, backward_output_inp), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
																					   bw_cell,
																					   fusion_mlp_y,
																					   dtype = tf.float32,
																					   sequence_length = inp_y_len,
																					   scope = "attention_fusion_lstm")
		fusion_lstm_y = tf.concat([forward_output_inp, backward_output_inp], axis = -1)

		return fusion_lstm_x, fusion_lstm_y

	def _classifier(self, inp_x, inp_y, inp_x_len, inp_y_len, l2_loss = None):
		inp_x_len = tf.cast(inp_x_len, tf.float32)
		inp_y_len = tf.cast(inp_y_len, tf.float32)

		inp_x_max = tf.reduce_max(inp_x * self.len2mask(inp_x_len)[:, :, None], axis = 1)
		inp_x_mean = tf.reduce_sum(inp_x * self.len2mask(inp_x_len)[:, :, None], axis = 1) / inp_x_len[:, None]

		inp_y_max = tf.reduce_max(inp_y * self.len2mask(inp_y_len)[:, :, None], axis = 1)
		inp_y_mean = tf.reduce_sum(inp_y * self.len2mask(inp_y_len)[:, :, None], axis = 1) / inp_y_len[:, None]

		inp_x_y = tf.concat([inp_x_max, inp_x_mean, inp_y_max, inp_y_mean], axis = -1)
		inp_x_y = tf.nn.dropout(inp_x_y, self.dropout_rate)

		dense_W = tf.get_variable(name = "dense_W",
								  shape = [self.dim_proj * 2 * 4, self.dim_proj],
								  initializer = tf.truncated_normal_initializer(stddev = 0.01))
		dense_b = tf.get_variable(name = "dense_b",
								  shape = [self.dim_proj],
								  initializer = tf.zeros_initializer())

		dense = tf.nn.tanh(tf.nn.xw_plus_b(inp_x_y, dense_W, dense_b, name = "dense"))
		dense = tf.nn.dropout(dense, self.dropout_rate)

		class_W = tf.get_variable(name = "class_W",
								  shape = [self.dim_proj, self.num_class],
								  initializer = tf.truncated_normal_initializer(stddev = 0.01))
		class_b = tf.get_variable(name = "class_b",
								  shape = [self.num_class],
								  initializer = tf.zeros_initializer())

		pred_prob = tf.nn.xw_plus_b(dense, class_W, class_b, name = "pred_prob")
		pred = tf.argmax(pred_prob, axis = -1, name = "pred")
		
		if l2_loss != None:
			l2_loss += tf.nn.l2_loss(dense_W)
			l2_loss += tf.nn.l2_loss(dense_b)
			l2_loss += tf.nn.l2_loss(class_W)
			l2_loss += tf.nn.l2_loss(class_b)

		return pred_prob, pred, l2_loss

	def _loss(self, logits, z, l2_loss = None):
		loss = tf.losses.sparse_softmax_cross_entropy(labels = z,
													  logits = logits,
													  reduction = tf.losses.Reduction.MEAN)
		
		if l2_loss != None:
			return loss + self.l2_reg_lambda * l2_loss
		else:
			return loss

	def _train_op(self):
		if self.optimizer == "adam":
			optimizer = tf.train.AdamOptimizer(self.lr)

		global_step = tf.Variable(0, name = "global_step", trainable=False)

		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.clip_c)
		
		train_op = optimizer.apply_gradients(grads_and_vars = zip(grads, tvars), global_step = global_step)
		
		return train_op, global_step

	def __call__(self):
		with tf.variable_scope("placeholder"):
			#batch_size X steps
			self.x = tf.placeholder(tf.int32, shape = [None, None], name = "x")
			self.x_len = tf.placeholder(tf.int32, shape = [None], name = "x_len")
			self.y = tf.placeholder(tf.int32, shape = [None, None], name = "y")
			self.y_len = tf.placeholder(tf.int32, shape = [None], name = "y_len")
			self.z = tf.placeholder(tf.int32, shape = [None], name = "z")
			self.dropout_rate = tf.placeholder(tf.float32, name = "dropout_rate")
			self.lr = tf.placeholder(tf.float32, name = "lr")

		l2_loss = None
		if self.l2_reg_lambda != 0:
			l2_loss = tf.constant(0.0)

		with tf.variable_scope("embedding"):
			self.Wemb = self._input(name = "Wemb")
			
			self.x_emb = tf.nn.embedding_lookup(self.Wemb, self.x)
			self.y_emb = tf.nn.embedding_lookup(self.Wemb, self.y)

			self.x_emb = tf.nn.dropout(self.x_emb, self.dropout_rate)
			self.y_emb = tf.nn.dropout(self.y_emb, self.dropout_rate)

		with tf.variable_scope("encoding"):
			self.proj_x = self._encoder(self.x_emb, self.x_len, name = "encoding_layer")
			tf.get_variable_scope().reuse_variables()
			self.proj_y = self._encoder(self.y_emb, self.y_len, name = "encoding_layer")

			#self.proj_x = tf.nn.dropout(self.proj_x, self.dropout_rate)
			#self.proj_y = tf.nn.dropout(self.proj_y, self.dropout_rate)

		with tf.variable_scope("matching"):
			self.fusion_x, self.fusion_y = self._matching(self.proj_x, self.proj_y, self.x_len, self.y_len, name = "matching_layer")

		with tf.variable_scope("classifier"):
			self.pred_prob, self.pred, l2_loss = self._classifier(self.fusion_x, self.fusion_y, self.x_len, self.y_len, l2_loss)

		with tf.variable_scope("loss"):
			self.loss = self._loss(self.pred_prob, self.z, l2_loss)

		with tf.variable_scope("train_op"):
			self.train_op, self.global_step = self._train_op()

	def train_step(self, sess, x_batch, x_len_batch, y_batch, y_len_batch, z_batch, dropout_rate, lr):
		feed_dict = {
			self.x: x_batch,
			self.x_len: x_len_batch,
			self.y: y_batch,
			self.y_len: y_len_batch,
			self.z: z_batch,
			self.dropout_rate: dropout_rate,
			self.lr: lr
		}

		_, step, loss = sess.run([self.train_op, self.global_step, self.loss], feed_dict)

		return step, loss

	def test_step(self, sess, x_batch, x_len_batch, y_batch, y_len_batch, z_batch):
		feed_dict = {
			self.x: x_batch,
			self.x_len: x_len_batch,
			self.y: y_batch,
			self.y_len: y_len_batch,
			self.z: z_batch,
			self.dropout_rate: float(1.0)
		}

		preds = sess.run(self.pred, feed_dict)

		return preds

	def load_emb(self, init_embedding):
		msg = 'load emb from ' + self.file_emb
		display(msg)

		filein = open(self.file_emb, 'r')
		emb_dict = {}
		emb_p = re.compile(r" |\t")
		for line in filein:
			array = emb_p.split(line.strip())
			vector = [float(array[i]) for i in range(1, len(array))]
			word = array[0]
			emb_dict[word] = vector
		filein.close()
		msg = "find %d words in %s" %(len(emb_dict), self.file_emb)
		display(msg)
		
		count = 0
		for k, v in self.word2idx.items():
			if k in emb_dict:
				init_embedding[v] = emb_dict[k]
				count += 1
		msg = "Summary: %d words in the vocabulary and %d of them appear in the %s" %(len(self.word2idx), count, self.file_emb)
		display(msg)
		
		return init_embedding

	def len2mask(self, lengths):
		flag = tf.sequence_mask(lengths)
		mask = tf.where(condition = flag, x = tf.ones_like(flag, tf.float32), y = tf.zeros_like(flag, tf.float32))
		return mask