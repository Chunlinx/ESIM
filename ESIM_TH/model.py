from collections import OrderedDict
import numpy
import theano
import theano.tensor as tensor
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import sys
sys.path.append(".")
from lib import *

optimizer = {"adam": adam, "adadelta": adadelta}

def _p(pp, name):
	return '%s_%s' % (pp, name)

def zipp(params, tparams):
	for kk, vv in params.iteritems():
		tparams[kk].set_value(vv)

def unzip(zipped):
	new_params = OrderedDict()
	for kk, vv in zipped.iteritems():
		new_params[kk] = vv.get_value()
	return new_params

class Model(object):
	def __init__(self,
				 word2idx,
				 file_emb,
				 size_vocab,
				 dim_emb,
				 dim_proj,
				 num_class,
				 seed,
				 dropout_rate,
				 folder,
				 reload_model,
				 decay_c,
				 optimizer,
				 clip_c):
		self.word2idx = word2idx
		self.file_emb = file_emb
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb
		self.dim_proj = dim_proj
		self.num_class = num_class
		self.seed = seed
		self.dropout_rate = dropout_rate
		self.folder = folder
		self.reload_model = reload_model
		self.decay_c = decay_c
		self.optimizer = optimizer
		self.clip_c = clip_c

		self._init_params()

	def _init_params(self):
		self.params = OrderedDict()
	
		self.emb_layer = EMB_layer(dim_in = self.size_vocab, dim_out = self.dim_emb, word2idx = self.word2idx, file_emb = self.file_emb)
		self.emb_layer.init_params(self.params)
		
		self.encoder_lstm_fw_layer = LSTM_layer(prefix = "encoder_lstm_fw_layer", dim_in = self.dim_emb, dim_out = self.dim_proj)
		self.encoder_lstm_fw_layer.init_params(self.params)

		self.encoder_lstm_bw_layer = LSTM_layer(prefix = "encoder_lstm_bw_layer", dim_in = self.dim_emb, dim_out = self.dim_proj)
		self.encoder_lstm_bw_layer.init_params(self.params)

		self.fusion_mlp_layer = MLP_layer(prefix = "fusion_mlp_layer", dim_in = self.dim_proj * 2 * 4, dim_out = self.dim_proj)
		self.fusion_mlp_layer.init_params(self.params)

		self.fusion_lstm_fw_layer = LSTM_layer(prefix = "fusion_lstm_fw_layer", dim_in = self.dim_proj, dim_out = self.dim_proj)
		self.fusion_lstm_fw_layer.init_params(self.params)

		self.fusion_lstm_bw_layer = LSTM_layer(prefix = "fusion_lstm_bw_layer", dim_in = self.dim_proj, dim_out = self.dim_proj)
		self.fusion_lstm_bw_layer.init_params(self.params)

		self.dense_mlp_layer = MLP_layer(prefix = "dense", dim_in = self.dim_proj * 2 * 4, dim_out = self.dim_proj)
		self.dense_mlp_layer.init_params(self.params)

		self.class_mlp_layer = MLP_layer(prefix = "predicator", dim_in = self.dim_proj, dim_out = self.num_class)
		self.class_mlp_layer.init_params(self.params)

	def load_params(self):
		pp = numpy.load(os.path.join(self.folder, self.reload_model))
		for kk, vv in self.params.iteritems():
			if kk not in pp:
				raise Warning("%s is not in the archive" % kk)
			self.params[kk] = pp[kk]

	def init_tparams(self):
		tparams = OrderedDict()
		for kk, pp in self.params.iteritems():
			tparams[kk] = theano.shared(self.params[kk], name=kk)
		return tparams

	def build_model(self):
		trng = RandomStreams(self.seed)

		# Used for dropout.
		self.use_noise = theano.shared(numpy_floatX(0.))

		if self.reload_model:
			self.load_params()
	
		self.tparams = self.init_tparams()

		self.lr = tensor.scalar(dtype=config.floatX)
		self.x, self.mask_x, emb_x, self.y, self.mask_y, emb_y, self.z = self.emb_layer.build(self.tparams)
	
		emb_x = dropout_layer(emb_x, self.use_noise, trng, self.dropout_rate)
		emb_y = dropout_layer(emb_y, self.use_noise, trng, self.dropout_rate)

		proj_x_fw = self.encoder_lstm_fw_layer.build(self.tparams, emb_x, self.mask_x)
		proj_x_bw = reverse(self.encoder_lstm_bw_layer.build(self.tparams, reverse(emb_x), reverse(self.mask_x)))

		proj_x = tensor.concatenate([proj_x_fw, proj_x_bw], axis = -1) * self.mask_x[:, :, None]

		proj_y_fw = self.encoder_lstm_fw_layer.build(self.tparams, emb_y, self.mask_y)
		proj_y_bw = reverse(self.encoder_lstm_bw_layer.build(self.tparams, reverse(emb_y), reverse(self.mask_y)))
		
		proj_y = tensor.concatenate([proj_y_fw, proj_y_bw], axis = -1) * self.mask_y[:, :, None]

		weight = tensor.batched_dot(proj_x.dimshuffle(1, 0, 2), proj_y.dimshuffle(1, 2, 0)).dimshuffle(1, 2, 0)
		weight_x = tensor.exp(weight - weight.max(axis = 0, keepdims = True))
		weight_y = tensor.exp(weight - weight.max(axis = 1, keepdims = True))

		weight_x = weight_x * self.mask_x[:, None, :]
		weight_y = weight_y * self.mask_y[None, :, :]

		alpha = weight_x / weight_x.sum(axis = 0, keepdims = True)
		beta = weight_y / weight_y.sum(axis = 1, keepdims = True)

		proj_y_att = (proj_x.dimshuffle(0, 'x', 1, 2) * alpha.dimshuffle(0, 1, 2, 'x')).sum(axis = 0)
		proj_x_att = (proj_y.dimshuffle('x', 0, 1, 2) * beta.dimshuffle(0, 1, 2, 'x')).sum(axis = 1)

		proj_x_cat = tensor.concatenate([proj_x, proj_x_att, proj_x - proj_x_att, proj_x * proj_x_att], axis = -1)
		proj_y_cat = tensor.concatenate([proj_y, proj_y_att, proj_y - proj_y_att, proj_y * proj_y_att], axis = -1)

		fusion_mlp_x = ReLU(self.fusion_mlp_layer.build(self.tparams, proj_x_cat))
		fusion_mlp_y = ReLU(self.fusion_mlp_layer.build(self.tparams, proj_y_cat))
		
		fusion_mlp_x = dropout_layer(fusion_mlp_x, self.use_noise, trng, self.dropout_rate)
		fusion_mlp_y = dropout_layer(fusion_mlp_y, self.use_noise, trng, self.dropout_rate)

		fusion_lstm_fw_x = self.fusion_lstm_fw_layer.build(self.tparams, fusion_mlp_x, self.mask_x)
		fusion_lstm_bw_x = reverse(self.fusion_lstm_bw_layer.build(self.tparams, reverse(fusion_mlp_x), reverse(self.mask_x)))

		fusion_lstm_x = tensor.concatenate([fusion_lstm_fw_x, fusion_lstm_bw_x], axis = -1)

		fusion_lstm_fw_y = self.fusion_lstm_fw_layer.build(self.tparams, fusion_mlp_y, self.mask_y)
		fusion_lstm_bw_y = reverse(self.fusion_lstm_bw_layer.build(self.tparams, reverse(fusion_mlp_y), reverse(self.mask_y)))

		fusion_lstm_y = tensor.concatenate([fusion_lstm_fw_y, fusion_lstm_bw_y], axis = -1)

		logit_x_mean = (fusion_lstm_x * self.mask_x[:, :, None]).sum(axis = 0) / self.mask_x.sum(axis = 0)[:, None]
		logit_x_max = (fusion_lstm_x * self.mask_x[:, :, None]).max(axis = 0)

		logit_y_mean = (fusion_lstm_y * self.mask_y[:, :, None]).sum(axis = 0) / self.mask_y.sum(axis = 0)[:, None]
		logit_y_max = (fusion_lstm_y * self.mask_y[:, :, None]).max(axis = 0)

		logit = tensor.concatenate([logit_x_mean, logit_x_max, logit_y_mean, logit_y_max], axis = -1)

		logit = dropout_layer(logit, self.use_noise, trng, self.dropout_rate)
		
		logit = tensor.tanh(self.dense_mlp_layer.build(self.tparams, logit))
		
		logit = dropout_layer(logit, self.use_noise, trng, self.dropout_rate)
		
		self.pred_prob = tensor.nnet.nnet.softmax(self.class_mlp_layer.build(self.tparams, logit))
		self.pred = self.pred_prob.argmax(axis = -1)

		off = 1e-8
		if self.pred_prob.dtype == 'float16':
			off = 1e-6
		
		self.log_cost = -tensor.log(self.pred_prob[tensor.arange(self.x.shape[1]), self.z] + off).mean()
		self.cost = self.log_cost
		if self.decay_c > 0.:
			decay_c = theano.shared(numpy.float32(self.decay_c), name='decay_c')
			weight_decay = 0.
			for kk, vv in self.tparams.iteritems():
				weight_decay += (vv ** 2).sum()
			weight_decay *= decay_c
			self.cost += weight_decay
		
		self.grads = tensor.grad(self.cost, wrt = self.tparams.values())
		g2 = 0.
		for g in self.grads:
			g2 += (g ** 2).sum()
		self.grad_norm = tensor.sqrt(g2)
		
		if self.clip_c > 0.:
			new_grads = []
			for g in self.grads:
				new_grads.append(tensor.switch(g2 > self.clip_c ** 2, g * self.clip_c / tensor.sqrt(g2), g))
			self.grads = new_grads

	def build_optimizer(self):
		msg = "compiling f_cost, f_update..."
		display(msg)
		f_cost, f_update = optimizer[self.optimizer](self.lr, self.tparams, (self.x, self.y, self.mask_x, self.mask_y, self.z), self.cost, self.grads)
		msg = "compiling f_log_cost..."
		display(msg)
		f_log_cost = theano.function(inputs = [self.x, self.y, self.mask_x, self.mask_y, self.z], outputs = self.log_cost, name = "f_log_cost")
		msg = "compiling f_grad_norm..."
		display(msg)
		f_grad_norm = theano.function(inputs = [self.x, self.y, self.mask_x, self.mask_y, self.z], outputs = self.grad_norm, name = "f_grad_norm")
		msg = "compiling f_pred_prob..."
		display(msg)
		f_pred_prob = theano.function(inputs = [self.x, self.y, self.mask_x, self.mask_y], outputs = self.pred_prob, name = "f_pred_prob")
		msg = "compiling f_pred..."
		display(msg)
		f_pred = theano.function(inputs = [self.x, self.y, self.mask_x, self.mask_y], outputs = self.pred, name = "f_pred")
		return f_cost, f_update, f_log_cost, f_grad_norm, f_pred_prob, f_pred
