import sys
sys.path.append(".")
from lib import *
import re

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLstm(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(BiLstm, self).__init__()

		self.dim_in = dim_in
		self.dim_out = dim_out

		self._init_params()

	def _init_params(self):
		self.bilstm = nn.LSTM(input_size = self.dim_in,
							  hidden_size = self.dim_out,
							  bidirectional = True)

	def forward(self, inp, inp_len):
		sorted_inp_len, sorted_idx = torch.sort(inp_len, dim = 0, descending=True)
		sorted_inp = torch.index_select(inp, dim = 1, index = sorted_idx)

		pack_inp = torch.nn.utils.rnn.pack_padded_sequence(sorted_inp, sorted_inp_len)
		proj_inp, _ = self.bilstm(pack_inp)
		proj_inp = torch.nn.utils.rnn.pad_packed_sequence(proj_inp)

		unsorted_idx = torch.zeros(sorted_idx.size()).long().cuda().scatter_(0, sorted_idx, torch.arange(inp.size()[1]).long().cuda())
		unsorted_proj_inp = torch.index_select(proj_inp[0], dim = 1, index = unsorted_idx)

		return unsorted_proj_inp

class MLP(nn.Module):
	def __init__(self, dim_in, dim_out):
		super(MLP, self).__init__()

		self.dim_in = dim_in
		self.dim_out = dim_out

		self._init_params()

	def _init_params(self):
		self.mlp = nn.Linear(in_features = self.dim_in,
							 out_features = self.dim_out)

	def forward(self, inp):
		proj_inp = self.mlp(inp)
		return proj_inp

class Matching(nn.Module):
	def __init__(self, dim_proj, dropout_rate):
		super(Matching, self).__init__()

		self.dim_proj = dim_proj
		self.dropout_rate = dropout_rate

		self.dropout = nn.Dropout(self.dropout_rate)

		self._init_params()

	def _init_params(self):
		self.fusion_mlp = nn.Linear(in_features = self.dim_proj * 2 * 4,
									out_features = self.dim_proj)

		self.fusion_lstm = BiLstm(dim_in = self.dim_proj,
								  dim_out = self.dim_proj)

	def forward(self, inp_x, inp_x_len, inp_x_mask, inp_y, inp_y_len, inp_y_mask):
		weight = torch.matmul(inp_x.permute(1, 0, 2), inp_y.permute(1, 2, 0)).permute(1, 2, 0)
		weight_x = torch.exp(weight - weight.max(dim = 0, keepdim = True)[0])
		weight_y = torch.exp(weight - weight.max(dim = 1, keepdim = True)[0])

		weight_x = weight_x * inp_x_mask[:, None, :]
		weight_y = weight_y * inp_y_mask[None, :, :]

		alpha = weight_x / weight_x.sum(dim = 0, keepdim = True)
		beta = weight_y / weight_y.sum(dim = 1, keepdim = True)

		inp_y_att = (torch.unsqueeze(inp_x, 1) * torch.unsqueeze(alpha, -1)).sum(dim = 0)
		inp_x_att = (torch.unsqueeze(inp_y, 0) * torch.unsqueeze(beta, -1)).sum(dim = 1)

		inp_x_cat = torch.cat((inp_x, inp_x_att, inp_x - inp_x_att, inp_x * inp_x_att), dim = -1)
		inp_y_cat = torch.cat((inp_y, inp_y_att, inp_y - inp_y_att, inp_y * inp_y_att), dim = -1)

		fusion_mlp_x = F.relu(self.fusion_mlp(inp_x_cat))
		fusion_mlp_y = F.relu(self.fusion_mlp(inp_y_cat))

		fusion_mlp_x = self.dropout(fusion_mlp_x)
		fusion_mlp_y = self.dropout(fusion_mlp_y)

		fusion_lstm_x = self.fusion_lstm(fusion_mlp_x, inp_x_len)
		fusion_lstm_y = self.fusion_lstm(fusion_mlp_y, inp_y_len)

		return fusion_lstm_x, fusion_lstm_y

class Model(nn.Module):
	def __init__(self,
				 word2idx,
				 file_emb,
				 size_vocab,
				 dim_emb,
				 dim_proj,
				 num_class,
				 dropout_rate):
		super(Model, self).__init__()

		self.word2idx = word2idx
		self.file_emb = file_emb
		self.size_vocab = size_vocab
		self.dim_emb = dim_emb
		self.dim_proj = dim_proj
		self.num_class = num_class
		self.dropout_rate = dropout_rate

		self.dropout = nn.Dropout(self.dropout_rate)

		self._init_params()

	def _init_Wemb(self):
		if self.file_emb == "":
			Wemb = nn.Embedding(self.size_vocab, self.dim_emb)
		else:
			init_embedding = 0.01 * np.random.randn(self.size_vocab, self.dim_emb).astype(np.float32)
			init_embedding = self.load_emb(init_embedding)
			Wemb = nn.Embedding.from_pretrained(torch.Tensor(init_embedding), freeze = False)

		return Wemb

	def _init_params(self):
		self.Wemb = self._init_Wemb()
		self.encoder = BiLstm(self.dim_emb, self.dim_proj)
		self.matching = Matching(self.dim_proj, self.dropout_rate)
		self.dense = MLP(self.dim_proj * 2 * 4, self.dim_proj)
		self.classifier = MLP(self.dim_proj, self.num_class)

	def forward(self,x, x_len, y, y_len):
		x_mask = self.len2mask(x_len.float())
		y_mask = self.len2mask(y_len.float())

		x_emb = self.Wemb(x)
		y_emb = self.Wemb(y)

		x_emb = self.dropout(x_emb)
		y_emb = self.dropout(y_emb)

		x_proj = self.encoder(x_emb, x_len)
		y_proj = self.encoder(y_emb, y_len)

		x_fusion, y_fusion = self.matching(x_proj, x_len, x_mask, y_proj, y_len, y_mask)

		logit_x_mean = torch.sum(x_fusion * x_mask[:, :, None], dim = 0) / torch.sum(x_mask, dim = 0)[:, None]
		logit_x_max = torch.max(x_fusion * x_mask[:, :, None], dim = 0)[0]

		logit_y_mean = torch.sum(y_fusion * y_mask[:, :, None], dim = 0) / torch.sum(y_mask, dim = 0)[:, None]
		logit_y_max = torch.max(y_fusion * y_mask[:, :, None], dim = 0)[0]

		logit = torch.cat((logit_x_mean, logit_x_max, logit_y_mean, logit_y_max), dim = -1)

		logit = self.dropout(logit)

		logit = F.tanh(self.dense(logit))

		logit = self.dropout(logit)

		logit = self.classifier(logit)

		return logit

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
		max_len = torch.max(lengths)
		idxes = torch.arange(0, max_len).unsqueeze(1).cuda()
		mask = (idxes < lengths.unsqueeze(0)).float()
		return mask

def optimizer_wrapper(optimizer, lr, parameters):
	if optimizer == "adam":
		opt = torch.optim.Adam(params = parameters, lr = lr)
	return opt