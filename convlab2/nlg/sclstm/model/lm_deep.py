import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

from convlab2.nlg.sclstm.model.layers.decoder_deep import DecoderDeep
from convlab2.nlg.sclstm.model.masked_cross_entropy import masked_cross_entropy


class LMDeep(nn.Module):
	def __init__(self, dec_type, input_size, output_size, hidden_size, d_size, n_layer=1, dropout=0.5, lr=0.001, use_cuda=False):
		super(LMDeep, self).__init__()
		self.dec_type = dec_type
		self.hidden_size = hidden_size
		print('Using deep version with {} layer'.format(n_layer))
		print('Using deep version with {} layer'.format(n_layer), file=sys.stderr)
		self.USE_CUDA = use_cuda
		self.dec = DecoderDeep(dec_type, input_size, output_size, hidden_size, d_size=d_size, n_layer=n_layer, dropout=dropout, use_cuda=use_cuda)
#		if self.dec_type != 'sclstm':
#			self.feat2hidden = nn.Linear(d_size, hidden_size)

		self.set_solver(lr)

	def	forward(self, input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1):
		batch_size = dataset.batch_size
		if self.dec_type == 'sclstm':
			init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
			if self.USE_CUDA:
				init_hidden = init_hidden.cuda()
			'''
			train/valid (gen=False, beam_search=False, beam_size=1)
	 		test w/o beam_search (gen=True, beam_search=False, beam_size=beam_size)
	 		test w/i beam_search (gen=True, beam_search=True, beam_size=beam_size)
			'''
			if beam_search:
				assert gen
				decoded_words = self.dec.beam_search(input_var, dataset, init_hidden=init_hidden, init_feat=feats_var, \
														gen=gen, beam_size=beam_size)
				return decoded_words # list (batch_size=1) of list (beam_size) with generated sentences

			# w/o beam_search
			sample_size = beam_size
			decoded_words = [ [] for _ in range(batch_size) ]
			for sample_idx in range(sample_size): # over generation
				self.output_prob, gens = self.dec(input_var, dataset, init_hidden=init_hidden, init_feat=feats_var, \
													gen=gen, sample_size=sample_size)
				for batch_idx in range(batch_size):
					decoded_words[batch_idx].append(gens[batch_idx])

			return decoded_words # list (batch_size) of list (sample_size) with generated sentences


		else: # TODO: vanilla lstm
			pass
#			last_hidden = self.feat2hidden(conds_batches)
#			self.output_prob, decoded_words = self.dec(input_seq, dataset, last_hidden=last_hidden, gen=gen, random_sample=self.random_sample)


	def generate(self, dataset, feats_var, beam_size=1):
		batch_size = dataset.batch_size
		init_hidden = Variable(torch.zeros(batch_size, self.hidden_size))
		if self.USE_CUDA:
			init_hidden = init_hidden.cuda()
		decoded_words = self.dec.beam_search(None, dataset, init_hidden=init_hidden, init_feat=feats_var, \
														gen=True, beam_size=beam_size)
		return decoded_words

	def set_solver(self, lr):
		if self.dec_type == 'sclstm':
			self.solver = torch.optim.Adam(self.dec.parameters(), lr=lr)
		else:
			self.solver = torch.optim.Adam([{'params': self.dec.parameters()}, {'params': self.feat2hidden.parameters()}], lr=lr)


	def get_loss(self, target_label, target_lengths):
		self.loss = masked_cross_entropy(
			self.output_prob.contiguous(), # -> batch x seq
			target_label.contiguous(), # -> batch x seq
			target_lengths)
		return self.loss


	def update(self, clip):
		# Back prop
		self.loss.backward()

		# Clip gradient norms
		_ = torch.nn.utils.clip_grad_norm(self.dec.parameters(), clip)

		# Update
		self.solver.step()

		# Zero grad
		self.solver.zero_grad()
