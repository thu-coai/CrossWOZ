import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import Constants
from .LSTM import BeamSearchNode
from queue import PriorityQueue
import operator
import json
import math
import copy

USE_CUDA = True

class SCLSTM(nn.Module):
	def __init__(self, vocab_size, d_word_vec, d_model, act_size, n_layer=1, dropout=0.5):
		super(SCLSTM, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = d_model
		self.output_size = vocab_size
		self.act_size = act_size
		self.n_layer = n_layer
		self.dropout = dropout

		print('Using sclstm as decoder with module list!')
		assert act_size != None
		# NOTE: using modulelist instead of python list
		self.w2h, self.h2h = nn.ModuleList(), nn.ModuleList()
		self.w2h_r, self.h2h_r = nn.ModuleList(), nn.ModuleList()
		self.dc = nn.ModuleList()
		for i in range(n_layer):
			if i == 0:
				self.w2h.append(nn.Linear(d_word_vec, d_model*4).cuda() )
				self.w2h_r.append(nn.Linear(d_word_vec, act_size).cuda() )
			else:
				self.w2h.append(nn.Linear(d_word_vec + i*d_model, d_model*4).cuda() )
				self.w2h_r.append(nn.Linear(d_word_vec + i*d_model, act_size).cuda() )

			self.h2h.append(nn.Linear(d_model, d_model*4).cuda() )
			self.h2h_r.append(nn.Linear(d_model, act_size).cuda() )
			self.dc.append(nn.Linear(act_size, d_model, bias=False).cuda() )

		self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_word_vec, padding_idx=Constants.PAD)
		self.out = nn.Linear(d_model*n_layer, self.output_size)

	def _step(self, last_emb, last_hidden, last_cell, last_dt, layer_idx):
		'''
		* Do feedforward for one step in one layer in sclstm *
		Args:
			seq: (batch_size, hidden_size)
			last_hidden: (batch_size, hidden_size)
			last_cell: (batch_size, hidden_size)
		Return:
			cell, hidden, dt at this time step, all: (batch_size, hidden_size)
		'''
		# get all gates
		w2h = self.w2h[layer_idx](last_emb) # (batch_size, hidden_size*4)
		w2h = torch.split(w2h, self.hidden_size, dim=1) # (batch_size, hidden_size) * 4
		h2h = self.h2h[layer_idx](last_hidden[layer_idx])
		h2h = torch.split(h2h, self.hidden_size, dim=1)

		gate_i = torch.sigmoid(w2h[0] + h2h[0]) # (batch_size, hidden_size)
		gate_f = torch.sigmoid(w2h[1] + h2h[1])
		gate_o = torch.sigmoid(w2h[2] + h2h[2])

		# updata dt
		alpha = 1. / self.n_layer
		# NOTE: avoid inplace operation which will cause backprop error on graph
		_gate_r = 0
		for i in range(self.n_layer):
			_gate_r += alpha * self.h2h_r[i](last_hidden[i])
		gate_r = torch.sigmoid(self.w2h_r[layer_idx](last_emb) + _gate_r)
		
		dt = gate_r * last_dt

		cell_hat = torch.tanh(w2h[3] + h2h[3])
		cell = gate_f * last_cell + gate_i * cell_hat + torch.tanh( self.dc[layer_idx](dt) )
		hidden = gate_o * torch.tanh(cell)

		return hidden, cell, dt


	def rnn_step(self, last_inp, last_hidden, last_cell, last_dt, gen=False):
		'''
		run a step over all layers in sclstm
		'''
		cur_hidden, cur_cell, cur_dt = [], [], []
		output_hidden = []
		last_emb = self.embed(last_inp)
		for i in range(self.n_layer):
			# prepare input_t
			if i == 0:
				input_t = last_emb
			else:
				pre_hidden = torch.cat(output_hidden, dim=1)
				input_t = torch.cat((last_emb, pre_hidden), dim=1)
			
			_hidden, _cell, _dt = self._step(input_t, last_hidden, last_cell[i], last_dt[i], i)		
			cur_hidden.append(_hidden)
			cur_cell.append(_cell)
			cur_dt.append(_dt)
			if gen:
				output_hidden.append(_hidden.clone() )
			else:
				output_hidden.append(nn.functional.dropout(_hidden.clone(), p=self.dropout, training=True) )

		last_hidden, last_cell, last_dt = cur_hidden, cur_cell, cur_dt
		if not gen:
			for i in range(self.n_layer):
				last_hidden[i] = nn.functional.dropout(last_hidden[i], p=self.dropout, training=True)
		output = self.out(torch.cat(last_hidden, dim=1))
		return output, last_hidden, last_cell, last_dt

	
	def forward(self, tgt_seq, enc_output, act_vecs=None, gen=False, sample_size=1, **kwargs):
		'''
		Args:
			input_seq: (batch_size, max_len)
			hidden: (batch_size, hidden_size) if exist
			feat: (batch_size, feat_size) if exist
		Return:
			output_prob: (batch_size, max_len, output_size)
		'''
		batch_size = tgt_seq.size(0)
		init_hidden = enc_output[:, 0, :]
		#input_emb = self.embed(tgt_seq)
		last_inp = tgt_seq[:, 0].view(-1) 
		max_len = 55 if gen else tgt_seq.size(1)
		output_prob = Variable(torch.zeros(batch_size, max_len, self.output_size)).cuda()

		# container for last cell, hidden for each layer
		# NOTE: for container, using just list instead of creating a torch variable causing inplace operation runtime error
		last_hidden, last_cell, last_dt = [], [], []
		for i in range(self.n_layer):
			last_hidden.append( init_hidden.clone() )
			last_cell.append( init_hidden.clone() ) # create a new variable with same content, but new history
			last_dt.append( act_vecs.clone() )

		decoded_words = []
		#vocab_t = self.get_onehot('SOS_token', batch_size=batch_size)
		for t in range(max_len):
			output, last_hidden, last_cell, last_dt = self.rnn_step(last_inp, last_hidden, last_cell, last_dt, gen=gen)
			output_prob[:, t, :] = output
			if gen:
				last_inp = self.logits2words(output, sample_size)
				decoded_words.append(last_inp)
			else:
				if t + 1 < max_len:
					last_inp = tgt_seq[:, t + 1].view(-1) 				
			
		return output_prob


	def logits2words(self, output, sample_size):
		'''
		* Decode words from logits output at a time step AND put decoded words in final results *
		* take argmax if sample size == 1
		'''
		batch_size = output.size(0)
		if sample_size == 1: # take argmax directly w/o sampling
			topv, topi = F.softmax(output, dim=1).data.topk(1) # both (batch_size, 1)
		else: # sample over word distribution
			topv, topi = [], []
			word_dis = F.softmax(output, dim=1) # (batch_size, output_size)
			# sample from part of the output distribution for word variations
			n_candidate = 3
			word_dis_sort, idx_of_idx = torch.sort(word_dis, dim=1, descending=True)
			word_dis_sort = word_dis_sort[:, :n_candidate]
			idx_of_idx = idx_of_idx[:, :n_candidate]
			sample_idx = torch.multinomial(word_dis_sort, 1) # (batch_size,)
			for b in range(batch_size):
				i = int(sample_idx[b])
				idx = int(idx_of_idx[b][i])
				prob = float(word_dis[b][idx])
				topi.append(idx)
				topv.append(prob)
				
			topv = torch.FloatTensor(topv).view(batch_size, 1)
			topi = torch.LongTensor(topi).view(batch_size, 1)
			
		decoded_words_t = np.zeros((batch_size, ))
		for b in range(batch_size):
			decoded_words_t[b] = topi[b][0]
		
		decoded_words_t = Variable(torch.from_numpy(decoded_words_t.astype(np.long)))

		if USE_CUDA:
			decoded_words_t = decoded_words_t.cuda()

		return decoded_words_t 
	
	def translate_batch(self, act_vecs, src_enc, gen=True, n_bm=10, max_token_seq_len=50):
		'''
		Args:
			input_var: (batch_size, max_len, emb_size)
			hidden: (batch_size, hidden_size) if exist
			feat: (batch_size, feat_size) if exist
		Return:
			decoded_words: (batch_size, beam_size)
			
		'''
		if n_bm > 1:  # wenqiang style - sequicity
			decoded_sentences = []
			batch_size = src_enc.size(0)
			decoder_hiddens = src_enc[:, 0, :]
			for idx in range(batch_size):
				decoder_hidden = decoder_hiddens[idx, :].unsqueeze(0).unsqueeze(1)
				decoder_cell = decoder_hiddens[idx, :].unsqueeze(0).unsqueeze(1)
				decoder_act = act_vecs
				
				# Beam start
				self.topk = 1
				endnodes = []  # stored end nodes
				number_required = min((self.topk + 1), self.topk - len(endnodes))
				decoder_input = torch.LongTensor([Constants.SOS]).to(src_enc.device)
			
				# starting node hidden vector, prevNode, wordid, logp, leng,
				node = BeamSearchNode((decoder_hidden, decoder_cell, decoder_act), None, decoder_input, 0, 1)
				nodes = PriorityQueue()  # start the queue
				nodes.put((-node.eval(None, None, None, None), node))
			
				# start beam search
				qsize = 1
				while True:
					# give up when decoding takes too long
					if qsize > 2000: break
			
					# fetch the best node
					score, n = nodes.get()
					decoder_input = n.wordid
					decoder_hidden, decoder_cell, decoder_act = n.h
			
					if n.wordid.item() == Constants.EOS and n.prevNode != None:  # its not empty
						endnodes.append((score, n))
						# if reach maximum # of sentences required
						if len(endnodes) >= number_required:
							break
						else:
							continue
					
					# decode for one step using decoder
					decoder_output, decoder_hidden, decoder_cell, decoder_act = \
						    self.rnn_step(decoder_input, decoder_hidden, decoder_cell, decoder_act, None)
			
					log_prob, indexes = torch.topk(decoder_output, n_bm)
					nextnodes = []
					for new_k in range(n_bm):
						decoded_t = indexes[0][new_k].view(-1)
						log_p = log_prob[0][new_k].item()
			
						node = BeamSearchNode((decoder_hidden, decoder_cell, decoder_act), n, decoded_t, n.logp + log_p, n.leng + 1)
						score = -node.eval(None, None, None, None)
						nextnodes.append((score, node))
			
					# put them into queue
					for i in range(len(nextnodes)):
						score, nn = nextnodes[i]
						nodes.put((score, nn))
			
					# increase qsize
					qsize += len(nextnodes)
			
				# choose nbest paths, back trace them
				if len(endnodes) == 0:
					endnodes = [nodes.get() for n in range(self.topk)]
			
				utterances = []
				for score, n in sorted(endnodes, key=operator.itemgetter(0)):
					utterance = []
					utterance.append(n.wordid.item())
					# back trace
					while n.prevNode != None:
						n = n.prevNode
						utterance.append(n.wordid.item())
			
					utterance = utterance[::-1]
					utterances.append(utterance)
			
				decoded_words = utterances[0][1:]
				#decoded_sentence = [self.output_index2word(str(ind.item())) for ind in decoded_words]
				#print(decoded_sentence)
				decoded_sentences.append(decoded_words)
			
			return decoded_sentences

