from __future__ import division, print_function, unicode_literals

import json
import math
import operator
import os
import random
from queue import PriorityQueue

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
from . import Constants

# Shawn beam search decoding
class BeamSearchNode(object):
    def __init__(self, h, prevNode, wordid, logp, leng):
        self.h = h
        self.prevNode = prevNode
        self.wordid = wordid
        self.logp = logp
        self.leng = leng

    def eval(self, repeatPenalty, tokenReward, scoreTable, alpha=1.0):
        reward = 0
        alpha = 1.0

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)


def init_gru(gru, gain=1):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i+gru.hidden_size],gain=gain)


def whatCellType(input_size, hidden_size, cell_type, dropout_rate):
    if cell_type == 'rnn':
        cell = nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'gru':
        cell = nn.GRU(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'lstm':
        cell = nn.LSTM(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell
    elif cell_type == 'bigru':
        cell = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'bilstm':
        cell = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)

        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1)  # [T,B,H] -> [B,T,H]
        attn_energies = self.score(H,encoder_outputs)  # compute attention score
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # normalize with softmax

    def score(self, hidden, encoder_outputs):
        cat = torch.cat([hidden, encoder_outputs], 2)
        energy = torch.tanh(self.attn(cat)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class SeqAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout_p=0.1, max_length=30):
        super(SeqAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.output_size = output_size
        self.n_layers = 1
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size + hidden_size, hidden_size, cell_type, dropout_rate=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, embedding_size)

        # attention
        self.method = 'concat'
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, input, hidden, encoder_outputs):
        if isinstance(hidden, tuple):
            h_t = hidden[0]
        else:
            h_t = hidden
        encoder_outputs = encoder_outputs.transpose(0, 1)
        embedded = self.embedding(input)  # .view(1, 1, -1)
        # embedded = F.dropout(embedded, self.dropout_p)

        # SCORE 3
        max_len = encoder_outputs.size(1)
        h_t = h_t.transpose(0, 1)  # [1,B,D] -> [B,1,D]
        h_t = h_t.repeat(1, max_len, 1)  # [B,1,D]  -> [B,T,D]
        energy = self.attn(torch.cat((h_t, encoder_outputs), 2))  # [B,T,2D] -> [B,T,D]
        energy = torch.tanh(energy)
        energy = energy.transpose(2, 1)  # [B,H,T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B,1,H]
        energy = torch.bmm(v, energy)  # [B,1,T]
        attn_weights = F.softmax(energy, dim=2)  # [B,1,T]

        # getting context
        context = torch.bmm(attn_weights, encoder_outputs)  # [B,1,H]

        # context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) #[B,1,H]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((embedded, context), 2)
        rnn_input = rnn_input.transpose(0, 1)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)

        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden  # , attn_weights



class DecoderCell(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        super(DecoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=Constants.PAD)
        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size, hidden_size, cell_type, dropout_rate=dropout)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1,B, D]
        embedded = F.dropout(embedded, self.dropout_rate)
        output = embedded
        #output = F.relu(embedded)
        output, hidden = self.rnn(output, hidden)

        logits = self.out(output.squeeze(0))
        #output = F.log_softmax(out, dim=1)

        return logits, hidden

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, d_word_vec, d_model):
        super(LSTMDecoder, self).__init__()
        self.cell = DecoderCell(d_word_vec, d_model, vocab_size, 'gru')
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_word_vec = d_word_vec
        self.teacher_ratio = 1.0
        
    def forward(self, tgt_seq, enc_output, act_vecs):
        """Given the user sentence, user belief state and database pointer,
        encode the sentence, decide what policy vector construct and
        feed it as the first hiddent state to the decoder."""
        target_length = tgt_seq.size(1)
        batch_size = tgt_seq.size(0)
        decoder_hidden = enc_output[:, 0, :].unsqueeze(0).contiguous()
        decoder_input = tgt_seq[:, 0].view(-1, 1)
        proba = torch.zeros(batch_size, target_length, self.vocab_size, device=tgt_seq.device)  # [B,T,V]        
        for t in range(target_length):
            decoder_output, decoder_hidden = self.cell(decoder_input, decoder_hidden, None)
            
            use_teacher_forcing = True if random.random() < self.teacher_ratio else False
            if use_teacher_forcing:
                if t + 1 < target_length:
                    decoder_input = tgt_seq[:, t + 1].view(-1, 1)  # [B,1] Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

            proba[:, t, :] = decoder_output

        return proba

    #def decode(self, target_tensor, decoder_hidden, encoder_outputs):
    def translate_batch(self, act_vecs, src_enc, n_bm, max_token_seq_len=30, return_all=False):    
        
        if n_bm > 1:  # wenqiang style - sequicity
            decoded_sentences = []
            batch_size = src_enc.size(0)
            decoder_hiddens = src_enc[:, 0, :]
            for idx in range(batch_size):
                decoder_hidden = decoder_hiddens[idx, :].unsqueeze(0).unsqueeze(1)
                #encoder_output = src_enc[idx, :, :]

                # Beam start
                self.topk = 1
                endnodes = []  # stored end nodes
                number_required = min((self.topk + 1), self.topk - len(endnodes))
                decoder_input = torch.LongTensor([[Constants.SOS]]).to(src_enc.device)

                # starting node hidden vector, prevNode, wordid, logp, leng,
                node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
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
                    decoder_hidden = n.h

                    if n.wordid.item() == Constants.EOS and n.prevNode != None:  # its not empty
                        endnodes.append((score, n))
                        # if reach maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue
                    
                    # decode for one step using decoder
                    decoder_output, decoder_hidden = self.cell(decoder_input, decoder_hidden, None)

                    log_prob, indexes = torch.topk(decoder_output, n_bm)
                    nextnodes = []

                    for new_k in range(n_bm):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        log_p = log_prob[0][new_k].item()

                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
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
