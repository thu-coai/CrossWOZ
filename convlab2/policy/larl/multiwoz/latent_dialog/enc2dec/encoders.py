import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.base_modules import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p,
                                         rnn_cell=rnn_cell,
                                         input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         output_dropout_p=output_dropout_p,
                                         bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size*2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        # add goals
        if goals is not None:
            batch_size, max_ctx_len, ctx_nhid = input_var.size()
            goals = goals.view(goals.size(0), 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, 1).view(
                batch_size, max_ctx_len, -1)  # (batch_size, max_ctx_len, goal_nhid)
            input_var = th.cat([input_var, goals_rep], dim=2)

        embedded = self.input_dropout(input_var)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,
                                                         batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True)

        return output, hidden


class RnnUttEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, feat_size, goal_nhid, rnn_cell,
                 utt_cell_size, num_layers, input_dropout_p, output_dropout_p,
                 bidirectional, variable_lengths, use_attn, embedding=None):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding

        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p,
                              rnn_cell=rnn_cell,
                              input_size=embedding_dim+feat_size+goal_nhid,
                              hidden_size=utt_cell_size,
                              num_layers=num_layers,
                              output_dropout_p=output_dropout_p,
                              bidirectional=bidirectional,
                              variable_lengths=variable_lengths)

        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, goals=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        # get word embeddings
        # (batch_size*max_ctx_len, max_utt_len)
        flat_words = utterances.view(-1, max_utt_len)
        # (batch_size*max_ctx_len, max_utt_len, embedding_dim)
        word_embeddings = self.embedding(flat_words)
        flat_mask = th.sign(flat_words).float()
        # add features
        if feats is not None:
            flat_feats = feats.view(-1, 1)  # (batch_size*max_ctx_len, 1)
            # (batch_size*max_ctx_len, max_utt_len, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            # (batch_size*max_ctx_len, max_utt_len, embedding_dim+1)
            word_embeddings = th.cat([word_embeddings, flat_feats], dim=2)

        # add goals
        if goals is not None:
            goals = goals.view(goals.size(0), 1, 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, max_utt_len, 1).view(
                batch_size*max_ctx_len, max_utt_len, -1)  # (batch_size*max_ctx_len, max_utt_len, goal_nhid)
            word_embeddings = th.cat([word_embeddings, goals_rep], dim=2)

        # enc_outs: (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
        # enc_last: (num_layers*num_directions, batch_size*max_ctx_len, utt_cell_size)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)

        if self.use_attn:
            # (batch_size*max_ctx_len, max_utt_len, utt_cell_size)
            fc1 = th.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            # (batch_size*max_ctx_len, max_utt_len)
            # (batch_size*max_ctx_len, max_utt_len, 1)
            attn = F.softmax(attn, attn.dim()-1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True)+1e-10)).unsqueeze(2)
            # (batch_size*max_ctx_len, max_utt_len, num_directions*utt_cell_size)
            utt_embedded = attn * enc_outs
            # (batch_size*max_ctx_len, num_directions*utt_cell_size)
            utt_embedded = th.sum(utt_embedded, dim=1)
        else:
            # FIXME bug for multi-layer
            attn = None
            # (batch_size*max_ctx_lens, num_layers*num_directions, utt_cell_size)
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            # (batch_size*max_ctx_len*num_layers, num_directions*utt_cell_size)
            utt_embedded = utt_embedded.view(-1, self.output_size)

        utt_embedded = utt_embedded.view(
            batch_size, max_ctx_len, self.output_size)
        return utt_embedded, word_embeddings.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1), \
            enc_outs.contiguous().view(batch_size, max_ctx_len*max_utt_len, -1)


class MlpGoalEncoder(nn.Module):
    def __init__(self, goal_vocab_size, k, nembed, nhid, init_range):
        super(MlpGoalEncoder, self).__init__()

        # create separate embedding for counts and values
        self.cnt_enc = nn.Embedding(goal_vocab_size, nembed)
        self.val_enc = nn.Embedding(goal_vocab_size, nembed)

        self.encoder = nn.Sequential(
            nn.Tanh(),
            nn.Linear(k*nembed, nhid)
        )

        self.cnt_enc.weight.data.uniform_(-init_range, init_range)
        self.val_enc.weight.data.uniform_(-init_range, init_range)
        self._init_cont(self.encoder, init_range)

    def _init_cont(self, cont, init_range):
        """initializes a container uniformly."""
        for m in cont:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_range, init_range)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

    def forward(self, goal):
        # goal: (batch_size, goal_len)
        goal = goal.transpose(0, 1).contiguous()  # (goal_len, batch_size)
        idx = np.arange(goal.size(0) // 2)

        # extract counts and values
        cnt_idx = Variable(th.from_numpy(2 * idx + 0))
        val_idx = Variable(th.from_numpy(2 * idx + 1))

        if goal.is_cuda:
            cnt_idx = cnt_idx.type(th.cuda.LongTensor)
            val_idx = val_idx.type(th.cuda.LongTensor)
        else:
            cnt_idx = cnt_idx.type(th.LongTensor)
            val_idx = val_idx.type(th.LongTensor)

        cnt = goal.index_select(0, cnt_idx)  # (3, batch_size)
        val = goal.index_select(0, val_idx)  # (3, batch_size)

        # embed counts and values
        cnt_emb = self.cnt_enc(cnt)  # (3, batch_size, nembed)
        val_emb = self.val_enc(val)  # (3, batch_size, nembed)

        # element wise multiplication to get a hidden state
        h = th.mul(cnt_emb, val_emb)  # (3, batch_size, nembed)
        # run the hidden state through the MLP
        h = h.transpose(0, 1).contiguous().view(
            goal.size(1), -1)  # (batch_size, 3*nembed)
        goal_h = self.encoder(h)  # (batch_size, nhid)

        return goal_h


class TaskMlpGoalEncoder(nn.Module):
    def __init__(self, goal_vocab_sizes, nhid, init_range):
        super(TaskMlpGoalEncoder, self).__init__()

        self.encoder = nn.ModuleList()
        for v_size in goal_vocab_sizes:
            domain_encoder = nn.Sequential(
                nn.Linear(v_size, nhid),
                nn.Tanh()
            )
            self._init_cont(domain_encoder, init_range)
            self.encoder.append(domain_encoder)

    def _init_cont(self, cont, init_range):
        """initializes a container uniformly."""
        for m in cont:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_range, init_range)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

    def forward(self, goals_list):
        # goals_list: list of tensor, 7*(batch_size, goal_len), goal_len varies among differnet domains
        outs = [encoder.forward(goal) for goal, encoder in zip(
            goals_list, self.encoder)]  # 7*(batch_size, goal_nhid)
        outs = th.sum(th.stack(outs), dim=0)  # (batch_size, goal_nhid)
        return outs


class SelfAttn(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, keys, values, attn_mask=None):
        """
        :param attn_inputs: batch_size x time_len x hidden_size
        :param attn_mask: batch_size x time_len
        :return: summary state
        """
        alpha = F.softmax(self.query(keys), dim=1)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
            alpha = alpha / th.sum(alpha, dim=1, keepdim=True)

        summary = th.sum(values * alpha, dim=1)
        return summary
