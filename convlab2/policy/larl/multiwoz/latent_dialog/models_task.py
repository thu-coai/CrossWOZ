import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from convlab2.policy.larl.multiwoz.latent_dialog.base_models import BaseModel
from convlab2.policy.larl.multiwoz.latent_dialog.corpora import SYS, EOS, PAD, BOS
from convlab2.policy.larl.multiwoz.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.encoders import RnnUttEncoder
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.decoders import DecoderRNN, GEN, TEACH_FORCE
from convlab2.policy.larl.multiwoz.latent_dialog.criterions import NLLEntropy, CatKLLoss, Entropy, NormKLLoss
from convlab2.policy.larl.multiwoz.latent_dialog import nn_lib
import numpy as np


class SysPerfectBD2Word(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Word, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.policy = nn.Sequential(nn.Linear(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                              config.dec_cell_size), nn.Tanh(), nn.Dropout(config.dropout))

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=self.utt_encoder.output_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            # h_dec_init_state = utt_summary.squeeze(1).unsqueeze(0)
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=self.nll(dec_outputs, labels),
                        latent_action=dec_init_state)
        else:
            return Pack(nll=self.nll(dec_outputs, labels))

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # pack attention context
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None

        # create decoder initial states
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)

        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=temp)
        return logprobs, outs


class SysPerfectBD2Cat(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                                   config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y.cuda()
            self.eye = self.eye.cuda()

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        if self.config.use_mi:
            total_loss += (loss.b_pr * self.beta)

        if self.config.use_diversity:
            total_loss += loss.diversity

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode==GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or (use_py is not None and use_py is True):
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               # (batch_size, response_size-1)
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               # (batch_size, max_ctx_len, ctx_cell_size)
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            # regularization qy to be uniform
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)  # b
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)

            result['pi_kl'] = pi_kl

            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)

        qy = F.softmax(logits_py / temp, dim=1)  # (batch_size, vocab_size, )
        log_qy = F.log_softmax(logits_py, dim=1)  # (batch_size, vocab_size, )
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)

        # pack attention context
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_y


class SysPerfectBD2Gauss(BaseModel):
    def __init__(self, corpus, config):
        super(SysPerfectBD2Gauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior

        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size,
                                         embedding_dim=config.embed_size,
                                         feat_size=0,
                                         goal_nhid=0,
                                         rnn_cell=config.utt_rnn_cell,
                                         utt_cell_size=config.utt_cell_size,
                                         num_layers=config.num_layers,
                                         input_dropout_p=config.dropout,
                                         output_dropout_p=config.dropout,
                                         bidirectional=config.bi_utt_cell,
                                         variable_lengths=False,
                                         use_attn=config.enc_use_attn,
                                         embedding=self.embedding)

        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size,
                                          config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size,
                                               config.y_size, is_lstm=False)

        self.decoder = DecoderRNN(input_dropout_p=config.dropout,
                                  rnn_cell=config.dec_rnn_cell,
                                  input_size=config.embed_size,
                                  hidden_size=config.dec_cell_size,
                                  num_layers=config.num_layers,
                                  output_dropout_p=config.dropout,
                                  bidirectional=False,
                                  vocab_size=self.vocab_size,
                                  use_attn=config.dec_use_attn,
                                  ctx_cell_size=config.dec_cell_size,
                                  attn_mode=config.dec_attn_mode,
                                  sys_id=self.bos_id,
                                  eos_id=self.eos_id,
                                  use_gpu=config.use_gpu,
                                  max_dec_len=config.max_dec_len,
                                  embedding=self.embedding)

        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl

        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)  # (batch_size, max_out_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # get decoder inputs
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        # create decoder initial states
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            # encode response and use posterior to find q(z|x, c)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))

            # use prior at inference time, otherwise use posterior
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size,
                                                               dec_inputs=dec_inputs,
                                                               dec_init_state=dec_init_state,  # tuple: (h, c)
                                                               attn_context=attn_context,
                                                               mode=mode,
                                                               gen_type=gen_type,
                                                               beam_size=self.config.beam_size)  # (batch_size, goal_nhid)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels

        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2*np.pi))
        logprob = constant - 0.5 * logvar - th.pow((mu-sample_z), 2) / (2.0*var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)  # (batch_size, max_ctx_len, max_utt_len)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))

        # create decoder initial states
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        # create decoder initial states
        p_mu, p_logvar = self.c2z(enc_last)

        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)

        # pack attention context
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None

        # decode
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        # decode
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size,
                                                 dec_init_state=dec_init_state,
                                                 attn_context=attn_context,
                                                 vocab=self.vocab,
                                                 max_words=max_words,
                                                 temp=0.1)
        return logprobs, outs, joint_logpz, sample_z

