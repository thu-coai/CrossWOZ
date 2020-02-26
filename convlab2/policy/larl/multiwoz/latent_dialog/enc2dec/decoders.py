import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.base_modules import BaseRNN
from convlab2.policy.larl.multiwoz.latent_dialog.utils import cast_type, LONG, FLOAT
from convlab2.policy.larl.multiwoz.latent_dialog.corpora import DECODING_MASKED_TOKENS, EOS


TEACH_FORCE = 'teacher_forcing'
TEACH_GEN = 'teacher_gen'
GEN = 'gen'
GEN_VALID = 'gen_valid'


class Attention(nn.Module):
    def __init__(self, dec_cell_size, ctx_cell_size, attn_mode, project):
        super(Attention, self).__init__()
        self.dec_cell_size = dec_cell_size
        self.ctx_cell_size = ctx_cell_size
        self.attn_mode = attn_mode
        if project:
            self.linear_out = nn.Linear(
                dec_cell_size+ctx_cell_size, dec_cell_size)
        else:
            self.linear_out = None

        if attn_mode == 'general':
            self.dec_w = nn.Linear(dec_cell_size, ctx_cell_size)
        elif attn_mode == 'cat':
            self.dec_w = nn.Linear(dec_cell_size, dec_cell_size)
            self.attn_w = nn.Linear(ctx_cell_size, dec_cell_size)
            self.query_w = nn.Linear(dec_cell_size, 1)

    def forward(self, output, context):
        # output: (batch_size, output_seq_len, dec_cell_size)
        # context: (batch_size, max_ctx_len, ctx_cell_size)
        batch_size = output.size(0)
        max_ctx_len = context.size(1)

        if self.attn_mode == 'dot':
            # (batch_size, output_seq_len, max_ctx_len)
            attn = th.bmm(output, context.transpose(1, 2))
        elif self.attn_mode == 'general':
            # (batch_size, output_seq_len, ctx_cell_size)
            mapped_output = self.dec_w(output)
            # (batch_size, output_seq_len, max_ctx_len)
            attn = th.bmm(mapped_output, context.transpose(1, 2))
        elif self.attn_mode == 'cat':
            # (batch_size, output_seq_len, dec_cell_size)
            mapped_output = self.dec_w(output)
            # (batch_size, max_ctx_len, dec_cell_size)
            mapped_attn = self.attn_w(context)
            # (batch_size, output_seq_len, max_ctx_len, dec_cell_size)
            tiled_output = mapped_output.unsqueeze(
                2).repeat(1, 1, max_ctx_len, 1)
            # (batch_size, 1, max_ctx_len, dec_cell_size)
            tiled_attn = mapped_attn.unsqueeze(1)
            # (batch_size, output_seq_len, max_ctx_len, dec_cell_size)
            fc1 = F.tanh(tiled_output+tiled_attn)
            # (batch_size, otuput_seq_len, max_ctx_len)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError('Unknown attention mode')

        # TODO mask
        # if self.mask is not None:

        # (batch_size, output_seq_len, max_ctx_len)
        attn = F.softmax(attn.view(-1, max_ctx_len),
                         dim=1).view(batch_size, -1, max_ctx_len)
        # (batch_size, output_seq_len, ctx_cell_size)
        mix = th.bmm(attn, context)
        # (batch_size, output_seq_len, dec_cell_size+ctx_cell_size)
        combined = th.cat((mix, output), dim=2)
        if self.linear_out is None:
            return combined, attn
        else:
            output = F.tanh(
                self.linear_out(combined.view(-1, self.dec_cell_size+self.ctx_cell_size))).view(
                batch_size, -1, self.dec_cell_size)  # (batch_size, output_seq_len, dec_cell_size)
            return output, attn


class DecoderRNN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p,
                 bidirectional, vocab_size, use_attn, ctx_cell_size, attn_mode, sys_id, eos_id, use_gpu,
                 max_dec_len, embedding=None):

        super(DecoderRNN, self).__init__(input_dropout_p=input_dropout_p,
                                         rnn_cell=rnn_cell,
                                         input_size=input_size,
                                         hidden_size=hidden_size,
                                         num_layers=num_layers,
                                         output_dropout_p=output_dropout_p,
                                         bidirectional=bidirectional)

        # TODO embedding is None or not
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.embedding = embedding

        # share parameters between encoder and decoder
        # self.rnn = ctx_encoder.rnn
        # self.FC = nn.Linear(input_size, utt_encoder.output_size)

        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(dec_cell_size=hidden_size,
                                       ctx_cell_size=ctx_cell_size,
                                       attn_mode=attn_mode,
                                       project=True)

        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.project = nn.Linear(self.dec_cell_size, self.output_size)
        self.log_softmax = F.log_softmax

        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len

    def forward(self, batch_size, dec_inputs, dec_init_state, attn_context, mode, gen_type, beam_size, goal_hid=None):
        # dec_inputs: (batch_size, response_size-1)
        # attn_context: (batch_size, max_ctx_len, ctx_cell_size)
        # goal_hid: (batch_size, goal_nhid)

        ret_dict = dict()

        if self.use_attn:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if mode == GEN:
            dec_inputs = None

        if gen_type != 'beam':
            beam_size = 1

        if dec_inputs is not None:
            decoder_input = dec_inputs
        else:
            # prepare the BOS inputs
            with th.no_grad():
                bos_var = Variable(th.LongTensor([self.sys_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(
                batch_size*beam_size, 1)  # (batch_size, 1)

        if mode == GEN and gen_type == 'beam':
            # TODO if beam search, repeat the initial states of the RNN
            pass
        else:
            decoder_hidden_state = dec_init_state

        # list of logprob | max_dec_len*(batch_size, 1, vocab_size)
        prob_outputs = []
        symbol_outputs = []  # list of word ids | max_dec_len*(batch_size, 1)
        # back_pointers = []
        # lengths = blabla...

        def decode(step, cum_sum, step_output, step_attn):
            prob_outputs.append(step_output)
            step_output_slice = step_output.squeeze(
                1)  # (batch_size, vocab_size)
            if self.use_attn:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == 'greedy':
                _, symbols = step_output_slice.topk(1)  # (batch_size, 1)
            elif gen_type == 'sample':
                # TODO FIXME
                # symbols = self.gumbel_max(step_output_slice)
                pass
            elif gen_type == 'beam':
                # TODO
                pass
            else:
                raise ValueError('Unsupported decoding mode')

            symbol_outputs.append(symbols)

            return cum_sum, symbols

        if mode == TEACH_FORCE:
            prob_outputs, decoder_hidden_state, attn = self.forward_step(
                input_var=decoder_input, hidden_state=decoder_hidden_state, encoder_outputs=attn_context, goal_hid=goal_hid)
        else:
            # do free running here
            cum_sum = None
            for step in range(self.max_dec_len):
                # Input:
                #   decoder_input: (batch_size, 1)
                #   decoder_hidden_state: tuple: (h, c)
                #   attn_context: (batch_size, max_ctx_len, ctx_cell_size)
                #   goal_hid: (batch_size, goal_nhid)
                # Output:
                #   decoder_output: (batch_size, 1, vocab_size)
                #   decoder_hidden_state: tuple: (h, c)
                #   step_attn: (batch_size, 1, max_ctx_len)
                decoder_output, decoder_hidden_state, step_attn = self.forward_step(
                    decoder_input, decoder_hidden_state, attn_context, goal_hid=goal_hid)
                cum_sum, symbols = decode(
                    step, cum_sum, decoder_output, step_attn)
                decoder_input = symbols

            # (batch_size, max_dec_len, vocab_size)
            prob_outputs = th.cat(prob_outputs, dim=1)

            # back tracking to recover the 1-best in beam search
            # if gen_type == 'beam':

        ret_dict[DecoderRNN.KEY_SEQUENCE] = symbol_outputs

        # prob_outputs: (batch_size, max_dec_len, vocab_size)
        # decoder_hidden_state: tuple: (h, c)
        # ret_dict[DecoderRNN.KEY_ATTN_SCORE]: max_dec_len*(batch_size, 1, max_ctx_len)
        # ret_dict[DecoderRNN.KEY_SEQUENCE]: max_dec_len*(batch_size, 1)
        return prob_outputs, decoder_hidden_state, ret_dict

    def forward_step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        # input_var: (batch_size, response_size-1 i.e. output_seq_len)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: (batch_size, max_ctx_len, ctx_cell_size)
        # goal_hid: (batch_size, goal_nhid)
        batch_size, output_seq_len = input_var.size()
        # (batch_size, output_seq_len, embedding_dim)
        embedded = self.embedding(input_var)

        # add goals
        if goal_hid is not None:
            # (batch_size, 1, goal_nhid)
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1))
            # (batch_size, output_seq_len, goal_nhid)
            goal_rep = goal_hid.repeat(1, output_seq_len, 1)
            # (batch_size, output_seq_len, embedding_dim+goal_nhid)
            embedded = th.cat([embedded, goal_rep], dim=2)

        embedded = self.input_dropout(embedded)

        # ############
        # embedded = self.FC(embedded.view(-1, embedded.size(-1))).view(batch_size, output_seq_len, -1)

        # output: (batch_size, output_seq_len, dec_cell_size)
        # hidden: tuple: (h, c)
        output, hidden_s = self.rnn(embedded, hidden_state)

        attn = None
        if self.use_attn:
            # output: (batch_size, output_seq_len, dec_cell_size)
            # encoder_outputs: (batch_size, max_ctx_len, ctx_cell_size)
            # attn: (batch_size, output_seq_len, max_ctx_len)
            output, attn = self.attention(output, encoder_outputs)

        # (batch_size*output_seq_len, vocab_size)
        logits = self.project(output.contiguous().view(-1, self.dec_cell_size))
        prediction = self.log_softmax(logits, dim=logits.dim(
        )-1).view(batch_size, output_seq_len, -1)  # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s, attn

    # special for rl
    def _step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        # input_var: (1, 1)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: (1, max_dlg_len, dlg_cell_size)
        # goal_hid: (1, goal_nhid)
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)  # (1, 1, embedding_dim)

        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(
                0), 1, goal_hid.size(1))  # (1, 1, goal_nhid)
            goal_rep = goal_hid.repeat(
                1, output_seq_len, 1)  # (1, 1, goal_nhid)
            # (1, 1, embedding_dim+goal_nhid)
            embedded = th.cat([embedded, goal_rep], dim=2)

        embedded = self.input_dropout(embedded)

        # ############
        # embedded = self.FC(embedded.view(-1, embedded.size(-1))).view(batch_size, output_seq_len, -1)

        # output: (1, 1, dec_cell_size)
        # hidden: tuple: (h, c)
        output, hidden_s = self.rnn(embedded, hidden_state)

        attn = None
        if self.use_attn:
            # output: (1, 1, dec_cell_size)
            # encoder_outputs: (1, max_dlg_len, dlg_cell_size)
            # attn: (1, 1, max_dlg_len)
            output, attn = self.attention(output, encoder_outputs)

        # (1*1, vocab_size)
        logits = self.project(output.view(-1, self.dec_cell_size))
        prediction = logits.view(
            batch_size, output_seq_len, -1)  # (1, 1, vocab_size)
        # prediction = self.log_softmax(logits, dim=logits.dim()-1).view(batch_size, output_seq_len, -1) # (batch_size, output_seq_len, vocab_size)
        return prediction, hidden_s

    # special for rl
    def write(self, input_var, hidden_state, encoder_outputs, max_words, vocab, stop_tokens, goal_hid=None, mask=True,
              decoding_masked_tokens=DECODING_MASKED_TOKENS):
        # input_var: (1, 1)
        # hidden_state: tuple: (h, c)
        # encoder_outputs: max_dlg_len*(1, 1, dlg_cell_size)
        # goal_hid: (1, goal_nhid)
        logprob_outputs = []  # list of logprob | max_dec_len*(1, )
        symbol_outputs = []  # list of word ids | max_dec_len*(1, )
        decoder_input = input_var
        decoder_hidden_state = hidden_state
        if type(encoder_outputs) is list:
            # (1, max_dlg_len, dlg_cell_size)
            encoder_outputs = th.cat(encoder_outputs, 1)
        # print('encoder_outputs.size() = {}'.format(encoder_outputs.size()))

        if mask:
            special_token_mask = Variable(th.FloatTensor(
                [-999. if token in decoding_masked_tokens else 0. for token in vocab]))
            special_token_mask = cast_type(
                special_token_mask, FLOAT, self.use_gpu)  # (vocab_size, )

        def _sample(dec_output, num_i):
            # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
            dec_output = dec_output.view(-1)  # (vocab_size, )
            # TODO temperature
            prob = F.softmax(dec_output/0.6, dim=0)  # (vocab_size, )
            logprob = F.log_softmax(dec_output, dim=0)  # (vocab_size, )
            symbol = prob.multinomial(num_samples=1).detach()  # (1, )
            # _, symbol = prob.topk(1) # (1, )
            _, tmp_symbol = prob.topk(1)  # (1, )
            # print('multinomial symbol = {}, prob = {}'.format(symbol, prob[symbol.item()]))
            # print('topk symbol = {}, prob = {}'.format(tmp_symbol, prob[tmp_symbol.item()]))
            logprob = logprob.gather(0, symbol)  # (1, )
            return logprob, symbol

        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(
                decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            # disable special tokens from being generated in a normal turn
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(1, -1)

            if vocab[symbol.item()] in stop_tokens:
                break

        assert len(logprob_outputs) == len(symbol_outputs)
        # logprob_list = [t.item() for t in logprob_outputs]
        logprob_list = logprob_outputs
        symbol_list = [t.item() for t in symbol_outputs]
        return logprob_list, symbol_list

    # For MultiWoz RL
    def forward_rl(self, batch_size, dec_init_state, attn_context, vocab, max_words, goal_hid=None, mask=True, temp=0.1):
        # prepare the BOS inputs
        with th.no_grad():
            bos_var = Variable(th.LongTensor([self.sys_id]))
        bos_var = cast_type(bos_var, LONG, self.use_gpu)
        decoder_input = bos_var.expand(batch_size, 1)  # (1, 1)
        decoder_hidden_state = dec_init_state  # tuple: (h, c)
        encoder_outputs = attn_context  # (1, ctx_len, ctx_cell_size)

        logprob_outputs = []  # list of logprob | max_dec_len*(1, )
        symbol_outputs = []  # list of word ids | max_dec_len*(1, )

        if mask:
            special_token_mask = Variable(th.FloatTensor(
                [-999. if token in DECODING_MASKED_TOKENS else 0. for token in vocab]))
            special_token_mask = cast_type(
                special_token_mask, FLOAT, self.use_gpu)  # (vocab_size, )

        def _sample(dec_output, num_i):
            # dec_output: (1, 1, vocab_size), need to softmax and log_softmax
            # (batch_size, vocab_size, )
            dec_output = dec_output.view(batch_size, -1)
            # (batch_size, vocab_size, )
            prob = F.softmax(dec_output/temp, dim=1)
            # (batch_size, vocab_size, )
            logprob = F.log_softmax(dec_output, dim=1)
            symbol = prob.multinomial(
                num_samples=1).detach()  # (batch_size, 1)
            # _, symbol = prob.topk(1) # (1, )
            _, tmp_symbol = prob.topk(1)  # (1, )
            # print('multinomial symbol = {}, prob = {}'.format(symbol, prob[symbol.item()]))
            # print('topk symbol = {}, prob = {}'.format(tmp_symbol, prob[tmp_symbol.item()]))
            logprob = logprob.gather(1, symbol)  # (1, )
            return logprob, symbol

        stopped_samples = set()
        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(
                decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            # disable special tokens from being generated in a normal turn
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(batch_size, -1)
            for b_id in range(batch_size):
                if vocab[symbol[b_id].item()] == EOS:
                    stopped_samples.add(b_id)

            if len(stopped_samples) == batch_size:
                break

        assert len(logprob_outputs) == len(symbol_outputs)
        symbol_outputs = th.cat(
            symbol_outputs, dim=1).cpu().data.numpy().tolist()
        logprob_outputs = th.cat(logprob_outputs, dim=1)
        logprob_list = []
        symbol_list = []
        for b_id in range(batch_size):
            b_logprob = []
            b_symbol = []
            for t_id in range(logprob_outputs.shape[1]):
                symbol = symbol_outputs[b_id][t_id]
                if vocab[symbol] == EOS and t_id != 0:
                    break

                b_symbol.append(symbol_outputs[b_id][t_id])
                b_logprob.append(logprob_outputs[b_id][t_id])

            logprob_list.append(b_logprob)
            symbol_list.append(b_symbol)

        # TODO backward compatible, if batch_size == 1, we remove the nested structure
        if batch_size == 1:
            logprob_list = logprob_list[0]
            symbol_list = symbol_list[0]

        return logprob_list, symbol_list
