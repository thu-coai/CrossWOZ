import torch as th
import torch.nn as nn
import torch.nn.functional as F
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.enc2dec.base_modules import BaseRNN


class EncoderGRUATTN(BaseRNN):
    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderGRUATTN, self).__init__(input_dropout_p=input_dropout_p, 
                                             rnn_cell=rnn_cell, 
                                             input_size=input_size, 
                                             hidden_size=hidden_size, 
                                             num_layers=num_layers, 
                                             output_dropout_p=output_dropout_p, 
                                             bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.nhid_attn = hidden_size
        self.output_size = hidden_size*2 if bidirectional else hidden_size

        # attention to combine selection hidden states
        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), 
            nn.Tanh(), 
            nn.Linear(hidden_size, 1)
        )

    def forward(self, residual_var, input_var, turn_feat, mask=None, init_state=None, input_lengths=None):
        # residual_var: (batch_size, max_dlg_len, 2*utt_cell_size)
        # input_var: (batch_size, max_dlg_len, dlg_cell_size)

        # TODO switch of mask
        # mask = None
        
        require_embed = True
        if require_embed:
            # input_cat = th.cat([input_var, residual_var], 2) # (batch_size, max_dlg_len, dlg_cell_size+2*utt_cell_size)
            input_cat = th.cat([input_var, residual_var, turn_feat], 2) # (batch_size, max_dlg_len, dlg_cell_size+2*utt_cell_size)
        else:
            # input_cat = th.cat([input_var], 2)
            input_cat = th.cat([input_var, turn_feat], 2)
        if mask is not None:
            input_mask = mask.view(input_cat.size(0), input_cat.size(1), 1) # (batch_size, max_dlg_len*max_utt_len, 1)
            input_cat = th.mul(input_cat, input_mask)
        embedded = self.input_dropout(input_cat)
        
        require_rnn = True
        if require_rnn:
            if init_state is not None:
                h, _ = self.rnn(embedded, init_state)
            else:
                h, _ = self.rnn(embedded) # (batch_size, max_dlg_len, 2*nhid_attn)
    
            logit = self.attn(h.contiguous().view(-1, 2*self.nhid_attn)).view(h.size(0), h.size(1)) # (batch_size, max_dlg_len)
            # if mask is not None:
            #     logit_mask = mask.view(input_cat.size(0), input_cat.size(1))
            #     logit_mask = -999.0 * logit_mask
            #     logit = logit_mask + logit
    
            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(h) # (batch_size, max_dlg_len, 2*nhid_attn)
            attn = th.sum(th.mul(h, prob), 1) # (batch_size, 2*nhid_attn)
            
            return attn

        else:
            logit = self.attn(embedded.contiguous().view(input_cat.size(0)*input_cat.size(1), -1)).view(input_cat.size(0), input_cat.size(1))
            if mask is not None:
                logit_mask = mask.view(input_cat.size(0), input_cat.size(1))
                logit_mask = -999.0 * logit_mask
                logit = logit_mask + logit

            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(embedded) # (batch_size, max_dlg_len, 2*nhid_attn)
            attn = th.sum(th.mul(embedded, prob), 1) # (batch_size, 2*nhid_attn)
            
            return attn


class FeatureProjecter(nn.Module):
    def __init__(self, input_dropout_p, input_size, output_size):
        super(FeatureProjecter, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.sel_encoder = nn.Sequential(
            nn.Linear(input_size, output_size), 
            nn.Tanh()
        )

    def forward(self, goals_h, attn_outs):
        h = th.cat([attn_outs, goals_h], 1) # (batch_size, 2*nhid_attn+goal_nhid)
        h = self.input_dropout(h)
        h = self.sel_encoder.forward(h) # (batch_size, nhid_sel)
        return h


class SelectionClassifier(nn.Module):
    def __init__(self, selection_length, input_size, output_size):
        super(SelectionClassifier, self).__init__()
        self.sel_decoders = nn.ModuleList()
        for _ in range(selection_length):
            self.sel_decoders.append(nn.Linear(input_size, output_size))

    def forward(self, proj_outs):
        outs = [decoder.forward(proj_outs).unsqueeze(1) for decoder in self.sel_decoders] # outcome_len*(batch_size, 1, outcome_vocab_size)
        outs = th.cat(outs, 1) # (batch_size, outcome_len, outcome_vocab_size)
        return outs
