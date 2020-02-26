import torch as th
import torch.nn as nn
import numpy as np
from torch.nn.modules.module import _addindent

def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            th.nn.modules.container.Container,
            th.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    return tmpstr


class BaseRNN(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, input_dropout_p, rnn_cell, 
                     input_size, hidden_size, num_layers, 
                     output_dropout_p, bidirectional):
        super(BaseRNN, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell Type: {0}'.format(rnn_cell))
        self.rnn = self.rnn_cell(input_size=input_size, 
                                 hidden_size=hidden_size,
                                 num_layers=num_layers, 
                                 batch_first=True, 
                                 dropout=output_dropout_p, 
                                 bidirectional=bidirectional)

        # TODO Trick for initializing LSTM gate parameters
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)
