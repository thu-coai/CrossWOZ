import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from convlab2.policy.larl.multiwoz.latent_dialog.utils import cast_type, FLOAT


class IdentityConnector(nn.Module):
    def __init(self):
        super(IdentityConnector, self).__init__()

    def forward(self, hidden_state):
        return hidden_state


class Bi2UniConnector(nn.Module):
    def __init__(self, rnn_cell, num_layer, hidden_size, output_size):
        super(Bi2UniConnector, self).__init__()
        if rnn_cell == 'lstm':
            self.fch = nn.Linear(hidden_size*2*num_layer, output_size)
            self.fcc = nn.Linear(hidden_size*2*num_layer, output_size)
        else:
            self.fc = nn.Linear(hidden_size*2*num_layer, output_size)

        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, hidden_state):
        """
        :param hidden_state: [num_layer, batch_size, feat_size]
        :param inputs: [batch_size, feat_size]
        :return: 
        """
        if self.rnn_cell == 'lstm':
            h, c = hidden_state
            num_layer = h.size()[0]
            flat_h = h.transpose(0, 1).contiguous()
            flat_c = c.transpose(0, 1).contiguous()
            new_h = self.fch(flat_h.view(-1, self.hidden_size*num_layer))
            new_c = self.fch(flat_c.view(-1, self.hidden_size*num_layer))
            return (new_h.view(1, -1, self.output_size),
                    new_c.view(1, -1, self.output_size))
        else:
            # FIXME fatal error here!
            num_layer = hidden_state.size()[0]
            new_s = self.fc(hidden_state.view(-1, self.hidden_size*num_layer))
            new_s = new_s.view(1, -1, self.output_size)
            return new_s


class Hidden2Gaussian(nn.Module):
    def __init__(self, input_size, output_size, is_lstm=False, has_bias=True):
        super(Hidden2Gaussian, self).__init__()
        if is_lstm:
            self.mu_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_h = nn.Linear(input_size, output_size, bias=has_bias)

            self.mu_c = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.mu = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar = nn.Linear(input_size, output_size, bias=has_bias)

        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)

            mu_h, mu_c = self.mu_h(h), self.mu_c(c)
            logvar_h, logvar_c = self.logvar_h(h), self.logvar_c(c)
            return mu_h+mu_c, logvar_h+logvar_c
        else:
            # if inputs.dim() == 3:
            #    inputs = inputs.squeeze(0)
            mu = self.mu(inputs)
            logvar = self.logvar(inputs)
            return mu, logvar


class Hidden2Discrete(nn.Module):
    def __init__(self, input_size, y_size, k_size, is_lstm=False, has_bias=True):
        super(Hidden2Discrete, self).__init__()
        self.y_size = y_size
        self.k_size = k_size
        latent_size = self.k_size*self.y_size
        if is_lstm:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)

            self.p_c = nn.Linear(input_size, latent_size, bias=has_bias)
        else:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)

        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)
            logits = self.p_h(h) + self.p_c(c)
        else:
            logits = self.p_h(inputs)
        logits = logits.view(-1, self.k_size)
        log_qy = F.log_softmax(logits, dim=1)
        return logits, log_qy


class GaussianConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GaussianConnector, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, mu, logvar):
        """
        Sample a sample from a multivariate Gaussian distribution with a diagonal covariance matrix using the
        reparametrization trick.
        TODO: this should be better be a instance method in a Gaussian class.
        :param mu: a tensor of size [batch_size, variable_dim]. Batch_size can be None to support dynamic batching
        :param logvar: a tensor of size [batch_size, variable_dim]. Batch_size can be None.
        :return:
        """
        epsilon = th.randn(logvar.size())
        epsilon = cast_type(Variable(epsilon), FLOAT, self.use_gpu)
        std = th.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z


class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = th.rand(logits.size())
        sample = Variable(-th.log(-th.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim()-1)

    def forward(self, logits, temperature=1.0, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = th.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(
                Variable(th.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y
