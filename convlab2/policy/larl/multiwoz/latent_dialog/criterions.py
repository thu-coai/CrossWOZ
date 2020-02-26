import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from convlab2.policy.larl.multiwoz.latent_dialog import domain
from convlab2.policy.larl.multiwoz.latent_dialog.utils import LONG


class NLLEntropy(_Loss):
    def __init__(self, padding_idx, avg_type):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = avg_type

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        pred = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)

        if self.avg_type is None:
            loss = F.nll_loss(pred, target, size_average=False,
                              ignore_index=self.padding_idx)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(pred, target, size_average=False,
                              ignore_index=self.padding_idx)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(
                pred, target, ignore_index=self.padding_idx, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = th.sum(loss, dim=1)
            word_cnt = th.sum(th.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = th.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(pred, target, size_average=True,
                              ignore_index=self.padding_idx)
        else:
            raise ValueError('Unknown average type')

        return loss


class NLLEntropy4CLF(_Loss):
    def __init__(self, dictionary, bad_tokens=['<disconnect>', '<disagree>'], reduction='elementwise_mean'):
        super(NLLEntropy4CLF, self).__init__()
        w = th.Tensor(len(dictionary)).fill_(1)
        for token in bad_tokens:
            w[dictionary[token]] = 0.0
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def forward(self, preds, labels):
        # preds: (batch_size, outcome_len, outcome_vocab_size)
        # labels: (batch_size, outcome_len)
        preds = preds.view(-1, preds.size(-1))
        labels = labels.view(-1)
        return self.crit(preds, labels)


class CombinedNLLEntropy4CLF(_Loss):
    def __init__(self, dictionary, corpus, np2var, bad_tokens=['<disconnect>', '<disagree>']):
        super(CombinedNLLEntropy4CLF, self).__init__()
        self.dictionary = dictionary
        self.domain = domain.get_domain('object_division')
        self.corpus = corpus
        self.np2var = np2var
        self.bad_tokens = bad_tokens

    def forward(self, preds, goals_id, outcomes_id):
        # preds: (batch_size, outcome_len, outcome_vocab_size)
        # goals_id: list of list, id, batch_size*goal_len
        # outcomes_id: list of list, id, batch_size*outcome_len
        batch_size = len(goals_id)
        losses = []
        for bth in range(batch_size):
            pred = preds[bth]  # (outcome_len, outcome_vocab_size)
            goal = goals_id[bth]  # list, id, len=goal_len
            goal_str = self.corpus.id2goal(goal)  # list, str, len=goal_len
            outcome = outcomes_id[bth]  # list, id, len=outcome_len
            outcome_str = self.corpus.id2outcome(
                outcome)  # list, str, len=outcome_len

            if outcome_str[0] in self.bad_tokens:
                continue

            # get all the possible choices
            choices = self.domain.generate_choices(goal_str)
            # outcome_len*(outcome_vocab_size, )
            sel_outs = [pred[i] for i in range(pred.size(0))]

            choices_logits = []  # outcome_len*(option_amount, 1)
            for i in range(self.domain.selection_length()):
                idxs = np.array([self.dictionary[c[i]] for c in choices])
                idxs_var = self.np2var(idxs, LONG)  # (option_amount, )
                choices_logits.append(
                    th.gather(sel_outs[i], 0, idxs_var).unsqueeze(1))

            choice_logit = th.sum(th.cat(choices_logits, 1),
                                  1, keepdim=False)  # (option_amount, )
            choice_logit = choice_logit.sub(
                choice_logit.max().item())  # (option_amount, )
            prob = F.softmax(choice_logit, dim=0)  # (option_amount, )

            label = choices.index(outcome_str)
            target_prob = prob[label]
            losses.append(-th.log(target_prob))
        return sum(losses) / float(len(losses))


class CatKLLoss(_Loss):
    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        qy = th.exp(log_qy)
        y_kl = th.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return th.mean(y_kl)
        else:
            return th.sum(y_kl)/batch_size


class Entropy(_Loss):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = th.exp(log_qy)
        h_q = th.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return th.mean(h_q)
        else:
            return th.sum(h_q) / batch_size


class BinaryNLLEntropy(_Loss):

    def __init__(self, size_average=True):
        super(BinaryNLLEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, net_output, label_output):
        """
        :param net_output: batch_size x
        :param labels:
        :return:
        """
        batch_size = net_output.size(0)
        loss = F.binary_cross_entropy_with_logits(
            net_output, label_output, size_average=self.size_average)
        if self.size_average is False:
            loss /= batch_size
        return loss


class NormKLLoss(_Loss):
    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        # find the KL divergence between two Gaussian distribution
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= th.div(th.pow(prior_mu - recog_mu, 2), th.exp(prior_logvar))
        loss -= th.div(th.exp(recog_logvar), th.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * th.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * th.sum(loss, dim=1)
        avg_kl_loss = th.mean(kl_loss)
        return avg_kl_loss
