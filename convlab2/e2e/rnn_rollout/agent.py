# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from collections import defaultdict

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import convlab2.e2e.rnn_rollout.utils as utils
from convlab2.e2e.rnn_rollout.dialog import DialogLogger
import convlab2.e2e.rnn_rollout.vis as vis
import convlab2.e2e.rnn_rollout.domain as domain
from convlab2.e2e.rnn_rollout.engines import Criterion
import math
from collections import Counter
import copy

from convlab2.dialog_agent import Agent


class RnnRolloutAgent(Agent):
    def __init__(self, model, args, name='Alice', allow_no_agreement=True, train=False, diverse=False, max_dec_len=20):
        self.model = model
        self.model.eval()
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)
        self.allow_no_agreement = allow_no_agreement
        self.max_dec_len = max_dec_len

        self.sel_model = utils.load_model(args.selection_model_file)
        self.sel_model.eval()
        self.ncandidate = 5
        self.nrollout = 3
        self.rollout_len = 100

    def response(self, observation):
        self.read(observation)
        output = self.write(max_words=self.max_dec_len)
        return output

    def init_session(self):
        pass

    def _encode(self, inpt, dictionary):
        encoded = torch.Tensor(dictionary.w2i(inpt)).long().unsqueeze(1)
        return encoded

    def _decode(self, out, dictionary):
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context):
        self.lang_hs = []
        self.sents = []
        self.words = []
        self.context = context
        self.ctx = self._encode(context, self.model.context_dict)
        self.ctx_h = self.model.forward_context(Variable(self.ctx))
        self.lang_h = self.model.zero_h(1, self.model.args.nhid_lang)

    def feed_partner_context(self, partner_context):
        pass

    def update(self, agree, reward, choice=None, partner_choice=None,
            partner_input=None, partner_reward=None):
        pass

    def read(self, inpt):
        self.sents.append(Variable(self._encode(['THEM:'] + inpt, self.model.word_dict)))
        inpt = self._encode(inpt, self.model.word_dict)
        lang_hs, self.lang_h = self.model.read(Variable(inpt), self.lang_h, self.ctx_h)
        self.lang_hs.append(lang_hs.squeeze(1))
        self.words.append(self.model.word2var('THEM:').unsqueeze(0))
        self.words.append(Variable(inpt))
        assert (torch.cat(self.words).size(0) == torch.cat(self.lang_hs).size(0))

    def _make_idxs(self, sents):
        lens, rev_idxs, hid_idxs = [], [], []
        for sent in sents:
            assert sent.size(1) == 1
            # remove last hidden state
            ln = torch.Tensor(1).fill_(sent.size(0) - 1).long()
            lens.append(ln)
            idx = torch.Tensor(sent.size(0), 1, 1).fill_(-1).long()
            for j in range(idx.size(0)):
                idx[j][0][0] = j if j >= sent.size(0) else sent.size(0) - j - 1
            rev_idxs.append(Variable(idx))
            hid_idxs.append(Variable(ln.view(1, 1, 1)))
        return lens, rev_idxs, hid_idxs

    def _choose(self, sample=False):
        sents = self.sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, Variable(self.ctx))

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].item())
        prob = F.softmax(choice_logit, dim=0)

        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=0).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.item()]

        # Pick only your choice
        return choices[idx.item()][:self.domain.selection_length()], logprob, p_agree.item()

    def choose(self):
        choice, _, _ = self._choose()
        return choice

    def __choose(self, local_sents, sample=False):
        sents = local_sents[:-1]
        lens, rev_idxs, hid_idxs = self._make_idxs(sents)
        sel_out = self.sel_model.forward(sents, lens, rev_idxs, hid_idxs, Variable(self.ctx))

        choices = self.domain.generate_choices(self.context, with_disagreement=True)

        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.sel_model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.Tensor(idxs).long())
            choices_logits.append(torch.gather(sel_out[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=True).squeeze(1)
        choice_logit = choice_logit.sub(choice_logit.max(0)[0].item())
        prob = F.softmax(choice_logit, dim=0)

        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=0).gather(0, idx)
        else:
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.item()]

        # Pick only your choice
        return choices[idx.item()][:self.domain.selection_length()], logprob, p_agree.item()

    def write(self, max_words=20):
        
        # print('\t\trollout write')
        best_score = -1
        res = None

        # print('start rollout')
        for _ in range(self.ncandidate):
            # print('\tcandidate')
            _, move, move_lang_h, move_lang_hs = self.model.write(
                self.lang_h, self.ctx_h, max_words, self.args.temperature)

            is_selection = len(move) == 1 and \
                self.model.word_dict.get_word(move.data[0][0]) == '<selection>'  # whether the candidate is a terminated

            score = 0
            for _ in range(self.nrollout):
                # print('\trollout')
                combined_lang_hs = self.lang_hs + [move_lang_hs]
                combined_words = self.words + [self.model.word2var('YOU:').view(1, 1), move]
                combined_sents = copy.deepcopy(self.sents)
                combined_sents.append(torch.cat([self.model.word2var('YOU:').unsqueeze(1), move], 0))

                last_lang_h = move_lang_h
                if not is_selection:  # if not terminated
                    # Complete the conversation with rollout_length samples
                    # _, rollout, _, rollout_lang_hs = self.model.write(
                    #     move_lang_h, self.ctx_h, self.rollout_len, self.args.temperature,
                    #     stop_tokens=['<selection>'], resume=True)
                    side_tag = False
                    rollout_len = 0
                    for _ in range(10):
                        acts, outs, last_lang_h, lang_hs = self.model.write(last_lang_h, self.ctx_h,
                                                                            max_words, self.args.temperature)
                        tag = 'YOU:' if side_tag else 'THEM:'
                        is_select = len(outs) == 1 and self.model.word_dict.get_word(outs.data[0][0]) == '<selection>'
                        combined_sents.append(torch.cat([self.model.word2var(tag).unsqueeze(1), outs], 0))
                        combined_lang_hs += [lang_hs]
                        combined_words += [outs]
                        rollout_len += 1
                        if is_select:
                            break
                        side_tag = not side_tag
                    # print('rollout_len: {}'.format(rollout_len))
                    # combined_lang_hs += [rollout_lang_hs]
                    # combined_words += [rollout]

                # Choose items
                rollout_score = None

                combined_lang_hs = torch.cat(combined_lang_hs)
                combined_words = torch.cat(combined_words)
                rollout_choice, _, p_agree = self.__choose(combined_sents, sample=False)
                rollout_score = self.domain.score(self.context, rollout_choice)
                score += p_agree * rollout_score

            # Take the candidate with the max expected reward
            if score > best_score:
                res = (move, move_lang_h, move_lang_hs)
                best_score = score

        outs, lang_h, lang_hs = res
        self.lang_h = lang_h
        self.lang_hs.append(lang_hs)
        self.words.append(self.model.word2var('YOU:').unsqueeze(0))
        self.words.append(outs)
        self.sents.append(torch.cat([self.model.word2var('YOU:').unsqueeze(1), outs], 0))
        return self._decode(outs, self.model.word_dict)
