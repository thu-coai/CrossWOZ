import numpy as np
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
from convlab.modules.word_policy.multiwoz.larl.latent_dialog import domain
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import USR, SYS, BOD, EOS, SEL


class Judger(object):
    def __init__(self, model, device_id):
        self.model = model
        self.domain = domain.get_domain('object_division')
        self.device_id = device_id

    def _process(self, dlg_text):
        token_seq = []
        for turn in dlg_text:
            assert turn[0] in [USR, SYS]
            if turn[0] == USR:
                turn[0] = SYS
            else:
                turn[0] = USR
            token_seq.extend(turn)
        return token_seq

    def choose(self, context, dlg_text):
        token_seq = self._process(dlg_text)
        assert token_seq[0] in [USR, SYS]
        assert token_seq[1] == BOD
        assert token_seq[2] == EOS
        assert token_seq[-1] == EOS
        assert token_seq[-2] == SEL
        assert token_seq[-3] in [USR, SYS]
        token_seq = token_seq[3:-1]
        goal_idxs = self.model.context_dict.w2i(context)
        word_idxs = self.model.word_dict.w2i(token_seq)
        goal = th.LongTensor(goal_idxs).unsqueeze(1) # (6, 1)
        word = th.LongTensor(word_idxs).unsqueeze(1) # (sent_len, 1)
        if self.device_id is not None:
            goal = goal.cuda(self.device_id)
            word = word.cuda(self.device_id)
        goal = Variable(goal)
        word = Variable(word)

        # get context hidden state
        ctx_h = self.model.forward_context(goal) # (1, 1, nhid_ctx)
        # create initial hidden state for the language rnn
        lang_h = self.model.zero_hid(ctx_h.size(1), self.model.args.nhid_lang) # (1, 1, nhid_lang)
        # perform forward for the language model
        out, lang_h = self.model.forward_lm(word, lang_h, ctx_h)
        # out: (sent_len, 1, vocab_size)
        # lang_h: (sent_len, 1, nhid_lang)
        lang_h = lang_h.squeeze(1) # (sent_len, nhid_lang)

        # logits for each of the item
        logits = self.model.generate_choice_logits(word, lang_h, ctx_h) # 6*(len(item_dict), )
        # get all the possible choices
        choices = self.domain.generate_choices(context)
        # construct probability distribution over only the valid choices
        choices_logits = [] # 6*(option_amount, 1)
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(th.from_numpy(np.array(idxs)))
            idxs = self.model.to_device(idxs) # (option_amount, )
            choices_logits.append(th.gather(logits[i], 0, idxs).unsqueeze(1))
        choice_logit = th.sum(th.cat(choices_logits, 1), 1, keepdim=False) # (option_amount, )
        # subtract the max to softmax more stable
        choice_logit = choice_logit.sub(choice_logit.max().item()) # (option_amount, )
        prob = F.softmax(choice_logit, dim=0) # (option_amount, )
        p_test, idx = prob.max(0, keepdim=True) # idx: (1, )

        return choices[idx.item()][:self.domain.selection_length()]
