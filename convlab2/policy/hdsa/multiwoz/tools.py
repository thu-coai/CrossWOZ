import transformer.Constants as Constants
import json
import math
import re
from collections import Counter
from nltk.util import ngrams
import numpy
import torch

def get_n_params(*params_list):
    pp=0
    for params in params_list:
        for p in params:
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
    return pp

def filter_sents(sents, END):
    hyps = []
    for batch_id in range(len(sents)):
        done = False
        for beam_id in range(len(sents[batch_id])):
            sent = sents[batch_id][beam_id]
            for s in sent[::-1]:
                if s in [Constants.PAD, Constants.EOS]:
                    pass
                elif s in END:
                    done = True
                    break
                elif s not in END:
                    done = False
                    break
            if done:
                hyps.append(sent)
                break
        if len(hyps) < batch_id + 1:
            hyps.append(sents[batch_id][0])
    return hyps

def obtain_TP_TN_FN_FP(pred, act, TP, TN, FN, FP, elem_wise=False):
    if isinstance(pred, torch.Tensor):
        if elem_wise:
            TP += ((pred.data == 1) & (act.data == 1)).sum(0)
            TN += ((pred.data == 0) & (act.data == 0)).sum(0)
            FN += ((pred.data == 0) & (act.data == 1)).sum(0)
            FP += ((pred.data == 1) & (act.data == 0)).sum(0)
        else:
            TP += ((pred.data == 1) & (act.data == 1)).cpu().sum().item()
            TN += ((pred.data == 0) & (act.data == 0)).cpu().sum().item()
            FN += ((pred.data == 0) & (act.data == 1)).cpu().sum().item()
            FP += ((pred.data == 1) & (act.data == 0)).cpu().sum().item()
        return TP, TN, FN, FP
    else:
        TP += ((pred > 0).astype('long') & (act > 0).astype('long')).sum()
        TN += ((pred == 0).astype('long') & (act == 0).astype('long')).sum()
        FN += ((pred == 0).astype('long') & (act > 0).astype('long')).sum()
        FP += ((pred > 0).astype('long') & (act == 0).astype('long')).sum()
        return TP, TN, FN, FP

class F1Scorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, hypothesis, corpus, n=1):
        # containers
        with open('data/placeholder.json') as f:
            placeholder = json.load(f)['placeholder']

        TP, TN, FN, FP = 0, 0, 0, 0
        # accumulate ngram statistics
        files = hypothesis.keys()
        for f in files:
            hyps = hypothesis[f]
            refs = corpus[f]

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # Shawn's evaluation
            #refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            #hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            for hyp, ref in zip(hyps, refs):
                pred = numpy.zeros((len(placeholder), ), 'float32')
                gt = numpy.zeros((len(placeholder), ), 'float32')
                for h in hyp:
                    if h in placeholder:
                        pred[placeholder.index(h)] += 1
                for r in ref:
                    if r in placeholder:
                        gt[placeholder.index(r)] += 1
                TP, TN, FN, FP = obtain_TP_TN_FN_FP(pred, gt, TP, TN, FN, FP)

        precision = TP / (TP + FP + 0.001)
        recall = TP / (TP + FN + 0.001)
        F1 = 2 * precision * recall / (precision + recall + 0.001)
        return F1

def sentenceBLEU(hyps, refs, n=1):
    count = [0, 0, 0, 0]
    clip_count = [0, 0, 0, 0]
    r = 0
    c = 0
    weights = [0.25, 0.25, 0.25, 0.25]
    hyps = [hyp.split() for hyp in hyps]
    refs = [ref.split() for ref in refs]
    # Shawn's evaluation
    refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
    hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
    for idx, hyp in enumerate(hyps):
        for i in range(4):
            # accumulate ngram counts
            hypcnts = Counter(ngrams(hyp, i + 1))
            cnt = sum(hypcnts.values())
            count[i] += cnt

            # compute clipped counts
            max_counts = {}
            for ref in refs:
                refcnts = Counter(ngrams(ref, i + 1))
                for ng in hypcnts:
                    max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
            clipcnt = dict((ng, min(count, max_counts[ng])) \
                           for ng, count in hypcnts.items())
            clip_count[i] += sum(clipcnt.values())

        # accumulate r & c
        bestmatch = [1000, 1000]
        for ref in refs:
            if bestmatch[0] == 0: break
            diff = abs(len(ref) - len(hyp))
            if diff < bestmatch[0]:
                bestmatch[0] = diff
                bestmatch[1] = len(ref)
        r += bestmatch[1]
        c += len(hyp)
        if n == 1:
            break
    p0 = 1e-7
    bp = 1 if c > r else math.exp(1 - float(r) / float(c))
    p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
            for i in range(4)]
    s = math.fsum(w * math.log(p_n) \
                  for w, p_n in zip(weights, p_ns) if p_n)
    bleu = bp * math.exp(s)
    return bleu

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, old_hypothesis, old_corpus, n=1):
        file_names = old_hypothesis.keys()
        hypothesis = []
        corpus = []
        for f in file_names:
            old_h = old_hypothesis[f]
            old_c = old_corpus[f]
            for h, c in zip(old_h, old_c):
                hypothesis.append([h])
                corpus.append([c])
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]
        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            # Shawn's evaluation
            refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']
            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu

class Tokenizer(object):
    def __init__(self, vocab, ivocab, use_field, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = ivocab
        self.vocab = vocab
        self.use_field = use_field
        if use_field:
            with open('data/placeholder.json') as f:
                self.fields = json.load(f)['field']

        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return sent.lower().split()
        else:
            return sent.split()

    def get_word_id(self, w, template=None):
        if self.use_field and template:
            if w in self.fields and w in template:
                return template.index(w) + self.vocab_len

        if w in self.vocab:
            return self.vocab[w]
        else:
            return self.vocab[Constants.UNK_WORD]


    def get_word(self, k, template=None):
        if k > self.vocab_len and self.use_field and template:
            return template[k - self.vocab_len]
        else:
            k = str(k)
            return self.ivocab[k]

    def convert_tokens_to_ids(self, sent, template=None):
        return [self.get_word_id(w, template) for w in sent]

    def convert_id_to_tokens(self, word_ids, template_ids=None, remain_eos=False):
        if isinstance(word_ids, list):
            if remain_eos:
                return " ".join([self.get_word(wid, None) for wid in word_ids
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid, None) for wid in word_ids
                                 if wid not in [Constants.PAD, Constants.EOS] ])
        else:
            if remain_eos:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids
                                 if wid not in [Constants.PAD, Constants.EOS]])

    def convert_template(self, template_ids):
        return [self.get_word(wid) for wid in template_ids if wid != Constants.PAD]

def nondetokenize(d_p, d_r):
    dialog_id = 0
    need_replace = 0
    success = 0
    for gt_dialog_info in d_r:
        file_name = gt_dialog_info['file']
        gt_dialog = gt_dialog_info['info']
        for turn_id in range(len(d_p[file_name])):
            kb = gt_dialog[turn_id]['source']
            act = gt_dialog[turn_id]['act']
            words = d_p[file_name][turn_id].split(' ')
            for i in range(len(words)):
                if "[" in words[i] and "]" in words[i]:
                    need_replace += 1.
                    if words[i] in kb:
                        words[i] = kb[words[i]]
                        success += 1.
                    elif "taxi" in words[i]:
                        if words[i] == "[taxi_type]" and "domain-taxi-inform-car" in act:
                            words[i] = act["domain-taxi-inform-car"]
                            success += 1.
                        elif words[i] == "[taxi_phone]" and "domain-taxi-inform-phone" in act:
                            words[i] = act["domain-taxi-inform-phone"]
                            success += 1.

            d_p[file_name][turn_id] = " ".join(words)
    success_rate = success / need_replace
    return success_rate
