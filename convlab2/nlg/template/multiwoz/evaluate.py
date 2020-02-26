"""
Evaluate NLG models on utterances of Multiwoz test dataset
Metric: dataset level BLEU-4, slot error rate
Usage: python evaluate.py [usr|sys|all]
"""
import json
import random
import sys
import zipfile

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from convlab2.nlg.template.multiwoz import TemplateNLG

seed = 2019
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_bleu4(dialog_acts, golden_utts, gen_utts):
    das2utts = {}
    for das, utt, gen in zip(dialog_acts, golden_utts, gen_utts):
        utt = utt.lower()
        gen = gen.lower()
        for da, svs in das.items():
            domain, act = da.split('-')
            if act == 'Request' or domain == 'general':
                continue
            else:
                for s, v in sorted(svs, key=lambda x: x[0]):
                    if s == 'Internet' or s == 'Parking' or s == 'none' or v == 'none':
                        continue
                    else:
                        v = v.lower()
                        if (' ' + v in utt) or (v + ' ' in utt):
                            utt = utt.replace(v, '{}-{}'.format(da, s), 1)
                        if (' ' + v in gen) or (v + ' ' in gen):
                            gen = gen.replace(v, '{}-{}'.format(da, s), 1)
        hash_key = ''
        for da in sorted(das.keys()):
            for s, v in sorted(das[da], key=lambda x: x[0]):
                hash_key += da + '-' + s + ';'
        das2utts.setdefault(hash_key, {'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append(gen)
    # pprint(das2utts)
    refs, gens = [], []
    for das in das2utts.keys():
        for gen in das2utts[das]['gens']:
            refs.append([s.split() for s in das2utts[das]['refs']])
            gens.append(gen.split())
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return bleu


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:")
        print("\t python evaluate.py data_key")
        print("\t data_key=usr/sys/all")
        sys.exit()
    data_key = sys.argv[1]
    if data_key=='all' or data_key=='usr':
        model_usr = TemplateNLG(is_user=True)
    if data_key=='all' or data_key=='sys':
        model_sys = TemplateNLG(is_user=False)

    archive = zipfile.ZipFile('../../../../data/multiwoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))

    dialog_acts = []
    golden_utts = []
    gen_utts = []
    gen_slots = []

    sen_num = 0
    sess_num = 0
    for no, sess in list(test_data.items()):
        sess_num+=1
        print('[%d/%d]' % (sess_num, len(test_data)))
        for i, turn in enumerate(sess['log']):
            if i % 2 == 0 and data_key == 'sys':
                continue
            elif i % 2 == 1 and data_key == 'usr':
                continue
            sen_num += 1
            model = model_usr if i%2==0 else model_sys
            dialog_acts.append(turn['dialog_act'])
            golden_utts.append(turn['text'])
            gen_utts.append(model.generate(turn['dialog_act']))

    bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts)

    print("Calculate bleu-4")
    print("BLEU-4: %.4f" % bleu4)

    print('Model on {} session {} sentences data_key={}'.format(len(test_data), sen_num, data_key))
