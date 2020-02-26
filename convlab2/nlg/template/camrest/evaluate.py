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

from convlab2.nlg.template.camrest import TemplateNLG

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
            if da == 'request' or da == 'nooffer':
                continue
            else:
                for s, v in sorted(svs, key=lambda x: x[0]):
                    if s == 'none' or v == 'none':
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

    archive = zipfile.ZipFile('../../../../data/camrest/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))

    dialog_acts = []
    golden_utts = []
    gen_utts = []
    gen_slots = []

    sen_num = 0
    sess_num = 0
    for sess in test_data:
        sess_num+=1
        print('[%d/%d]' % (sess_num, len(test_data)))
        for i, turn in enumerate(sess['dial']):
            if data_key == 'usr' or data_key == 'all':
                sen_num += 1
                model = model_usr
                dialog_acts.append(turn['usr']['dialog_act'])
                golden_utts.append(turn['usr']['transcript'])
                gen_utts.append(model.generate(turn['usr']['dialog_act']))
            if data_key == 'sys' or data_key == 'all':
                sen_num += 1
                model = model_sys
                dialog_acts.append(turn['sys']['dialog_act'])
                golden_utts.append(turn['sys']['sent'])
                gen_utts.append(model.generate(turn['sys']['dialog_act']))

    bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts)

    print("Calculate bleu-4")
    print("BLEU-4: %.4f" % bleu4)

    print('Model on {} session {} sentences data_key={}'.format(len(test_data), sen_num, data_key))
