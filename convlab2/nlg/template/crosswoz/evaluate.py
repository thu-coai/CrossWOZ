import json
import random
import sys
import zipfile
import copy
import re
import jieba
from collections import defaultdict
import os
import pickle as pkl
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pprint import pprint
from convlab2.nlg.template.crosswoz.nlg import TemplateNLG


seed = 2019
random.seed(seed)


def split_delex_sentence(sen):
    res_sen = ''
    pattern = re.compile(r'(\[[^\[^\]]+\])')
    slots = pattern.findall(sen)
    for slot in slots:
        sen = sen.replace(slot, '[slot]')
    sen = sen.split('[slot]')
    for part in sen:
        part = ' '.join(jieba.lcut(part))
        res_sen += part
        if slots:
            res_sen += ' ' + slots.pop(0) + ' '
    return res_sen


def act2intent(dialog_act: list):
    cur_act = copy.deepcopy(dialog_act)
    if '酒店设施' in cur_act[2]:
        if cur_act[0] == 'Inform':
            cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
        elif cur_act[0] == 'Request':
            cur_act[2] = cur_act[2].split('-')[0]
    if cur_act[0] == 'Select':
        cur_act[2] = '源领域+' + cur_act[3]
    intent = '+'.join(cur_act[:-1])
    if '+'.join(cur_act) == 'Inform+景点+门票+免费' or cur_act[-1] == '无':
        intent = '+'.join(cur_act)
    return intent


def value_replace(sentences, dialog_act):
    dialog_act = copy.deepcopy(dialog_act)
    intent_frequency = defaultdict(int)
    for act in dialog_act:
        intent = act2intent(copy.deepcopy(act))
        intent_frequency[intent] += 1
        if intent_frequency[intent] > 1:  # if multiple same intents...
            intent += str(intent_frequency[intent])

        if '酒店设施' in intent:
            try:
                sentences = sentences.replace('[' + intent + ']', act[2].split('-')[1])
                sentences = sentences.replace('[' + intent + '1]', act[2].split('-')[1])
            except Exception as e:
                print('Act causing problem in replacement:')
                pprint(act)
                raise e
        else:
            sentences = sentences.replace('[' + intent + ']', act[3])
            sentences = sentences.replace('[' + intent + '1]', act[3])  # if multiple same intents and this is 1st

    if '[' in sentences and ']' in sentences:
        print('\n\nValue replacement not completed!!! Current sentence: %s' % sentences)
        print('current da:')
        print(dialog_act)
        pattern = re.compile(r'(\[[^\[^\]]+\])')
        slots = pattern.findall(sentences)
        for slot in slots:
            sentences = sentences.replace(slot, ' ')
        print('after replace:', sentences)
        # raise Exception('\n\nValue replacement not completed!!! Current sentence: %s' % sentences)
    return sentences


def get_bleu4(dialog_acts, golden_utts, gen_utts, data_key):
    das2utts = {}
    for das, utt, gen in zip(dialog_acts, golden_utts, gen_utts):
        intent_frequency = defaultdict(int)
        for act in das:
            cur_act = copy.copy(act)

            # intent list
            facility = None  # for 酒店设施
            if '酒店设施' in cur_act[2]:
                facility = cur_act[2].split('-')[1]
                if cur_act[0] == 'Inform':
                    cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
                elif cur_act[0] == 'Request':
                    cur_act[2] = cur_act[2].split('-')[0]
            if cur_act[0] == 'Select':
                cur_act[2] = '源领域+' + cur_act[3]
            intent = '+'.join(cur_act[:-1])
            if '+'.join(cur_act) == 'Inform+景点+门票+免费' or cur_act[-1] == '无':
                intent = '+'.join(cur_act)

            intent_frequency[intent] += 1

            # utt content replacement
            if (act[0] in ['Inform', 'Recommend'] or '酒店设施' in intent) and not intent.endswith('无'):
                if act[3] in utt or (facility and facility in utt):
                    # value to be replaced
                    if '酒店设施' in intent:
                        value = facility
                    else:
                        value = act[3]

                    # placeholder
                    placeholder = '[' + intent + ']'
                    placeholder_one = '[' + intent + '1]'
                    placeholder_with_number = '[' + intent + str(intent_frequency[intent]) + ']'

                    if intent_frequency[intent] > 1:
                        utt = utt.replace(placeholder, placeholder_one)
                        utt = utt.replace(value, placeholder_with_number)

                        gen = gen.replace(placeholder, placeholder_one)
                        gen = gen.replace(value, placeholder_with_number)
                    else:
                        utt = utt.replace(value, placeholder)
                        gen = gen.replace(value, placeholder)

        hash_key = ''
        for act in sorted(das):
            hash_key += act2intent(act)
        das2utts.setdefault(hash_key, {'refs': [], 'gens': []})
        das2utts[hash_key]['refs'].append(utt)
        das2utts[hash_key]['gens'].append({
            'das': das,
            'gen': gen
        })

    refs, gens = [], []
    for das in das2utts.keys():
        assert len(das2utts[das]['refs']) == (len(das2utts[das]['gens']))
        for gen_pair in das2utts[das]['gens']:
            lex_das = gen_pair['das']  # das w/ value
            gen = gen_pair['gen']
            lex_gen = value_replace(gen, lex_das)
            gens.append([x for x in jieba.lcut(lex_gen) if x.strip()])
            refs.append(
                [[x for x in jieba.lcut(value_replace(s, lex_das)) if x.strip()] for s in das2utts[das]['refs']])

    with open(os.path.join('', 'generated_sens_%s.json' % data_key), 'w', encoding='utf-8') as f:
        json.dump({'refs': refs, 'gens': gens}, f, indent=4, sort_keys=True, ensure_ascii=False)
    print('generated_sens_%s.txt saved!' % data_key)

    print('Start calculating bleu score...')
    bleu = corpus_bleu(refs, gens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    return bleu


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:")
        print("\t python evaluate.py data_key")
        print("\t data_key=usr/sys")
        sys.exit()
    data_key = sys.argv[1]
    assert data_key == 'usr' or data_key == 'sys'

    model = TemplateNLG(is_user=(data_key=='usr'), mode='auto_manual')

    archive = zipfile.ZipFile('../../../../data/crosswoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))

    dialog_acts = []
    golden_utts = []
    gen_utts = []
    gen_slots = []
    intent_list = []
    dialog_acts2genutts = defaultdict(list)

    sen_num = 0
    sess_num = 0

    # if os.path.isfile('dialog_acts_%s.pkl' % data_key) and os.path.isfile(
    #         'golden_utts_%s.pkl' % data_key) and os.path.isfile('gen_utts_%s.pkl' % data_key):
    #     with open('dialog_acts_%s.pkl' % data_key, 'rb') as fda:
    #         dialog_acts = pkl.load(fda)
    #     with open('golden_utts_%s.pkl' % data_key, 'rb') as fgold:
    #         golden_utts = pkl.load(fgold)
    #     with open('gen_utts_%s.pkl' % data_key, 'rb') as fgen:
    #         gen_utts = pkl.load(fgen)
    #     for no, sess in list(test_data.items()):
    #         sess_num += 1
    #         for turn in sess['messages']:
    #             if turn['role'] == 'usr' and data_key == 'sys':
    #                 continue
    #             elif turn['role'] == 'sys' and data_key == 'usr':
    #                 continue
    #             sen_num += 1
    #     print('sen_num: ', sen_num)
    #     print('Loaded from existing data!')
    # else:
    if True:
        for no, sess in list(test_data.items()):
            sess_num += 1
            print('[%d/%d]' % (sess_num, len(test_data)))
            for turn in sess['messages']:
                if turn['role'] == 'usr' and data_key == 'sys':
                    continue
                elif turn['role'] == 'sys' and data_key == 'usr':
                    continue
                sen_num += 1

                dialog_acts.append(turn['dialog_act'])
                golden_utts.append(turn['content'])  # slots **values**
                gen_utt = model.generate(turn['dialog_act'])
                gen_utts.append(gen_utt)  # slots **values**

        with open('dialog_acts_%s.pkl' % data_key, 'wb') as fda:
            pkl.dump(dialog_acts, fda)
        with open('golden_utts_%s.pkl' % data_key, 'wb') as fgold:
            pkl.dump(golden_utts, fgold)
        with open('gen_utts_%s.pkl' % data_key, 'wb') as fgen:
            pkl.dump(gen_utts, fgen)

    bleu4 = get_bleu4(dialog_acts, golden_utts, gen_utts, data_key)
    print("Calculate bleu-4")
    print("BLEU-4: %.4f" % bleu4)
    print('Model on {} session {} sentences data_key={}'.format(len(test_data), sen_num, data_key))
