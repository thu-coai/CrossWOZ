#!/usr/bin/env python
# coding: utf-8

# # Generate training data

import os
import json
import jieba
import re
from pprint import pprint
from collections import defaultdict
from copy import copy
import random
import zipfile
import functools


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def cmp_intent(intent1: str, intent2: str):
    assert role in ['sys', 'usr']
    intent_order = {
        'usr': (
            'General+greet+none',
            'Inform+出租+出发地',
            'Inform+出租+目的地',
            'Inform+地铁+出发地',
            'Inform+地铁+目的地',
            'Inform+景点+名称',
            'Inform+景点+游玩时间',
            'Inform+景点+评分',
            'Inform+景点+门票',
            'Inform+景点+门票+免费',
            'Inform+酒店+价格',
            'Inform+酒店+名称',
            'Inform+酒店+评分',
            'Inform+酒店+酒店类型',
            'Inform+酒店+酒店设施+否',
            'Inform+酒店+酒店设施+是',
            'Inform+餐馆+人均消费',
            'Inform+餐馆+名称',
            'Inform+餐馆+推荐菜',
            'Inform+餐馆+推荐菜1+推荐菜2',
            'Inform+餐馆+评分',
            'Select+景点+源领域+景点',
            'Select+景点+源领域+酒店',
            'Select+景点+源领域+餐馆',
            'Select+酒店+源领域+景点',
            'Select+酒店+源领域+餐馆',
            'Select+餐馆+源领域+景点',
            'Select+餐馆+源领域+酒店',
            'Select+餐馆+源领域+餐馆',
            'Request+出租+车型',
            'Request+出租+车牌',
            'Request+地铁+出发地附近地铁站',
            'Request+地铁+目的地附近地铁站',
            'Request+景点+名称',
            'Request+景点+周边景点',
            'Request+景点+周边酒店',
            'Request+景点+周边餐馆',
            'Request+景点+地址',
            'Request+景点+游玩时间',
            'Request+景点+电话',
            'Request+景点+评分',
            'Request+景点+门票',
            'Request+酒店+价格',
            'Request+酒店+名称',
            'Request+酒店+周边景点',
            'Request+酒店+周边餐馆',
            'Request+酒店+地址',
            'Request+酒店+电话',
            'Request+酒店+评分',
            'Request+酒店+酒店类型',
            'Request+酒店+酒店设施',
            'Request+餐馆+人均消费',
            'Request+餐馆+名称',
            'Request+餐馆+周边景点',
            'Request+餐馆+周边酒店',
            'Request+餐馆+周边餐馆',
            'Request+餐馆+地址',
            'Request+餐馆+推荐菜',
            'Request+餐馆+电话',
            'Request+餐馆+营业时间',
            'Request+餐馆+评分',
            'General+thank+none',
            'General+bye+none'
        ),
        'sys': (
            'General+greet+none',
            'General+thank+none',
            'General+welcome+none',
            'NoOffer+景点+none',
            'NoOffer+酒店+none',
            'NoOffer+餐馆+none',
            'Inform+主体+属性+无',
            'Inform+出租+车型',
            'Inform+出租+车牌',
            'Inform+地铁+出发地附近地铁站',
            'Inform+地铁+目的地附近地铁站',
            'Inform+景点+名称',
            'Inform+景点+周边景点',
            'Inform+景点+周边景点1+周边景点2',
            'Inform+景点+周边景点1+周边景点2+周边景点3',
            'Inform+景点+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+景点+周边酒店',
            'Inform+景点+周边酒店1+周边酒店2',
            'Inform+景点+周边酒店1+周边酒店2+周边酒店3',
            'Inform+景点+周边酒店1+周边酒店2+周边酒店3+周边酒店4',
            'Inform+景点+周边餐馆',
            'Inform+景点+周边餐馆1+周边餐馆2',
            'Inform+景点+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+景点+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+景点+地址',
            'Inform+景点+游玩时间',
            'Inform+景点+电话',
            'Inform+景点+评分',
            'Inform+景点+门票',
            'Inform+景点+门票+免费',
            'Inform+酒店+价格',
            'Inform+酒店+名称',
            'Inform+酒店+周边景点',
            'Inform+酒店+周边景点1+周边景点2',
            'Inform+酒店+周边景点1+周边景点2+周边景点3',
            'Inform+酒店+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+酒店+周边餐馆',
            'Inform+酒店+周边餐馆1+周边餐馆2',
            'Inform+酒店+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+酒店+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+酒店+地址',
            'Inform+酒店+电话',
            'Inform+酒店+评分',
            'Inform+酒店+酒店类型',
            'Inform+酒店+酒店设施+否',
            'Inform+酒店+酒店设施+是',
            'Inform+餐馆+人均消费',
            'Inform+餐馆+名称',
            'Inform+餐馆+周边景点',
            'Inform+餐馆+周边景点1+周边景点2',
            'Inform+餐馆+周边景点1+周边景点2+周边景点3',
            'Inform+餐馆+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+餐馆+周边酒店',
            'Inform+餐馆+周边酒店1+周边酒店2',
            'Inform+餐馆+周边酒店1+周边酒店2+周边酒店3',
            'Inform+餐馆+周边酒店1+周边酒店2+周边酒店3+周边酒店4',
            'Inform+餐馆+周边餐馆',
            'Inform+餐馆+周边餐馆1+周边餐馆2',
            'Inform+餐馆+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+餐馆+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+餐馆+地址',
            'Inform+餐馆+推荐菜',
            'Inform+餐馆+推荐菜1+推荐菜2',
            'Inform+餐馆+推荐菜1+推荐菜2+推荐菜3',
            'Inform+餐馆+推荐菜1+推荐菜2+推荐菜3+推荐菜4',
            'Inform+餐馆+电话',
            'Inform+餐馆+营业时间',
            'Inform+餐馆+评分',
            'Recommend+景点+名称',
            'Recommend+景点+名称1+名称2',
            'Recommend+景点+名称1+名称2+名称3',
            'Recommend+景点+名称1+名称2+名称3+名称4',
            'Recommend+酒店+名称',
            'Recommend+酒店+名称1+名称2',
            'Recommend+酒店+名称1+名称2+名称3',
            'Recommend+酒店+名称1+名称2+名称3+名称4',
            'Recommend+餐馆+名称',
            'Recommend+餐馆+名称1+名称2',
            'Recommend+餐馆+名称1+名称2+名称3',
            'Recommend+餐馆+名称1+名称2+名称3+名称4',
            'General+reqmore+none',
            'General+bye+none'
        )
    }
    intent1 = intent1.split('1')[0]
    intent2 = intent2.split('1')[0]
    if 'Inform' in intent1 and '无' in intent1:
        intent1 = 'Inform+主体+属性+无'
    if 'Inform' in intent2 and '无' in intent2:
        intent2 = 'Inform+主体+属性+无'
    try:
        assert intent1 in intent_order[role] and intent2 in intent_order[role]
    except AssertionError:
        print(role, intent1, intent2)
    return intent_order[role].index(intent1) - intent_order[role].index(intent2)


data_dir = '../../../../data/crosswoz'
train_archive = zipfile.ZipFile(os.path.join(data_dir, 'train.json.zip'), 'r')
train_data = json.load(train_archive.open('train.json'))
valid_archive = zipfile.ZipFile(os.path.join(data_dir, 'val.json.zip'), 'r')
valid_data = json.load(valid_archive.open('val.json'))
test_archive = zipfile.ZipFile(os.path.join(data_dir, 'test.json.zip'), 'r')
test_data = json.load(test_archive.open('test.json'))

data = {'train': train_data, 'valid': valid_data, 'test': test_data}

print("Length of train_data:", len(train_data))

# ## For system/user

user_multi_intent_dict = defaultdict(list)
sys_multi_intent_dict = defaultdict(list)

role = None
dialogue_id = 1
for dialogue in train_data.values():
    # print('Processing the %dth dialogue' % dialogue_id)
    dialogue_id += 1
    for round in dialogue['messages']:
        # original content
        content = round['content']
        intent_list = []
        intent_frequency = defaultdict(int)
        role = round['role']
        usable = True
        for act in round['dialog_act']:
            cur_act = copy(act)

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
            intent_list.append(intent)

            # content replacement
            if (act[0] in ['Inform', 'Recommend'] or '酒店设施' in intent) and not intent.endswith('无'):
                if act[3] in content or (facility and facility in content):
                    intent_frequency[intent] += 1

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
                        content = content.replace(placeholder, placeholder_one)
                        content = content.replace(value, placeholder_with_number)
                    else:
                        content = content.replace(value, placeholder)
                else:
                    usable = False

        # multi-intent name
        try:
            intent_list = sorted(intent_list, key=functools.cmp_to_key(cmp_intent))
        except:
            print(round['content'])
        multi_intent = '*'.join(intent_list)
        if usable:
            if round['role'] == 'usr':
                user_multi_intent_dict[multi_intent].append(content)
            else:
                sys_multi_intent_dict[multi_intent].append(content)


output_data_dir = 'resource'
for require_role in ['sys', 'usr']:
    print('\nProcessing %s data...' % require_role)
    if require_role == 'usr':
        output_data_dir += '_usr'
    if not os.path.exists(output_data_dir):
        os.mkdir(output_data_dir)
    if require_role == 'usr':
        template_data = user_multi_intent_dict.copy()
    else:
        template_data = sys_multi_intent_dict.copy()
    print('Number of intents in templates:', len(template_data))

    sens = []
    for key, ls in template_data.items():
        if key:
            sens += ls
    print('Number of sentences in templates:', len(sens))

    # ### vocab.txt

    vocab_dict = defaultdict(int)
    pattern = re.compile(r'(\[[^\[^\]]+\])')
    for ls in template_data.values():
        for sen in ls:
            slots = pattern.findall(sen)
            for slot in slots:
                vocab_dict[slot] += 1
                sen = sen.replace(slot, '')
            for word in jieba.lcut(sen):
                vocab_dict[word] += 1
    len(vocab_dict)

    vocab_dict = {word: frequency for word, frequency in vocab_dict.items() if vocab_dict[word] > 3}
    len(vocab_dict)

    with open(os.path.join(output_data_dir, 'vocab.txt'), 'w', encoding='utf-8') as fvocab:
        fvocab.write('PAD_token\nSOS_token\nEOS_token\nUNK_token\n')
        for key, value in sorted(vocab_dict.items(), key=lambda x: int(x[1])):
            if key.strip():
                fvocab.write(key + '\t' + str(value) + '\n')

    # ### text.json, feat.json

    def split_delex_sentence(sen):
        ori_sen = copy(sen)
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

    dialogue_id = 0
    text_dict = {'train': defaultdict(dict), 'valid': defaultdict(dict), 'test': defaultdict(dict)}
    all_text_dict = defaultdict(dict)
    feat_dict = defaultdict(dict)
    template_list = []
    unk_sen_num = []

    for split in ['train', 'valid', 'test']:
        for idx, dialogue in data[split].items():
            dialogue_id += 1
            # if (dialogue_id % 500 == 0):
            #     print('Processing the %dth dialogue' % dialogue_id)
            round_id = 0
            for round in dialogue['messages']:
                # original content
                content = round['content']
                ori_content = content
                intent_list = []
                intent_frequency = defaultdict(int)
                role = round['role']

                # now we consider the system/user:
                if role != require_role:
                    continue
                round_id += 1

                # usable = True
                for act in round['dialog_act']:
                    cur_act = copy(act)

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
                    intent_list.append(intent)

                    intent_frequency[intent] += 1

                    # content replacement
                    value = 'none'
                    freq = 'none'
                    if (act[0] in ['Inform', 'Recommend'] or '酒店设施' in intent) and not intent.endswith('无'):
                        if act[3] in content or (facility and facility in content):
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
                                content = content.replace(placeholder, placeholder_one)
                                content = content.replace(value, placeholder_with_number)
                            else:
                                content = content.replace(value, placeholder)

                        freq = str(intent_frequency[intent])
                    elif act[0] == 'Request':
                        freq = '?'
                        value = '?'
                    elif act[0] == 'Select':
                        value = act[3]
                    # save to feat.json
                    new_act = intent.split('+')
                    if new_act[0] == 'General':
                        feat_key = new_act[0] + '-' + new_act[1]
                    else:
                        feat_key = new_act[1] + '-' + new_act[0]
                    if new_act[2] == '酒店设施' and new_act[0] == 'Inform':
                        try:
                            feat_value = [new_act[2] + '+' + new_act[3], freq, value]
                        except:
                            print(new_act)
                    elif intent.endswith('无'):
                        feat_value = [new_act[2] + '+无', freq, value]
                    elif intent.endswith('免费'):
                        feat_value = [new_act[2] + '+免费', freq, value]
                    else:
                        feat_value = [new_act[2], freq, value]

                    feat_dict[idx][round_id] = feat_dict[idx].get(round_id, dict())
                    feat_dict[idx][round_id][feat_key] = feat_dict[idx][round_id].get(feat_key, [])
                    feat_dict[idx][round_id][feat_key].append(feat_value)

                # save to text.json
                split_delex = split_delex_sentence(content)
                unk_sen = [word if word in vocab_dict else 'UNK_token' for word in
                       re.split(r'\s+', split_delex) if word]
                unk_sen_num.append('UNK_token' in unk_sen)
                text_dict[split][idx][round_id] = {
                    "delex": split_delex,
                    "ori": ori_content
                }
                all_text_dict[idx][round_id] = {
                    "delex": split_delex,
                    "ori": ori_content
                }

    print('unk sen ratio', sum(unk_sen_num)*1.0/len(unk_sen_num))

    with open(os.path.join(output_data_dir, 'text.json'), 'w', encoding='utf-8') as f:
        json.dump(all_text_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    with open(os.path.join(output_data_dir, 'feat.json'), 'w', encoding='utf-8') as f:
        json.dump(feat_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    # ### template.txt

    template_set = set()
    for dialogue in feat_dict.values():
        for r in dialogue.values():
            for k, ls in r.items():
                template_set.add('d:' + k.split('-')[0])
                template_set.add('d-a:' + k)
                for v in ls:
                    template_set.add('d-a-s-v:' + k + '-' + v[0] + '-' + str(v[1]))
    with open(os.path.join(output_data_dir, 'template.txt'), 'w', encoding='utf-8') as ftem:
        ftem.write('\n'.join(sorted(list(template_set), reverse=True)))

    # ### split data

    split_dict = {'valid': [], 'test': [], 'train': []}
    for split in split_dict.keys():
        all_candidates = []
        for d_id, sens in text_dict[split].items():
            for i in range(len(sens)):
                all_candidates.append([d_id, str(i + 1), "-"])
        split_dict[split] = copy(all_candidates)

    with open(os.path.join(output_data_dir, 'split.json'), 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
