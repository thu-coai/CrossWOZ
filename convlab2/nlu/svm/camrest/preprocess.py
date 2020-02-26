"""
Preprocess camrest data for SVMNLU.

Usage:
    python preprocess [mode=all|usr|sys]
    mode: which side data will be use

Require:
    - ``../../../../data/camrest/[train|val|test].json.zip`` data file
    - ``../../../../data/camrest/db`` database dir

Output:
    - ``configs/ontology_camrest_[mode].json`` ontology file
    - ``data/[mode]_data/`` processed data dir
"""
import json
import os
import zipfile
from collections import Counter
import sys


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode=='all' or mode=='usr' or mode=='sys'
    data_dir = '../../../../data/camrest'
    db_dir = os.path.join(data_dir, 'db')

    processed_data_dir = 'data/{}_data'.format(mode)
    data_key = ['val', 'test', 'train']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir,key+'.json.zip'), key+'.json')
        print('load {}, size {}'.format(key, len(data[key])))

    db = json.load(open(os.path.join(db_dir,'CamRestDB.json')))

    db_slot2value = {x:[] for x in ['address', 'area', 'food', 'phone', 'pricerange', 'postcode', 'name']}
    for item in db:
        for key in item:
            if key in db_slot2value:
                db_slot2value[key].append(item[key])
    for k, v in db_slot2value.items():
        db_slot2value[k] = list(set(v))
    print('database:')
    print([(k,len(v)) for k,v in db_slot2value.items()])

    da2slot2value = {}
    for dialog in data['train']:
        for turn in dialog['dial']:
            if mode == 'usr' or mode == 'all':
                for da, svs in turn['usr']['dialog_act'].items():
                    da2slot2value.setdefault(da, {})
                    for s, v in svs:
                        da2slot2value[da].setdefault(s, [])
                        da2slot2value[da][s].append(v.lower())
            if mode == 'sys' or mode == 'all':
                for da, svs in turn['sys']['dialog_act'].items():
                    da2slot2value.setdefault(da, {})
                    for s, v in svs:
                        da2slot2value[da].setdefault(s, [])
                        da2slot2value[da][s].append(v.lower())

    for da in da2slot2value:
        for slot, values in da2slot2value[da].items():
            da2slot2value[da][slot] = list(set(values))

    requestable_slots = list(da2slot2value['request'].keys())
    informable_slots = list(da2slot2value['inform'].keys())
    # informable_onto = {s: db_slot2value[s] for s in informable_slots}
    informable_onto = da2slot2value['inform']
    informable_onto['none'] = ['none']
    slots_enumerated = ["area", "food", "pricerange", "none"]
    all_tuples = []
    for da in da2slot2value:
        if da == 'request':
            continue
        for slot, values in da2slot2value[da].items():
            if slot not in slots_enumerated:
                all_tuples.append((da, slot))
            else:
                for value in values:
                    if value in informable_onto[slot] or value.lower() in informable_onto[slot]:
                        all_tuples.append((da, slot, value))

    ontology_multiwoz = {
        "requestable": requestable_slots,
        "informable": informable_onto,
        "all_tuples": all_tuples
    }
    json.dump(ontology_multiwoz, open('configs/ontology_camrest_{}.json'.format(mode), 'w'), indent=4)

    for d_key, d in data.items():
        d_dir = os.path.join(processed_data_dir, d_key)
        filelist = []
        if not os.path.exists(d_dir):
            os.makedirs(d_dir)
        for dialog in d:
            no = dialog['dialogue_id']
            label_json = {"session-id": no}
            label_turns = []
            log_json = {"session-id": no}
            log_turns = []
            for turn in dialog['dial']:
                if mode == 'usr' or mode == 'all':
                    text = turn['usr']['transcript']
                    log_turn = {'input': {'live': {'asr-hyps': [{'asr-hyp': text, 'score': 0}]}}}
                    log_turns.append(log_turn)

                    new_das = []
                    for da, svs in turn['usr']['dialog_act'].items():
                        for s, v in svs:
                            if da == 'request':
                                new_das.append({'act': da, 'slots': [['slot', s]]})
                            else:\
                                new_das.append({'act': da, 'slots': [[s, v.lower()]]})
                    label_turns.append({'semantics': {'json': new_das}})
                if mode == 'sys' or mode == 'all':
                    text = turn['sys']['sent']
                    log_turn = {'input': {'live': {'asr-hyps': [{'asr-hyp': text, 'score': 0}]}}}
                    log_turns.append(log_turn)

                    new_das = []
                    for da, svs in turn['sys']['dialog_act'].items():
                        for s, v in svs:
                            if da == 'request':
                                new_das.append({'act': da, 'slots': [['slot', s]]})
                            else:
                                new_das.append({'act': da, 'slots': [[s, v.lower()]]})
                    label_turns.append({'semantics': {'json': new_das}})
            label_json['turns'] = label_turns
            log_json['turns'] = log_turns
            f_dir = os.path.join(d_dir, str(no))
            filelist.append(str(no))
            if not os.path.exists(f_dir):
                os.makedirs(f_dir)

            json.dump(label_json, open(os.path.join(f_dir, 'label.json'), 'w'), indent=4)
            json.dump(log_json, open(os.path.join(f_dir, 'log.json'), 'w'), indent=4)
        f = open('configs/{}ListFile'.format(d_key), 'w')
        f.writelines([x+'\n' for x in filelist])
        f.close()
