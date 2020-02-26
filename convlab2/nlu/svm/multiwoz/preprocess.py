"""
Preprocess multiwoz data for SVMNLU.

Usage:
    python preprocess [mode=all|usr|sys]
    mode: which side data will be use

Require:
    - ``../../../../data/multiwoz/[train|val|test].json.zip`` data file
    - ``../../../../data/multiwoz/db`` database dir

Output:
    - ``configs/ontology_multiwoz_[mode].json`` ontology file
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
    data_dir = '../../../../data/multiwoz'
    db_dir = os.path.join(data_dir, 'db')

    processed_data_dir = 'data/{}_data'.format(mode)
    data_key = ['val', 'test', 'train']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir,key+'.json.zip'), key+'.json')
        print('load {}, size {}'.format(key, len(data[key])))

    db = {
        'attraction': json.load(open(os.path.join(db_dir,'attraction_db.json'))),
        'hotel': json.load(open(os.path.join(db_dir,'hotel_db.json'))),
        'restaurant': json.load(open(os.path.join(db_dir,'restaurant_db.json'))),
        'police': json.load(open(os.path.join(db_dir,'police_db.json'))),
        'hospital': json.load(open(os.path.join(db_dir,'hospital_db.json'))),
        'taxi': json.load(open(os.path.join(db_dir,'taxi_db.json'))),
        'train': json.load(open(os.path.join(db_dir,'train_db.json')))
    }
    domain2slot2value = {}
    for domain in db.keys():
        domain2slot2value[domain] = {}
        if domain == 'taxi':
            continue
        for item in db[domain]:
            for s, v in item.items():
                if isinstance(v, type(u'')):
                    domain2slot2value[domain].setdefault(s, Counter())
                    domain2slot2value[domain][s] += Counter([v])
                else:
                    domain2slot2value[domain].setdefault(s, [])
                    domain2slot2value[domain][s].append(v)

    requestable_slots = []
    informable_slots = []
    for no, sess in data['train'].items():
        for i, turn in enumerate(sess['log']):
            if mode == 'usr' and i % 2 == 1:
                continue
            elif mode == 'sys' and i % 2 == 0:
                continue
            for da, svs in turn['dialog_act'].items():
                if 'Request' in da:
                    requestable_slots.extend([da.split('-')[0] + '-' + s for s, v in svs])
                else:
                    informable_slots.extend([s for s, v in svs])
    requestable_slots = list(set(requestable_slots))
    informable_slots = list(set(informable_slots))


    def slot2all_value(slot):
        all_value = []
        for domain in domain2slot2value.keys():
            if slot in domain2slot2value[domain]:
                all_value.extend(list(domain2slot2value[domain][slot]))
        return list(set(all_value))


    informable_onto = {}
    informable_onto['Fee'] = slot2all_value('entrance fee')
    informable_onto['Addr'] = slot2all_value('address')
    informable_onto['Area'] = slot2all_value('area')
    informable_onto['Stars'] = slot2all_value('stars') + ['zero', 'one', 'two', 'three', 'four', 'five']
    informable_onto['Internet'] = slot2all_value('internet')
    informable_onto['Department'] = slot2all_value('department')
    informable_onto['Stay'] = list([str(i) for i in range(10)]) + ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                                                                   'seven', 'eight', 'nine', 'ten']
    informable_onto['Ref'] = []
    informable_onto['Food'] = slot2all_value('food')
    informable_onto['Type'] = slot2all_value('type')
    informable_onto['Price'] = slot2all_value('pricerange')
    informable_onto['Choice'] = list([str(i) for i in range(20)]) + ['zero', 'one', 'two', 'three', 'four', 'five',
                                                                     'six', 'seven', 'eight', 'nine', 'ten']
    informable_onto['Phone'] = []
    informable_onto['Ticket'] = list(domain2slot2value['train']['price'])
    informable_onto['Day'] = slot2all_value('day')
    informable_onto['Name'] = slot2all_value('name')
    informable_onto['Car'] = [i + ' ' + j for i in ["black", "white", "red", "yellow", "blue", "grey"] for j in
                              ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen",
                               "tesla"]]
    informable_onto['Leave'] = []
    informable_onto['Time'] = slot2all_value('Duration')
    informable_onto['Arrive'] = []
    informable_onto['Post'] = slot2all_value('postcode')
    informable_onto['none'] = ['none']
    informable_onto['Depart'] = slot2all_value('departure')
    informable_onto['People'] = list([str(i) for i in range(10)]) + ['zero', 'one', 'two', 'three', 'four', 'five',
                                                                     'six', 'seven', 'eight', 'nine', 'ten']
    informable_onto['Dest'] = slot2all_value('destination')
    informable_onto['Parking'] = slot2all_value('parking')
    informable_onto['Id'] = slot2all_value('trainID')
    # remove `Open`

    da2slot2value = {}
    for d_key, d in data.items():
        d_dir = os.path.join(processed_data_dir, d_key)
        if not os.path.exists(d_dir):
            os.makedirs(d_dir)
        for no, sess in d.items():
            label_json = {"session-id": no}
            label_turns = []
            log_json = {"session-id": no}
            log_turns = []
            for i, turn in enumerate(sess['log']):
                if mode == 'usr' and i % 2 == 1:
                    continue
                elif mode == 'sys' and i % 2 == 0:
                    continue
                new_das = []
                for da, svs in turn['dialog_act'].items():
                    for s, v in svs:
                        if s == 'Open':
                            continue
                        if 'Request' in da:
                            domain, act = da.split('-')
                            new_das.append({'act': 'request', 'slots': [['slot', domain + '-' + s]]})
                        else:
                            new_das.append({'act': da, 'slots': [[s, v.lower()]]})
                        da2slot2value.setdefault(da, {})
                        da2slot2value[da].setdefault(s, [])
                        da2slot2value[da][s].append(v)
                label_turns.append({'semantics': {'json': new_das}})
                log_turn = {'input': {'live': {'asr-hyps': [{'asr-hyp': turn['text'], 'score': 0}]}}}
                log_turns.append(log_turn)
            label_json['turns'] = label_turns
            log_json['turns'] = log_turns
            f_dir = os.path.join(d_dir,no)
            if not os.path.exists(f_dir):
                os.makedirs(f_dir)

            json.dump(label_json, open(os.path.join(f_dir,'label.json'), 'w'), indent=4)
            json.dump(log_json, open(os.path.join(f_dir,'log.json'), 'w'), indent=4)

    all_tuples = []
    slots_enumerated = ["Area", "Type", "Price", "Day", "Internet", "none", "Parking"]
    for da, sv in da2slot2value.items():
        if 'Request' in da:
            pass
        else:
            for s, v in sv.items():
                v_cnt = Counter(v)
                if s not in slots_enumerated:
                    all_tuples.append((da, s))
                else:
                    for i, c in dict(v_cnt).items():
                        if c > 0 and i in informable_onto[s] or i.lower() in informable_onto[s]:
                            all_tuples.append((da, s, i))

    ontology_multiwoz = {
        "requestable": requestable_slots,
        "informable": informable_onto,
        "all_tuples": all_tuples
    }
    json.dump(ontology_multiwoz, open('configs/ontology_multiwoz_{}.json'.format(mode), 'w'), indent=4)
