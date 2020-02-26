import csv
import json
import logging
import os
import pickle
import random
import re

import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from convlab2.e2e.sequicity.config import global_config as cfg


def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        idx = s[sidx:].find(r)
        if idx == -1:
            return s, -1
        idx += sidx
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
    return s


class _ReaderBase:
    class LabelSet:
        def __init__(self):
            self._idx2item = {}
            self._item2idx = {}
            self._freq_dict = {}

        def __len__(self):
            return len(self._idx2item)

        def _absolute_add_item(self, item):
            idx = len(self)
            self._idx2item[idx] = item
            self._item2idx[item] = idx

        def add_item(self, item):
            if item not in self._freq_dict:
                self._freq_dict[item] = 0
            self._freq_dict[item] += 1

        def construct(self, limit):
            l = sorted(self._freq_dict.keys(), key=lambda x: -self._freq_dict[x])
            print('Actual label size %d' % (len(l) + len(self._idx2item)))
            if len(l) + len(self._idx2item) < limit:
                logging.warning('actual label set smaller than that configured: {}/{}'
                                .format(len(l) + len(self._idx2item), limit))
            for item in l:
                if item not in self._item2idx:
                    idx = len(self._idx2item)
                    self._idx2item[idx] = item
                    self._item2idx[item] = idx
                    if len(self._idx2item) >= limit:
                        break

        def encode(self, item):
            return self._item2idx[item]

        def decode(self, idx):
            return self._idx2item[idx]

    class Vocab(LabelSet):
        def __init__(self, init=True):
            _ReaderBase.LabelSet.__init__(self)
            if init:
                self._absolute_add_item('<pad>')  # 0
                self._absolute_add_item('<go>')  # 1
                self._absolute_add_item('<unk>')  # 2
                self._absolute_add_item('<go2>')  # 3

        def load_vocab(self, vocab_path):
            f = open(vocab_path, 'rb')
            dic = pickle.load(f)
            self._idx2item = dic['idx2item']
            self._item2idx = dic['item2idx']
            self._freq_dict = dic['freq_dict']
            f.close()

        def save_vocab(self, vocab_path):
            f = open(vocab_path, 'wb')
            dic = {
                'idx2item': self._idx2item,
                'item2idx': self._item2idx,
                'freq_dict': self._freq_dict
            }
            pickle.dump(dic, f)
            f.close()

        def sentence_encode(self, word_list):
            return [self.encode(_) for _ in word_list]

        def sentence_decode(self, index_list, eos=None):
            l = [self.decode(_) for _ in index_list]
            if not eos or eos not in l:
                return ' '.join(l)
            else:
                idx = l.index(eos)
                return ' '.join(l[:idx])

        def nl_decode(self, l, eos=None):
            return [self.sentence_decode(_, eos) + '\n' for _ in l]

        def encode(self, item):
            if item in self._item2idx:
                return self._item2idx[item]
            else:
                return self._item2idx['<unk>']

        def decode(self, idx):
            idx = np.int(idx)
            if idx < len(self):
                return self._idx2item[idx]
            else:
                return 'ITEM_%d' % (idx - cfg.vocab_size)

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = self.Vocab()
        self.result_file = ''

    def _construct(self, *args):
        """
        load data, construct vocab and store them in self.train/dev/test
        :param args:
        :return:
        """
        raise NotImplementedError('This is an abstract class, bro')

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return turn_bucket

    def _mark_batch_as_supervised(self, all_batches):
        supervised_num = int(len(all_batches) * cfg.spv_proportion / 100)
        for i, batch in enumerate(all_batches):
            for dial in batch:
                for turn in dial:
                    turn['supervised'] = i < supervised_num
                    if not turn['supervised']:
                        turn['degree'] = [0.] * cfg.degree_size  # unsupervised learning. DB degree should be unknown
        return all_batches

    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            if len(batch) == cfg.batch_size:
                all_batches.append(batch)
                batch = []
        # if remainder > 1/2 batch_size, just put them in the previous batch, otherwise form a new batch
        if len(batch) > 0.5 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def _transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def mini_batch_iterator(self, set_name):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            batches = self._construct_mini_batch(turn_bucket[k])
            all_batches += batches
        self._mark_batch_as_supervised(all_batches)
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self._transpose_batch(batch)

    def wrap_result(self, turn_batch, gen_m, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """

        results = []
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2'}
        batch_size = len(turn_batch['user'])
        for i in range(batch_size):
            entry = {}
            if prev_z is not None:
                src = prev_z[i] + turn_batch['user'][i]
            else:
                src = turn_batch['user'][i]
            for key in turn_batch:
                entry[key] = turn_batch[key][i]
                if key in eos_syntax:
                    entry[key] = self.vocab.sentence_decode(entry[key], eos=eos_syntax[key])
            if gen_m:
                entry['generated_response'] = self.vocab.sentence_decode(gen_m[i], eos='EOS_M')
            else:
                entry['generated_response'] = ''
            if gen_z:
                entry['generated_bspan'] = self.vocab.sentence_decode(gen_z[i], eos='EOS_Z2')
            else:
                entry['generated_bspan'] = ''
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(cfg.result_path, 'w')
            self.result_file.write(str(cfg))
            write_header = True

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results

    def db_search(self, constraints):
        raise NotImplementedError('This is an abstract method')

    def db_degree_handler(self, z_samples, *args, **kwargs):
        """
        returns degree of database searching and it may be used to control further decoding.
        One hot vector, indicating the number of entries found: [0, 1, 2, 3, 4, >=5]
        :param z_samples: nested list of B * [T]
        :return: an one-hot control *numpy* control vector
        """
        control_vec = []

        for cons_idx_list in z_samples:
            constraints = set()
            for cons in cons_idx_list:
                if not isinstance(cons, str):
                    cons = self.vocab.decode(cons)
                if cons == 'EOS_Z1':
                    break
                constraints.add(cons)
            match_result = self.db_search(constraints)
            degree = len(match_result)
            # modified
            # degree = 0
            control_vec.append(self._degree_vec_mapping(degree))
        return np.array(control_vec)

    def _degree_vec_mapping(self, match_num):
        l = [0.] * cfg.degree_size
        l[min(cfg.degree_size - 1, match_num)] = 1.
        return l


class CamRest676Reader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.data, cfg.db)
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        tokenized_data = []
        vk_map = self._value_key_map(db_data)
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                requested = []
                for slot in turn['usr']['slu']:
                    if slot['act'] == 'inform':
                        s = slot['slots'][0][1]
                        if s not in ['dontcare', 'none']:
                            constraint.extend(word_tokenize(s))
                    else:
                        requested.extend(word_tokenize(slot['slots'][0][1]))
                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                constraint.append('EOS_Z1')
                requested.append('EOS_Z2')
                user = word_tokenize(turn['usr']['transcript']) + ['EOS_U']
                response = word_tokenize(self._replace_entity(turn['sys']['sent'], vk_map, constraint)) + ['EOS_M']
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'constraint': constraint,
                    'requested': requested,
                    'degree': degree,
                })
                if construct_vocab:
                    for word in user + response + constraint + requested:
                        self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def _replace_entity(self, response, vk_map, constraint):
        response = re.sub('[cC][., ]*[bB][., ]*\d[., ]*\d[., ]*\w[., ]*\w', 'postcode_SLOT', response)
        response = re.sub('\d{5}\s?\d{6}', 'phone_SLOT', response)
        constraint_str = ' '.join(constraint)
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 \
                    or (start_idx != 0 and response[start_idx - 1] != ' ') \
                    or (v in constraint_str):
                continue
            if k not in ['name', 'address']:
                response = clean_replace(response, v, k + '_SLOT', forward=True, backward=False)
            else:
                response = clean_replace(response, v, k + '_SLOT', forward=False, backward=False)
        return response

    def _value_key_map(self, db_data):
        requestable_keys = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_keys:
                    value_key[v] = k
        return value_key

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                constraint = self.vocab.sentence_encode(turn['constraint'])
                requested = self.vocab.sentence_encode(turn['requested'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']

                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': prev_response + user,
                    'response': response,
                    'bspan': constraint + requested,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def _construct(self, data_json_path, db_json_path):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        raw_data_json = open(data_json_path)
        raw_data = json.loads(raw_data_json.read().lower())
        db_json = open(db_json_path)
        db_data = json.loads(db_json.read().lower())
        self.db = db_data
        tokenized_data = self._get_tokenized_data(raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        encoded_data = self._get_encoded_data(tokenized_data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        raw_data_json.close()
        db_json.close()

    def db_search(self, constraints):
        match_results = []
        for entry in self.db:
            entry_values = ' '.join(entry.values())
            match = True
            for c in constraints:
                if c not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results


class KvretReader(_ReaderBase):
    def __init__(self):
        super().__init__()

        self.entity_dict = {}
        self.abbr_dict = {}

        self.wn = WordNetLemmatizer()
        self.db = {}

        self.tokenized_data_path = './data/kvret/'
        self._construct(cfg.train, cfg.dev, cfg.test, cfg.entity)

    def _construct(self, train_json_path, dev_json_path, test_json_path, entity_json_path):
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        train_json, dev_json, test_json = open(train_json_path), open(dev_json_path), open(test_json_path)
        entity_json = open(entity_json_path)
        train_data, dev_data, test_data = json.loads(train_json.read().lower()), json.loads(dev_json.read().lower()), \
                                          json.loads(test_json.read().lower())
        entity_data = json.loads(entity_json.read().lower())
        self._get_entity_dict(entity_data)

        tokenized_train = self._get_tokenized_data(train_data, construct_vocab, 'train')
        tokenized_dev = self._get_tokenized_data(dev_data, construct_vocab, 'dev')
        tokenized_test = self._get_tokenized_data(test_data, construct_vocab, 'test')

        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)

        self.train, self.dev, self.test = map(self._get_encoded_data, [tokenized_train, tokenized_dev,
                                                                       tokenized_test])
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
		
        train_json.close()
        dev_json.close()
        test_json.close()
        entity_json.close()

    def _save_tokenized_data(self, data, filename):
        path = self.tokenized_data_path + filename + '.tokenized.json'
        f = open(path,'w')
        json.dump(data,f,indent=2)
        f.close()

    def _load_tokenized_data(self, filename):
        '''
        path = self.tokenized_data_path + filename + '.tokenized.json'
        try:
            f = open(path,'r')
        except FileNotFoundError:
            return None
        data = json.load(f)
        f.close()
        return data
        '''
        return None

    def _tokenize(self, sent):
        return ' '.join(word_tokenize(sent))

    def _lemmatize(self, sent):
        return ' '.join([self.wn.lemmatize(_) for _ in sent.split()])

    def _replace_entity(self, response, vk_map, prev_user_input, intent):
        response = re.sub('\d+-?\d*fs?', 'temperature_SLOT', response)
        response = re.sub('\d+\s?miles?', 'distance_SLOT', response)
        response = re.sub('\d+\s\w+\s(dr)?(ct)?(rd)?(road)?(st)?(ave)?(way)?(pl)?\w*[.]?', 'address_SLOT', response)
        response = self._lemmatize(self._tokenize(response))
        requestable = {
            'weather': ['weather_attribute'],
            'navigate': ['poi', 'traffic_info', 'address', 'distance'],
            'schedule': ['event', 'date', 'time', 'party', 'agenda', 'room']
        }
        reqs = set()
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 or k not in requestable[intent]:
                continue
            end_idx = start_idx + len(v)
            while end_idx < len(response) and response[end_idx] != ' ':
                end_idx += 1
            # test whether they are indeed the same word
            lm1, lm2 = v.replace('.', '').replace(' ', '').replace("'", ''), \
                       response[start_idx:end_idx].replace('.', '').replace(' ', '').replace("'", '')
            if lm1 == lm2 and lm1 not in prev_user_input and v not in prev_user_input:
                response = clean_replace(response, response[start_idx:end_idx], k + '_SLOT')
                reqs.add(k)
        return response,reqs

    def _clean_constraint_dict(self, constraint_dict, intent, prefer='short'):
        """
        clean the constraint dict so that every key is in "informable" and similar to one in provided entity dict.
        :param constraint_dict:
        :return:
        """
        informable = {
            'weather': ['date', 'location', 'weather_attribute'],
            'navigate': ['poi_type', 'distance'],
            'schedule': ['event', 'date', 'time', 'agenda', 'party', 'room']
        }

        del_key = set(constraint_dict.keys()).difference(informable[intent])
        for key in del_key:
            constraint_dict.pop(key)
        invalid_key = []
        for k in constraint_dict:
            constraint_dict[k] = constraint_dict[k].strip()
            v = self._lemmatize(self._tokenize(constraint_dict[k]))
            v = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), v)
            v = re.sub('(\d+)\s?(mile)s?', lambda x: x.group(1) + ' ' + x.group(2), v)
            if v in self.entity_dict:
                if prefer == 'short':
                    constraint_dict[k] = v
                elif prefer == 'long':
                    constraint_dict[k] = self.abbr_dict.get(v, v)
            elif v.split()[0] in self.entity_dict:
                if prefer == 'short':
                    constraint_dict[k] = v.split()[0]
                elif prefer == 'long':
                    constraint_dict[k] = self.abbr_dict.get(v.split()[0], v)
            else:
                invalid_key.append(k)
        for key in invalid_key:
            constraint_dict.pop(key)
        return constraint_dict

    def _get_tokenized_data(self, raw_data, add_to_vocab, data_type, is_test=False):
        """
        Somerrthing to note: We define requestable and informable slots as below in further experiments
        (including other baselines):

        informable = {
            'weather': ['date','location','weather_attribute'],
            'navigate': ['poi_type','distance'],
            'schedule': ['event']
        }

        requestable = {
            'weather': ['weather_attribute'],
            'navigate': ['poi','traffic','address','distance'],
            'schedule': ['event','date','time','party','agenda','room']
        }
        :param raw_data:
        :param add_to_vocab:
        :param data_type:
        :return:
        """
        tokenized_data = self._load_tokenized_data(data_type)
        if tokenized_data is not None:
            logging.info('directly loading %s' % data_type)
            return tokenized_data
        tokenized_data = []
        state_dump = {}
        for dial_id, raw_dial in enumerate(raw_data):
            tokenized_dial = []
            prev_utter = ''
            single_turn = {}
            constraint_dict = {}
            intent = raw_dial['scenario']['task']['intent']
            if cfg.intent != 'all' and cfg.intent != intent:
                if intent not in ['navigate', 'weather', 'schedule']:
                    raise ValueError('what is %s intent bro?' % intent)
                else:
                    continue
            prev_response = []
            for turn_num, dial_turn in enumerate(raw_dial['dialogue']):
                state_dump[(dial_id, turn_num)] = {}
                if dial_turn['turn'] == 'driver':
                    u = self._lemmatize(self._tokenize(dial_turn['data']['utterance']))
                    u = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), u)
                    single_turn['user'] = prev_response + u.split() + ['EOS_U']
                    prev_utter += u
                elif dial_turn['turn'] == 'assistant':
                    s = dial_turn['data']['utterance']
                    # find entities and replace them
                    s = re.sub('(\d+) ([ap]m)', lambda x: x.group(1) + x.group(2), s)
                    s, reqs = self._replace_entity(s, self.entity_dict, prev_utter, intent)
                    single_turn['response'] = s.split() + ['EOS_M']
                    # get constraints
                    if not constraint_dict:
                        constraint_dict = dial_turn['data']['slots']
                    else:
                        for k, v in dial_turn['data']['slots'].items():
                            constraint_dict[k] = v
                    constraint_dict = self._clean_constraint_dict(constraint_dict, intent)

                    raw_constraints = constraint_dict.values()
                    raw_constraints = [self._lemmatize(self._tokenize(_)) for _ in raw_constraints]

                    # add separator
                    constraints = []
                    for item in raw_constraints:
                        if constraints:
                            constraints.append(';')
                        constraints.extend(item.split())
                    # get requests
                    dataset_requested = set(
                        filter(lambda x: dial_turn['data']['requested'][x], dial_turn['data']['requested'].keys()))
                    requestable = {
                        'weather': ['weather_attribute'],
                        'navigate': ['poi', 'traffic', 'address', 'distance'],
                        'schedule': ['date', 'time', 'party', 'agenda', 'room']
                    }
                    requests = sorted(list(dataset_requested.intersection(reqs)))

                    single_turn['constraint'] = constraints + ['EOS_Z1']
                    single_turn['requested'] = requests + ['EOS_Z2']
                    single_turn['turn_num'] = len(tokenized_dial)
                    single_turn['dial_id'] = dial_id
                    single_turn['degree'] = self.db_degree(constraints, raw_dial['scenario']['kb']['items'])
                    self.db[dial_id] = raw_dial['scenario']['kb']['items']
                    if 'user' in single_turn:
                        state_dump[(dial_id, len(tokenized_dial))]['constraint'] = constraint_dict
                        state_dump[(dial_id, len(tokenized_dial))]['request'] = requests
                        tokenized_dial.append(single_turn)
                    prev_response = single_turn['response']
                    single_turn = {}
            if add_to_vocab:
                for single_turn in tokenized_dial:
                    for word_token in single_turn['constraint'] + single_turn['requested'] + \
                            single_turn['user'] + single_turn['response']:
                        self.vocab.add_item(word_token)
            tokenized_data.append(tokenized_dial)
        self._save_tokenized_data(tokenized_data, data_type)
        return tokenized_data

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            new_dial = []
            for turn in dial:
                turn['constraint'] = self.vocab.sentence_encode(turn['constraint'])
                turn['requested'] = self.vocab.sentence_encode(turn['requested'])
                turn['bspan'] = turn['constraint'] + turn['requested']
                turn['user'] = self.vocab.sentence_encode(turn['user'])
                turn['response'] = self.vocab.sentence_encode(turn['response'])
                turn['u_len'] = len(turn['user'])
                turn['m_len'] = len(turn['response'])
                turn['degree'] = self._degree_vec_mapping(turn['degree'])
                new_dial.append(turn)
            encoded_data.append(new_dial)
        return encoded_data

    def _get_entity_dict(self, entity_data):
        entity_dict = {}
        for k in entity_data:
            if isinstance(entity_data[k][0], str):
                for entity in entity_data[k]:
                    entity = self._lemmatize(self._tokenize(entity))
                    entity_dict[entity] = k
                    if k in ['event', 'poi_type']:
                        entity_dict[entity.split()[0]] = k
                        self.abbr_dict[entity.split()[0]] = entity
            elif isinstance(entity_data[k][0], dict):
                for entity_entry in entity_data[k]:
                    for entity_type, entity in entity_entry.items():
                        entity_type = 'poi_type' if entity_type == 'type' else entity_type
                        entity = self._lemmatize(self._tokenize(entity))
                        entity_dict[entity] = entity_type
                        if entity_type in ['event', 'poi_type']:
                            entity_dict[entity.split()[0]] = entity_type
                            self.abbr_dict[entity.split()[0]] = entity
        self.entity_dict = entity_dict

    def db_degree(self, constraints, items):
        cnt = 0
        if items is not None:
            for item in items:
                item = item.values()
                flg = True
                for c in constraints:
                    itemvaluestr = " ".join(list(item))
                    if c not in itemvaluestr:
                        flg = False
                        break
                if flg:
                    cnt += 1
        return cnt

    def db_degree_handler(self, z_samples, idx=None, *args, **kwargs):
        control_vec = []
        for i,cons_idx_list in enumerate(z_samples):
            constraints = set()
            for cons in cons_idx_list:
                if not isinstance(cons, str):
                    cons = self.vocab.decode(cons)
                if cons == 'EOS_Z1':
                    break
                constraints.add(cons)
            items = self.db[idx[i]]
            degree = self.db_degree(constraints, items)
            control_vec.append(self._degree_vec_mapping(degree))
        return np.array(control_vec)

class MultiWozReader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.train, cfg.dev, cfg.test, cfg.db)
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        requestable_keys = ['addr', 'area', 'fee', 'name', 'phone', 'post', 'price', 'type', 'department', 'internet', 'parking', 'stars', 'food', 'arrive', 'day', 'depart', 'dest', 'leave', 'ticket', 'id']
        
        tokenized_data = []
        vk_map = self._value_key_map(db_data)
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                requested = []
                for slot_act in turn['usr']['slu']:
                    if slot_act == 'inform':
                        slot_values = turn['usr']['slu'][slot_act]
                        for v in slot_values:
                            s = v[1]
                            if s not in ['dont_care', 'none']:
                                constraint.append(s)
                    elif slot_act == 'request':
                        slot_values = turn['usr']['slu'][slot_act]
                        for v in slot_values:
                            s = v[0]
                            if s in requestable_keys:
                                requested.append(s)
                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                constraint.append('EOS_Z1')
                requested.append('EOS_Z2')
                user = turn['usr']['transcript'].split() + ['EOS_U']
                response = self._replace_entity(turn['sys']['sent'], vk_map, constraint).split() + ['EOS_M']
                response_origin = turn['sys']['sent'].split()
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'response_origin': response_origin,
                    'constraint': constraint,
                    'requested': requested,
                    'degree': degree,
                })
                if construct_vocab:
                    for word in user + response + constraint + requested:
                        self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def _replace_entity(self, response, vk_map, constraint):
        response = re.sub('[cC][., ]*[bB][., ]*\d[., ]*\d[., ]*\w[., ]*\w', 'postcode_SLOT', response)
        response = re.sub('\d{5}\s?\d{6}', 'phone_SLOT', response)
        constraint_str = ' '.join(constraint)
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 \
                    or (start_idx != 0 and response[start_idx - 1] != ' ') \
                    or (v in constraint_str):
                continue
            response = clean_replace(response, v, k + '_SLOT')
        return response

    def _value_key_map(self, db_data):
        def normal(string):
            string = string.lower()
            string = re.sub(r'\s*-\s*', '', string)
            string = re.sub(r' ', '_', string)
            string = re.sub(r',', '_,', string)
            string = re.sub(r'\'', '_', string)
            string = re.sub(r'\.', '_.', string)
            string = re.sub(r'_+', '_', string)
            string = re.sub(r'children', 'child_-s', string)
            return string
        requestable_dict = {'address':'addr', 
                            'area':'area',
                            'entrance fee':'fee',
                            'name':'name',
                            'phone':'phone', 
                            'postcode':'post',
                            'pricerange':'price', 
                            'type':'type',
                            'department':'department',
                            'internet':'internet',
                            'parking':'parking',
                            'stars':'stars',
                            'food':'food',
                            'arriveBy':'arrive',
                            'day':'day',
                            'departure':'depart',
                            'destination':'dest',
                            'leaveAt':'leave',
                            'price':'ticket',
                            'trainId':'id'}
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_dict:
                    value_key[normal(v)] = requestable_dict[k]
        return value_key

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                response_origin = ' '.join(turn['response_origin'])
                constraint = self.vocab.sentence_encode(turn['constraint'])
                requested = self.vocab.sentence_encode(turn['requested'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']

                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': prev_response + user,
                    'response': response,
                    'response_origin': response_origin,
                    'bspan': constraint + requested,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _get_clean_db(self, raw_db_data):
        for entry in raw_db_data:
            for k, v in list(entry.items()):
                if not isinstance(v, str) or v == '?':
                    entry.pop(k)

    def _construct(self, train_json_path, dev_json_path, test_json_path, db_json_path):
        """
        construct encoded train, dev, test set.
        :param train_json_path:
        :param dev_json_path:
        :param test_json_path:
        :param db_json_path: list
        :return:
        """
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        with open(train_json_path) as f:
            train_raw_data = json.loads(f.read().lower())
        with open(dev_json_path) as f:
            dev_raw_data = json.loads(f.read().lower())
        with open(test_json_path) as f:
            test_raw_data = json.loads(f.read().lower())
        db_data = list()
        for domain_db_json_path in db_json_path:
            with open(domain_db_json_path) as f:
                db_data_domain = json.loads(f.read().lower())
                for i, item in enumerate(db_data_domain):
                    item['ref'] = f'{i:08d}'
                db_data += db_data_domain
        self._get_clean_db(db_data)
        self.db = db_data
        
        train_tokenized_data = self._get_tokenized_data(train_raw_data, db_data, construct_vocab)
        dev_tokenized_data = self._get_tokenized_data(dev_raw_data, db_data, construct_vocab)
        test_tokenized_data = self._get_tokenized_data(test_raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        self.train = self._get_encoded_data(train_tokenized_data)
        self.dev = self._get_encoded_data(dev_tokenized_data)
        self.test = self._get_encoded_data(test_tokenized_data)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def db_search(self, constraints):
        match_results = []
        for entry in self.db:
            entry_values = ' '.join(entry.values())
            match = True
            for c in constraints:
                if c not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results
    
    def wrap_result(self, turn_batch, gen_m, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """

        results = []
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2'}
        batch_size = len(turn_batch['user'])
        for i in range(batch_size):
            entry = {}
            if prev_z is not None:
                src = prev_z[i] + turn_batch['user'][i]
            else:
                src = turn_batch['user'][i]
            for key in turn_batch:
                entry[key] = turn_batch[key][i]
                if key in eos_syntax:
                    entry[key] = self.vocab.sentence_decode(entry[key], eos=eos_syntax[key])
            if gen_z:
                entry['generated_bspan'] = self.vocab.sentence_decode(gen_z[i], eos='EOS_Z2')
            else:
                entry['generated_bspan'] = ''
            if gen_m:
                entry['generated_response'] = self.vocab.sentence_decode(gen_m[i], eos='EOS_M')
                constraint_request = entry['generated_bspan'].split()
                constraints = constraint_request[:constraint_request.index('EOS_Z1')] if 'EOS_Z1' \
                    in constraint_request else constraint_request
                for j, ent in enumerate(constraints):
                    constraints[j] = ent.replace('_', ' ')
                degree = self.db_search(constraints)
                #print('constraints',constraints)
                #print('degree',degree)
                venue = random.sample(degree, 1)[0] if degree else dict()
                l = [self.vocab.decode(_) for _ in gen_m[i]]
                if 'EOS_M' in l:
                    l = l[:l.index('EOS_M')]
                l_origin = []
                for word in l:
                    if 'SLOT' in word:
                        word = word[:-5]
                        if word in venue.keys():
                            value = venue[word]
                            if value != '?':
                                l_origin.append(value.replace(' ', '_'))
                    else:
                        l_origin.append(word)
                entry['generated_response_origin'] = ' '.join(l_origin)
            else:
                entry['generated_response'] = ''
                entry['generated_response_origin'] = ''
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(cfg.result_path, 'w')
            self.result_file.write(str(cfg))
            write_header = True

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised', 'generated_response_origin', 'response_origin']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)
    if maxlen is not None and cfg.truncated:
        maxlen = min(seq_maxlen, maxlen)
    else:
        maxlen = seq_maxlen
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_glove_matrix(vocab, initial_embedding_np):
    """
    return a glove embedding matrix
    :param self:
    :param glove_file:
    :param initial_embedding_np:
    :return: np array of [V,E]
    """
    ef = open(cfg.glove_path, 'r')
    cnt = 0
    vec_array = initial_embedding_np
    old_avg = np.average(vec_array)
    old_std = np.std(vec_array)
    vec_array = vec_array.astype(np.float32)
    new_avg, new_std = 0, 0

    for line in ef.readlines():
        line = line.strip().split(' ')
        word, vec = line[0], line[1:]
        vec = np.array(vec, np.float32)
        word_idx = vocab.encode(word)
        if word.lower() in ['unk', '<unk>'] or word_idx != vocab.encode('<unk>'):
            cnt += 1
            vec_array[word_idx] = vec
            new_avg += np.average(vec)
            new_std += np.std(vec)
    new_avg /= cnt
    new_std /= cnt
    ef.close()
    logging.info('%d known embedding. old mean: %f new mean %f, old std %f new std %f' % (cnt, old_avg,
                                                                                          new_avg, old_std, new_std))
    return vec_array
