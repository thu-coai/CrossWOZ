"""
Dataloader base class. Every dataset should inherit this class and implement its own dataloader.
"""
from abc import ABC, abstractmethod
import os
import json
from pprint import pprint
from convlab2.util.file_util import read_zipped_json


class DatasetDataloader(ABC):
    def __init__(self):
        self.data = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """
        load data from file, according to what is need
        :param args:
        :param kwargs:
        :return: data
        """
        pass


class MultiWOZDataloader(DatasetDataloader):
    def __init__(self):
        super(MultiWOZDataloader, self).__init__()

    def load_data(self,
                  data_dir=os.path.abspath(os.path.join(os.path.abspath(__file__), '../../../../data/multiwoz')),
                  data_key='all',
                  role='all',
                  utterance=False,
                  dialog_act=False,
                  context=False,
                  context_window_size=0,
                  context_dialog_act=False,
                  belief_state=False,
                  last_opponent_utterance=False,
                  last_self_utterance=False,
                  ontology = False,
                  session_id=False,
                  span_info=False,
                  terminated=False,
                  goal=False
                  ):

        def da2tuples(dialog_act):
            tuples = []
            for domain_intent, svs in dialog_act.items():
                for slot, value in svs:
                    domain, intent = domain_intent.split('-')
                    tuples.append([intent, domain, slot, value])
            return tuples

        assert role in ['system', 'user', 'all']
        info_list = list(filter(eval, ['utterance', 'dialog_act', 'context', 'context_dialog_act', 'belief_state',
                                       'last_opponent_utterance', 'last_self_utterance', 'session_id', 'span_info',
                                       'terminated', 'goal']))
        self.data = {'train': {}, 'val': {}, 'test': {}, 'role': role}
        if data_key=='all':
            data_key_list = ['train', 'val', 'test']
        else:
            data_key_list = [data_key]
        for data_key in data_key_list:
            data = read_zipped_json(os.path.join(data_dir, '{}.json.zip'.format(data_key)), '{}.json'.format(data_key))
            print('loaded {}, size {}'.format(data_key, len(data)))
            for x in info_list:
                self.data[data_key][x] = []
            for sess_id, sess in data.items():
                cur_context = []
                cur_context_dialog_act = []
                for i, turn in enumerate(sess['log']):
                    text = turn['text']
                    da = da2tuples(turn['dialog_act'])
                    if role=='system' and i % 2 == 0:
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)
                        continue
                    elif role=='user' and i % 2 == 1:
                        cur_context.append(text)
                        cur_context_dialog_act.append(da)
                        continue
                    if utterance:
                        self.data[data_key]['utterance'].append(text)
                    if dialog_act:
                        self.data[data_key]['dialog_act'].append(da)
                    if context and context_window_size:
                        self.data[data_key]['context'].append(cur_context[-context_window_size:])
                    if context_dialog_act and context_window_size:
                        self.data[data_key]['context_dialog_act'].append(cur_context_dialog_act[-context_window_size:])
                    if belief_state:
                        self.data[data_key]['belief_state'].append(turn['metadata'])
                    if last_opponent_utterance:
                        self.data[data_key]['last_opponent_utterance'].append(
                            cur_context[-2] if len(cur_context) > 1 else '')
                    if last_self_utterance:
                        self.data[data_key]['last_self_utterance'].append(
                            cur_context[-3] if len(cur_context) > 2 else '')
                    if session_id:
                        self.data[data_key]['session_id'].append(sess_id)
                    if span_info:
                        self.data[data_key]['span_info'].append(turn['span_info'])
                    if terminated:
                        self.data[data_key]['terminated'].append(i+2 >= len(sess['log']))
                    if goal:
                        self.data[data_key]['goal'].append(sess['goal'])
                    cur_context.append(text)
                    cur_context_dialog_act.append(da)
        if ontology:
            ontology_path = os.path.join(data_dir, 'ontology.json')
            self.data['ontology'] = json.load(open(ontology_path))

        return self.data


if __name__ == '__main__':
    m = MultiWOZDataloader()
    pprint(m.load_data(role='system', context=True, context_dialog_act=True, belief_state=True,
                       last_opponent_utterance=True, last_self_utterance=True, context_window_size=0, span_info=False))
