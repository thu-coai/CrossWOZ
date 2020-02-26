# -*- coding: utf-8 -*-
"""
@author: keshuichonglx 
"""
import collections
import json
import os
import pickle
import random

import numpy as np

informable_keys = ['address', 'area', 'food', 'name', 'phone', 'pricerange']

class UserDataManager(object):

    def __init__(self):
        
        self.__org_goals = None
        self.__org_usr_dass = None
        self.__org_sys_dass = None

        self.__goals = None
        self.__usr_dass = None
        self.__sys_dass = None

        self.__goals_seg = None
        self.__usr_dass_seg = None
        self.__sys_dass_seg = None

        self.__voc_goal = None
        self.__voc_usr = None
        self.__voc_usr_rev = None
        self.__voc_sys = None

        self.pad = '<PAD>'
        self.unk = '<UNK>'
        self.sos = '<SOS>'
        self.eos = '<EOS>'
        self.special_words = [self.pad, self.unk, self.sos, self.eos]
        
        self.voc_goal, self.voc_usr, self.voc_sys = self.vocab_loader()

    @staticmethod
    def usrgoal2seq(goal: dict):
        def add(lt: list, domain_goal: dict, intent: str):
            mapping = domain_goal.get(intent, {})
            slots = [slot for slot in mapping.keys() if isinstance(mapping[slot], str)]
            if len(slots) > 0:
                lt.append(intent)
                lt.append('(')
                slots.sort()
                lt.extend(slots)
                lt.append(')')

        ret = []
        domain_goal = goal
        # info
        add(ret, domain_goal, 'info')
        # reqt
        slots = domain_goal.get('reqt', [])
        if slots:
            ret.append('reqt')
            ret.append('(')
            ret.extend(slots)
            ret.append(')')

        return ret

    @staticmethod
    def query_goal_for_sys(slot, value, goal):
        ret = None
        goal_slot = slot if slot in informable_keys else 'Unknow'
        if goal_slot != 'Unknow':
            check = {'info': None}
            for zone in check.keys():
                dt = goal.get(zone, {})
                if goal_slot in dt.keys():
                    check[zone] = (dt[goal_slot].lower() == value.lower())
            if True in check.values() or False in check.values():
                # in constraint
                ret = 'InConstraint'
                if True in check.values() and False in check.values():
                    ret += '[C]'
                elif True in check.values() and False not in check.values():
                    ret += '[R]'
                elif True not in check.values() and False in check.values():
                    ret += '[W]'
            elif goal_slot in goal.get('reqt', []):
                # in Request
                ret = 'InRequest'
            else:
                # not in goal
                ret = 'NotInGoal'
        elif goal_slot == 'Unknow':
            ret = 'Unknow'
        else:
            ret = None
        return ret

    @staticmethod
    def sysda2seq(sys_da: dict, goal: dict):
        ret = []
        for (act, pairs) in sorted(sys_da.items(), key=lambda x: x[0]):
            ret.append(act)
            ret.append('(')
            if 'general' not in act:
                for [slot, val] in sorted(pairs, key=lambda x: x[0]):
                    if (act == 'request'):
                        ret.append(slot)
                    else:
                        m_val = UserDataManager.query_goal_for_sys(slot, val, goal)
                        if m_val is not None:
                            ret.append(slot + '=' + m_val)

            ret.append(')')
        # assert len(ret) > 0, str(sys_da)
        return ret

    @staticmethod
    def query_goal_for_usr(slot, value, goal):
        ret = None
        goal_slot = slot if slot in informable_keys else 'Unknow'
        if goal_slot is not None and goal_slot != 'Unknow':
            check = {'info': None}
            for zone in check.keys():
                dt = goal.get(zone, {})
                if goal_slot in dt.keys():
                    check[zone] = (dt[goal_slot].replace(' ', '').lower() == value.replace(' ', '').lower())
            if True in check.values() or False in check.values():
                # in constraint
                if True in check.values():
                    ret = 'In' + [zone for (zone, value) in check.items() if value == True][0].capitalize()
                else:
                    ret = 'In' + [zone for (zone, value) in check.items() if value == False][0].capitalize()
            else:
                # not in goal
                ret = 'DoNotCare'
        elif goal_slot == 'Unknow':
            ret = 'Unknow'
        else:
            ret = None
        return ret

    @staticmethod
    def usrda2seq(usr_da: dict, goal: dict):
        ret = []
        for (act, pairs) in sorted(usr_da.items(), key=lambda x: x[0]):
            ret.append(act)
            ret.append('(')
            if 'general' not in act:
                for [slot, val] in sorted(pairs, key=lambda x: x[0]):
                    if (act == 'request'):
                        ret.append(slot)
                    else:
                        m_val = UserDataManager.query_goal_for_usr(slot, val, goal)
                        if m_val is not None:
                            ret.append(slot + '=' + m_val)

            ret.append(')')

        return ret

    @staticmethod
    def usrseq2da(usr_seq: list, goal: dict):
        ret = {}
        cur_act = None
        for word in usr_seq:
            if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
                continue

            # act
            if '-' in word:
                cur_act = word
                ret[cur_act] = []

            # slot-value
            elif '=' in word and cur_act is not None:
                slot_da, value_pos = word.split('=')
                value_pos = value_pos.lower()
                slot_goal = slot_da if slot_da in informable_keys else None
                if slot_goal is not None:
                    value = None
                    if value_pos == 'ininfo':
                        value = goal.get('info', {}).get(slot_goal, None)
                    elif value_pos == 'donotcare':
                        value = 'dontcare'

                    if value is not None:
                        ret[cur_act].append([slot_da, value])
                    else:
                        pass
                        #assert False, slot_da
                else:
                    pass
                    # assert False, '%s - %s' % (domain, slot_da)

            # slot in reqt
            elif cur_act is not None:
                ret[cur_act].append([word, '?'])

        return ret

    @staticmethod
    def ref_data2stand(da):
        return eval(str(da).lower())

    def org_data_loader(self):
        if self.__org_goals is None or self.__org_usr_dass is None or self.__org_sys_dass is None:
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
                                     'data/camrest/CamRest676_v2.json')
            # 'data/multiwoz/annotated_user_da_with_span_100sample.json')
            with open(file_path) as f:
                full_data = json.load(f)
            goals = []
            usr_dass = []
            sys_dass = []
            for session in full_data:
                goal = session.get('goal', {})
                logs = session.get('dial', [])
                usr_das, sys_das = [], []
                for turn in range(len(logs)):
                    # <class 'dict'>: {'Hotel-Inform': [['Price', 'cheap'], ['Type', 'hotel']]}
                    usr_da = self.ref_data2stand(logs[turn]['usr'].get('dialog_act', {}))
                    sys_da = self.ref_data2stand(logs[turn]['sys'].get('dialog_act', {}))
                    usr_das.append(usr_da)
                    sys_das.append(sys_da)
                else:
                    goals.append(goal)
                    usr_dass.append(usr_das)
                    sys_dass.append(sys_das)

            self.__org_goals = [UserDataManager.usrgoal2seq(goal) for goal in goals]
            self.__org_usr_dass = [[UserDataManager.usrda2seq(usr_da, goal) for usr_da in usr_das] for (usr_das, goal) in zip(usr_dass, goals)]
            self.__org_sys_dass = [[UserDataManager.sysda2seq(sys_da, goal) for sys_da in sys_das] for (sys_das, goal) in zip(sys_dass, goals)]

        return self.__org_goals, self.__org_usr_dass, self.__org_sys_dass

    def vocab_loader(self):
        if self.__voc_goal is None or self.__voc_usr is None or self.__voc_usr_rev is None or self.__voc_sys is None:
            vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
                                      'data/camrest/goal/vocab.pkl')

            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    self.__voc_goal, self.__voc_usr, self.__voc_sys = pickle.load(f)
            else:
                goals, usr_dass, sys_dass = self.org_data_loader()

                # goals
                counter = collections.Counter()
                for goal in goals:
                    for word in goal:
                        counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_goal = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for usr_das in usr_dass:
                    for usr_da in usr_das:
                        for word in usr_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_usr = {x: i for i, x in enumerate(word_list)}

                counter = collections.Counter()
                for sys_das in sys_dass:
                    for sys_da in sys_das:
                        for word in sys_da:
                            counter[word] += 1
                word_list = self.special_words + [x for x in counter.keys()]
                self.__voc_sys = {x: i for i, x in enumerate(word_list)}
                
                with open(vocab_path, 'wb') as f:
                    pickle.dump((self.__voc_goal, self.__voc_usr, self.__voc_sys), f)
                print('voc build ok')

            self.__voc_usr_rev = {val: key for (key, val) in self.__voc_usr.items()}

        return self.__voc_goal, self.__voc_usr, self.__voc_sys

    def get_voc_size(self):
        return len(self.voc_goal), len(self.voc_usr), len(self.voc_sys)

    def data_loader(self):
        if self.__goals is None or self.__usr_dass is None or self.__sys_dass is None:
            org_goals, org_usr_dass, org_sys_dass = self.org_data_loader()
            self.__goals = [self.get_goal_id(goal) for goal in org_goals]
            self.__usr_dass = [self.get_usrda_id(usr_das) for usr_das in org_usr_dass]
            self.__sys_dass = [self.get_sysda_id(sys_das) for sys_das in org_sys_dass]
        assert len(self.__goals) == len(self.__usr_dass)
        assert len(self.__goals) == len(self.__sys_dass)
        return self.__goals, self.__usr_dass, self.__sys_dass

    @staticmethod
    def train_test_val_split(goals, usr_dass, sys_dass, test_size=0.1, val_size=0.1):
        idx = range(len(goals))
        idx_test = random.sample(idx, int(len(goals) * test_size))
        idx_train = list(set(idx) - set(idx_test))
        idx_val = random.sample(idx_train, int(len(goals) * val_size))
        idx_train = list(set(idx_train) - set(idx_val))
        idx_train = random.sample(idx_train, len(idx_train))
        return np.array(goals)[idx_train], np.array(usr_dass)[idx_train], np.array(sys_dass)[idx_train], \
               np.array(goals)[idx_test], np.array(usr_dass)[idx_test], np.array(sys_dass)[idx_test], \
               np.array(goals)[idx_val], np.array(usr_dass)[idx_val], np.array(sys_dass)[idx_val]

    def data_loader_seg(self):
        if self.__goals_seg is None or self.__usr_dass_seg is None or self.__sys_dass_seg is None:
            self.data_loader()
            self.__goals_seg, self.__usr_dass_seg, self.__sys_dass_seg = [], [], []

            for (goal, usr_das, sys_das) in zip(self.__goals, self.__usr_dass, self.__sys_dass):
                goals, usr_dass, sys_dass = [], [], []
                for length in range(len(usr_das)):
                    goals.append(goal)
                    usr_dass.append([usr_das[idx] for idx in range(length + 1)])
                    sys_dass.append([sys_das[idx] for idx in range(length + 1)])

                self.__goals_seg.append(goals)
                self.__usr_dass_seg.append(usr_dass)
                self.__sys_dass_seg.append(sys_dass)

        assert len(self.__goals_seg) == len(self.__usr_dass_seg)
        assert len(self.__goals_seg) == len(self.__sys_dass_seg)
        return self.__goals_seg, self.__usr_dass_seg, self.__sys_dass_seg

    def id2sentence(self, ids):
        sentence = [self.__voc_usr_rev[id] for id in ids]
        return sentence

    @staticmethod
    def train_test_val_split_seg(goals_seg, usr_dass_seg, sys_dass_seg, test_size=0.1, val_size=0.1):
        def dr(dss):
            return np.array([d for ds in dss for d in ds])

        idx = range(len(goals_seg))
        idx_test = random.sample(idx, int(len(goals_seg) * test_size))
        idx_train = list(set(idx) - set(idx_test))
        idx_val = random.sample(idx_train, int(len(goals_seg) * val_size))
        idx_train = list(set(idx_train) - set(idx_val))
        idx_train = random.sample(idx_train, len(idx_train))
        return dr(np.array(goals_seg)[idx_train]), dr(np.array(usr_dass_seg)[idx_train]), dr(np.array(sys_dass_seg)[idx_train]), \
               dr(np.array(goals_seg)[idx_test]), dr(np.array(usr_dass_seg)[idx_test]), dr(np.array(sys_dass_seg)[idx_test]), \
               dr(np.array(goals_seg)[idx_val]), dr(np.array(usr_dass_seg)[idx_val]), dr(np.array(sys_dass_seg)[idx_val])

    def get_goal_id(self, goal):
        return [self.voc_goal.get(word, self.voc_goal[self.unk]) for word in goal]
    
    def get_sysda_id(self, sys_das):
        return [[self.voc_sys.get(word, self.voc_sys[self.unk]) for word in sys_da] for sys_da in sys_das]
    
    def get_usrda_id(self, usr_das):
        return [[self.voc_usr[self.sos]] + [self.voc_usr.get(word, self.voc_usr[self.unk]) for word in usr_da] + [self.voc_usr[self.eos]]
                 for usr_da in usr_das]
