"""
"""

import json
import os
import pickle
import random
from collections import Counter
from copy import deepcopy

import numpy as np

from convlab2.util.camrest.dbquery import Database

days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
domain_keywords = {
    'restaurant': 'place to dine',
    'train': 'train',
    'hotel': 'place to stay',
    'attraction': 'places to go',
    'police': 'help',
    'taxi': 'taxi',
    'hospital': 'hospital'
}
templates = {
    'intro': 'You are looking forward to trying local restaurants.',
    'request': 'Once you find a restaurnat, make sure you get {}.',
    'area': 'The restaurant should be in the {}.',
    'food': 'The restaurant should serve {} food.',
    'name': 'You are looking for a particular restaurant. Its name is called {}.',
    'pricerange': 'The restaurant should be in the {} price range.',
}

def nomial_sample(counter: Counter):
    return list(counter.keys())[np.argmax(np.random.multinomial(1, list(counter.values())))]

class GoalGenerator:
    """User goal generator."""

    def __init__(self,
                 goal_model_path=os.path.join(
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     'data/camrest/goal/goal_model.pkl'),
                 corpus_path=None):
        """
        Args:
            goal_model_path: path to a goal model 
            corpus_path: path to a dialog corpus to build a goal model 
        """
        self.goal_model_path = goal_model_path
        self.corpus_path = corpus_path
        self.db = Database()
        if os.path.exists(self.goal_model_path):
            self.ind_slot_dist, self.ind_slot_value_dist = pickle.load(
                open(self.goal_model_path, 'rb'))
            print('Loading goal model is done')
        else:
            self._build_goal_model()
            print('Building goal model is done')

    def _build_goal_model(self):
        dialogs = json.load(open(self.corpus_path))

        # independent goal slot distribution
        ind_slot_value_cnt = dict()

        for d in dialogs:
            if 'info' in d['goal']:
                for slot in d['goal']['info']:
                    if 'info' not in ind_slot_value_cnt:
                        ind_slot_value_cnt['info'] = {}
                    if slot not in ind_slot_value_cnt['info']:
                        ind_slot_value_cnt['info'][slot] = Counter()
                    ind_slot_value_cnt['info'][slot][d['goal']['info'][slot]] += 1
            if 'reqt' in d['goal']:
                for slot in d['goal']['reqt']:
                    if 'reqt' not in ind_slot_value_cnt:
                        ind_slot_value_cnt['reqt'] = Counter()
                    ind_slot_value_cnt['reqt'][slot] += 1

        self.ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
        self.ind_slot_dist = dict()
        if 'info' in ind_slot_value_cnt:
            for slot in ind_slot_value_cnt['info']:
                if 'info' not in self.ind_slot_dist:
                    self.ind_slot_dist['info'] = {}
                if slot not in self.ind_slot_dist['info']:
                    self.ind_slot_dist['info'][slot] = {}
                self.ind_slot_dist['info'][slot] = sum(ind_slot_value_cnt['info'][slot].values()) / len(dialogs)
                slot_total = sum(ind_slot_value_cnt['info'][slot].values())
                for val in self.ind_slot_value_dist['info'][slot]:
                    self.ind_slot_value_dist['info'][slot][val] = ind_slot_value_cnt['info'][slot][
                                                                              val] / slot_total
        if 'reqt' in ind_slot_value_cnt:
            for slot in ind_slot_value_cnt['reqt']:
                if 'reqt' not in self.ind_slot_dist:
                    self.ind_slot_dist['reqt'] = {}
                self.ind_slot_dist['reqt'][slot] = ind_slot_value_cnt['reqt'][slot] / len(dialogs)
                self.ind_slot_value_dist['reqt'][slot] = ind_slot_value_cnt['reqt'][slot] / len(dialogs)

        pickle.dump((self.ind_slot_dist, self.ind_slot_value_dist),
                    open(self.goal_model_path, 'wb'))

    def _get_restaurant_goal(self):
        cnt_slot = self.ind_slot_dist
        cnt_slot_value = self.ind_slot_value_dist

        while True:
            # domain_goal = defaultdict(lambda: {})
            # domain_goal = {'info': {}, 'reqt': []}
            domain_goal = {'info': {}}
            # inform
            if 'info' in cnt_slot:
                for slot in cnt_slot['info']:
                    if random.random() < cnt_slot['info'][slot]:
                        domain_goal['info'][slot] = nomial_sample(cnt_slot_value['info'][slot])

                if 'name' in domain_goal['info'] and len(
                        domain_goal['info']) > 1:
                    if random.random() < cnt_slot['info']['name']:
                        domain_goal['info'] = {'name': domain_goal['info']['name']}
                    else:
                        del domain_goal['info']['name']

                if domain_goal['info'] == {}:
                    continue
            # request
            if 'reqt' in cnt_slot:
                reqt = [slot for slot in cnt_slot['reqt']
                        if random.random() < cnt_slot['reqt'][slot] and slot not in
                        domain_goal['info']]
                if len(reqt) > 0:
                    domain_goal['reqt'] = reqt
                    
            # fail_info
            if 'info' in domain_goal and len(self.db.query(domain_goal['info'].items())) == 0:
                num_trial = 0
                while num_trial < 100:
                    adjusted_info = self._adjust_info(domain_goal['info'])
                    if len(self.db.query(adjusted_info.items())) > 0:
                        domain_goal['info'] = adjusted_info

                        break
                    num_trial += 1

                if num_trial >= 100:
                    continue

            # at least there is one request and book
            if 'reqt' in domain_goal:
                break

        return domain_goal

    def get_user_goal(self):
        user_goal = self._get_restaurant_goal()
        assert len(user_goal.keys()) > 0

        return user_goal

    def _adjust_info(self, info):
        # adjust one of the slots of the info
        adjusted_info = deepcopy(info)
        slot = random.choice(list(info.keys()))
        adjusted_info[slot] = random.choice(list(self.ind_slot_value_dist['info'][slot].keys()))
        return adjusted_info

    def build_message(self, user_goal):
        message = []
        state = deepcopy(user_goal)

        # intro
        m = [templates['intro']]

        # info
        def fill_info_template(user_goal, slot, info):
            return templates[slot].format(user_goal[info][slot])

        info = 'info'

        while len(state[info]) > 0:
            num_acts = random.randint(1, min(len(state[info]), 3))
            slots = random.sample(list(state[info].keys()), num_acts)
            sents = [fill_info_template(user_goal, slot, info) for slot in slots]
            m.extend(sents)
            message.append(' '.join(m))
            m = []
            for slot in slots:
                del state[info][slot]
                    
        # reqt
        if 'reqt' in state:
            slot_strings = []
            for slot in state['reqt']:
                if slot in ['food']:
                    continue
                slot_strings.append(slot)
            if len(slot_strings) > 0:
                message.append(templates['request'].format(', '.join(slot_strings)))
            if 'food' in state['reqt']:
                message.append('Make sure to ask about what food it serves.')

        return message


if __name__ == "__main__":
    goal_generator = GoalGenerator(corpus_path=os.path.join(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                'data/camrest/CamRest676_v2.json'))
    while True:
        user_goal = goal_generator.get_user_goal()
        print(user_goal)
        # message = goal_generator.build_message(user_goal)
        # pprint(message)
