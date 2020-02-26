# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from convlab2.util.camrest.lexicalize import delexicalize_da, flat_da, deflat_da, lexicalize_da
from convlab2.util.camrest.dbquery import Database


class CamrestVector(Vector):

    def __init__(self, voc_file, voc_opp_file, character='sys',
                 intent_file=os.path.join(
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     'data/camrest/trackable_intent.json')):

        with open(intent_file) as f:
            intents = json.load(f)
        self.informable = intents['informable']
        self.requestable = intents['requestable']
        self.db = Database()

        with open(voc_file) as f:
            self.da_voc = f.read().splitlines()
        with open(voc_opp_file) as f:
            self.da_voc_opp = f.read().splitlines()
        self.character = character
        self.generate_dict()

    def generate_dict(self):
        """
        init the dict for mapping state/action into vector
        """
        self.act2vec = dict((a, i) for i, a in enumerate(self.da_voc))
        self.vec2act = dict((v, k) for k, v in self.act2vec.items())
        self.da_dim = len(self.da_voc)
        self.opp2vec = dict((a, i) for i, a in enumerate(self.da_voc_opp))
        self.da_opp_dim = len(self.da_voc_opp)

        self.inform_da, self.request_da = [], []
        for da in self.da_voc:
            i, s, v = da.split('-')
            if s == 'none':
                continue
            if i in self.informable:
                self.inform_da.append('-'.join([s, v]))
            elif i in self.requestable:
                self.request_da.append(s)
        self.inform2vec = dict((a, i) for i, a in enumerate(self.inform_da))
        self.inform_dim = len(self.inform_da)
        self.request2vec = dict((a, i) for i, a in enumerate(self.request_da))
        self.request_dim = len(self.request_da)

        self.state_dim = self.da_opp_dim + self.da_dim + self.inform_dim + 6 + 1

    def pointer(self, turn):
        constraint = turn.items()
        entities = self.db.query(constraint)
        pointer_vector = self.one_hot_vector(len(entities))

        return pointer_vector

    def one_hot_vector(self, num):
        """Return number of available entities for particular domain."""
        if num == 0:
            vector = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector = np.array([0, 0, 0, 0, 0, 1])

        return vector

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (dict):
                Dialog state
            action (tuple):
                Dialog act
        Returns:
            state_vec (np.array): Dialog state vector
        """
        self.state = state['belief_state']

        action = state['user_action'] if self.character == 'sys' else state['system_action']
        opp_action = delexicalize_da(action, self.requestable)
        opp_action = flat_da(opp_action)
        opp_act_vec = np.zeros(self.da_opp_dim)
        for da in opp_action:
            if da in self.opp2vec:
                opp_act_vec[self.opp2vec[da]] = 1.

        action = state['system_action'] if self.character == 'sys' else state['user_action']
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        last_act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                last_act_vec[self.act2vec[da]] = 1.

        inform = np.zeros(self.inform_dim)
        for slot, value in state['belief_state'].items():
            p = 1
            key = slot + '-' + str(p)
            while inform[self.inform2vec[key]]:
                p += 1
                key = slot + '-' + str(p)
                if key not in self.inform2vec:
                    break
            else:
                inform[self.inform2vec[key]] = 1.

        degree = self.pointer(state['belief_state'])

        final = 1. if state['terminated'] else 0.

        state_vec = np.r_[opp_act_vec, last_act_vec, inform, degree, final]
        return state_vec

    def action_devectorize(self, action_vec):
        """recover an action

        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple): Dialog act
        """
        act_array = []
        for i, idx in enumerate(action_vec):
            if idx == 1:
                act_array.append(self.vec2act[i])
        action = deflat_da(act_array)
        constraint = self.state.items()
        entities = self.db.query(constraint)
        action = lexicalize_da(action, entities, self.state, self.requestable)
        return action

    def action_vectorize(self, action):
        action = delexicalize_da(action, self.requestable)
        action = flat_da(action)
        act_vec = np.zeros(self.da_dim)
        for da in action:
            if da in self.act2vec:
                act_vec[self.act2vec[da]] = 1.
        return act_vec
