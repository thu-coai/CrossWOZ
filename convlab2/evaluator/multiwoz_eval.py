# -*- coding: utf-8 -*-

import re
import numpy as np
from copy import deepcopy
from pprint import pprint
from convlab2.evaluator.evaluator import Evaluator
from convlab2.util.multiwoz.dbquery import Database

requestable = \
    {'attraction': ['post', 'phone', 'addr', 'fee', 'area', 'type'],
     'restaurant': ['addr', 'phone', 'post', 'ref', 'price', 'area', 'food'],
     'train': ['ticket', 'time', 'ref', 'id', 'arrive', 'leave'],
     'hotel': ['addr', 'post', 'phone', 'ref', 'price', 'internet', 'parking', 'area', 'type', 'stars'],
     'taxi': ['car', 'phone'],
     'hospital': ['post', 'phone', 'addr'],
     'police': ['addr', 'post', 'phone']}

belief_domains = requestable.keys()

mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'price': 'pricerange'},
           'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name',
                     'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
           'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'type': 'type'},
           'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination',
                     'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
           'taxi': {'car': 'car type', 'phone': 'phone'},
           'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
           'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}

time_re = re.compile(r'^(([01]\d|2[0-3]):([0-5]\d)|24:00)$')
NUL_VALUE = ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care"]

class MultiWozEvaluator(Evaluator):
    def __init__(self):
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = {}
        self.cur_domain = ''
        self.booked = {}
        self.dbs = Database().dbs

    def _init_dict(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = {'info': {}, 'book': {}, 'reqt': []}
        return dic

    def _init_dict_booked(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = None
        return dic

    def _expand(self, _goal):
        goal = deepcopy(_goal)
        for domain in belief_domains:
            if domain not in goal:
                goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
                continue
            if 'info' not in goal[domain]:
                goal[domain]['info'] = {}
            if 'book' not in goal[domain]:
                goal[domain]['book'] = {}
            if 'reqt' not in goal[domain]:
                goal[domain]['reqt'] = []
        return goal

    def add_goal(self, goal):
        """init goal and array

        args:
            goal:
                dict[domain] dict['info'/'book'/'reqt'] dict/dict/list[slot]
        """
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = goal
        self.cur_domain = ''
        self.booked = self._init_dict_booked()

    def add_sys_da(self, da_turn):
        """add sys_da into array

        args:
            da_turn:
                list[intent, domain, slot, value]
        """
        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.sys_da_array.append(da + '-' + value)

            if da == 'booking-book-ref' and self.cur_domain in ['hotel', 'restaurant', 'train']:
                if not self.booked[self.cur_domain] and re.match(r'^\d{8}$', value) and \
                        len(self.dbs[self.cur_domain]) > int(value):
                    self.booked[self.cur_domain] = self.dbs[self.cur_domain][int(value)]
            elif da == 'train-offerbooked-ref' or da == 'train-inform-ref':
                if not self.booked['train'] and re.match(r'^\d{8}$', value) and len(self.dbs['train']) > int(value):
                    self.booked['train'] = self.dbs['train'][int(value)]
            elif da == 'taxi-inform-car':
                if not self.booked['taxi']:
                    self.booked['taxi'] = 'booked'

    def add_usr_da(self, da_turn):
        """add usr_da into array

        args:
            da_turn:
                list[intent, domain, slot, value]
        """
        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.usr_da_array.append(da + '-' + value)

    def _book_rate_goal(self, goal, booked_entity, domains=None):
        """
        judge if the selected entity meets the constraint
        """
        if domains is None:
            domains = belief_domains
        score = []
        for domain in domains:
            if 'book' in goal[domain] and goal[domain]['book']:
                tot = len(goal[domain]['info'].keys())
                if tot == 0:
                    continue
                entity = booked_entity[domain]
                if entity is None:
                    score.append(0)
                    continue
                if domain == 'taxi':
                    score.append(1)
                    continue
                match = 0
                for k, v in goal[domain]['info'].items():
                    if k in ['destination', 'departure', 'name']:
                        tot -= 1
                    elif k == 'leaveAt':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arriveBy':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(
                                entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[k].strip():
                            match += 1
                if tot != 0:
                    score.append(match / tot)
        return score

    def _inform_F1_goal(self, goal, sys_history, domains=None):
        """
        judge if all the requested information is answered
        """
        if domains is None:
            domains = belief_domains
        inform_slot = {}
        for domain in domains:
            inform_slot[domain] = set()
        TP, FP, FN = 0, 0, 0
        
        inform_not_reqt = set()
        reqt_not_inform = set()
        bad_inform = set()

        for da in sys_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and \
                    domain in domains and slot in mapping[domain] and value.strip() not in NUL_VALUE:
                key = mapping[domain][slot]
                if self._check_value(key, value):
                    # print('add key', key)
                    inform_slot[domain].add(key)
                else:
                    bad_inform.add((intent, domain, key))
                    FP += 1
        
        for domain in domains:
            for k in goal[domain]['reqt']:
                if k in inform_slot[domain]:
                    # print('k: ', k)
                    TP += 1
                else:
                    # print('FN + 1')
                    reqt_not_inform.add(('request', domain, k))
                    FN += 1
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal[domain]['reqt'] \
                        and k not in goal[domain]['info'] \
                        and k in requestable[domain]:
                    # print('FP + 1 @2', k)
                    inform_not_reqt.add(('inform', domain, k,))
                    FP += 1
        return TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt

    def _check_value(self, key, value):
        if key == "area":
            return value.lower() in ["centre", "east", "south", "west", "north"]
        elif key == "arriveBy" or key == "leaveAt":
            return time_re.match(value)
        elif key == "day":
            return value.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday",
                              "saturday", "sunday"]
        elif key == "duration":
            return 'minute' in value
        elif key == "internet" or key == "parking":
            return value in ["yes", "no"]
        elif key == "phone":
            return re.match(r'^\d{11}$', value)
        elif key == "price" or key == "entrance fee":
            return 'pound' in value or value == "free" or value == '?'
        elif key == "pricerange":
            return value in ["cheap", "expensive", "moderate", "free"]
        elif key == "postcode":
            return re.match(r'^cb\d{2,3}[a-z]{2}$', value)
        elif key == "stars":
            return re.match(r'^\d$', value)
        elif key == "trainID":
            return re.match(r'^tr\d{4}$', value.lower())
        else:
            return True

    def book_rate(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
        score = self._book_rate_goal(goal, self.booked)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def inform_F1(self, ref2goal=True, aggregate=True):
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)
        TP, FP, FN, _, _, _ = self._inform_F1_goal(goal, self.sys_da_array)
        if aggregate:
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
        else:
            return [TP, FP, FN]

    def task_success(self, ref2goal=True):
        """
        judge if all the domains are successfully completed
        """
        book_sess = self.book_rate(ref2goal)
        inform_sess = self.inform_F1(ref2goal)
        # book rate == 1 & inform recall == 1
        if (book_sess == 1 and inform_sess[1] == 1) \
                or (book_sess == 1 and inform_sess[1] is None) \
                or (book_sess is None and inform_sess[1] == 1):
            return 1
        else:
            return 0

    def domain_reqt_inform_analyze(self, domain, ref2goal=True):
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        return inform
        

    def domain_success(self, domain, ref2goal=True):
        """
        judge if the domain (subtask) is successfully completed
        """
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        book_rate = self._book_rate_goal(goal, self.booked, [domain])
        book_rate = np.mean(book_rate) if book_rate else None

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        try:
            inform_rec = inform[0] / (inform[0] + inform[2])
        except ZeroDivisionError:
            inform_rec = None

        if (book_rate == 1 and inform_rec == 1) \
                or (book_rate == 1 and inform_rec is None) \
                or (book_rate is None and inform_rec == 1):
            return 1
        else:
            return 0
