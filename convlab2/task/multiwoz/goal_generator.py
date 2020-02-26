"""
"""

import json
import os
import pickle
import random
from collections import Counter
from copy import deepcopy

import numpy as np

from convlab2.util.multiwoz.dbquery import Database

domains = {'attraction', 'hotel', 'restaurant', 'train', 'taxi', 'hospital', 'police'}
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
request_slot_string_map = {
    'phone': 'phone number',
    'pricerange': 'price range',
    'duration': 'travel time',
    'arriveBy': 'arrival time',
    'leaveAt': 'departure time',
    'trainID': 'train ID'
}
templates = {
    'intro': 'You are looking for information in Cambridge.',
    'restaurant': {
        'intro': 'You are looking forward to trying local restaurants.',
        'request': 'Once you find a restaurnat, make sure you get {}.',
        'area': 'The restaurant should be in the {}.',
        'food': 'The restaurant should serve {} food.',
        'name': 'You are looking for a particular restaurant. Its name is called {}.',
        'pricerange': 'The restaurant should be in the {} price range.',
        'book': 'Once you find the restaurant you want to book a table {}.',
        'fail_info food': 'If there is no such restaurant, how about one that serves {} food.',
        'fail_info area': 'If there is no such restaurant, how about one in the {} area.',
        'fail_info pricerange': 'If there is no such restaurant, how about one in the {} price range.',
        'fail_book time': 'If the booking fails how about {}.',
        'fail_book day': 'If the booking fails how about {}.'
    },
    'hotel': {
        'intro': 'You are looking for a place to stay.',
        'request': 'Once you find a hotel, make sure you get {}.',
        'stars': 'The hotel should have a star of {}.',
        'area': 'The hotel should be in the {}.',
        'type': 'The hotel should be in the type of {}.',
        'pricerange': 'The hotel should be in the {} price range.',
        'name': 'You are looking for a particular hotel. Its name is called {}.',
        'internet yes': 'The hotel should include free wifi.',
        'internet no': 'The hotel does not need to include free wifi.',
        'parking yes': 'The hotel should include free parking.',
        'parking no': 'The hotel does not need to include free parking.',
        'book': 'Once you find the hotel you want to book it {}.',
        'fail_info type': 'If there is no such hotel, how about one that is in the type of {}.',
        'fail_info area': 'If there is no such hotel, how about one that is in the {} area.',
        'fail_info stars': 'If there is no such hotel, how about one that has a star of {}.',
        'fail_info pricerange': 'If there is no such hotel, how about one that is in the {} price range.',
        'fail_info parking yes': 'If there is no such hotel, how about one that has free parking.',
        'fail_info parking no': 'If there is no such hotel, how about one that does not has free parking.',
        'fail_info internet yes': 'If there is no such hotel, how about one that has free wifi.',
        'fail_info internet no': 'If there is no such hotel, how about one that does not has free wifi.',
        'fail_book stay': 'If the booking fails how about {} nights.',
        'fail_book day': 'If the booking fails how about {}.'
    },
    'attraction': {
        'intro': 'You are excited about seeing local tourist attractions.',
        'request': 'Once you find an attraction, make sure you get {}.',
        'area': 'The attraction should be in the {}.',
        'type': 'The attraction should be in the type of {}.',
        'name': 'You are looking for a particular attraction. Its name is called {}.',
        'fail_info type': 'If there is no such attraction, how about one that is in the type of {}.',
        'fail_info area': 'If there is no such attraction, how about one in the {} area.'
    },
    'taxi': {
        'intro': 'You are also looking for a taxi.',
        'commute': 'You also want to book a taxi to commute between the two places.',
        'restaurant': 'You want to make sure it arrives the restaurant by the booked time.',
        'request': 'Once you find a taxi, make sure you get {}.',
        'departure': 'The taxi should depart from {}.',
        'destination': 'The taxi should go to {}.',
        'leaveAt': 'The taxi should leave after {}.',
        'arriveBy': 'The taxi should arrive by {}.'
    },
    'train': {
        'intro': 'You are also looking for a train.',
        'request': 'Once you find a train, make sure you get {}.',
        'departure': 'The train should depart from {}.',
        'destination': 'The train should go to {}.',
        'day': 'The train should leave on {}.',
        'leaveAt': 'The train should leave after {}.',
        'arriveBy': 'The train should arrive by {}.',
        'book': 'Once you find the train you want to make a booking {}.'
    },
    'police': {
        'intro': 'You were robbed and are looking for help.',
        'request': 'Make sure you get {}.'
    },
    'hospital': {
        'intro': 'You got injured and are looking for a hospital nearby',
        'request': 'Make sure you get {}.',
        'department': 'The hospital should have the {} department.'
    }
}

pro_correction = {
    # "info": 0.2,
    "info": 0.0,
    # "reqt": 0.2,
    "reqt": 0.0,
    # "book": 0.2
    "book": 0.0
}


def null_boldify(content):
    return content

def do_boldify(content):
    return '<b>' + content + '</b>'

def nomial_sample(counter: Counter):
    return list(counter.keys())[np.argmax(np.random.multinomial(1, list(counter.values())))]

class GoalGenerator:
    """User goal generator."""

    def __init__(self,
                 goal_model_path=os.path.join(
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                     'data/multiwoz/goal/goal_model.pkl'),
                 corpus_path=None,
                 boldify=False):
        """
        Args:
            goal_model_path: path to a goal model 
            corpus_path: path to a dialog corpus to build a goal model 
        """
        self.goal_model_path = goal_model_path
        self.corpus_path = corpus_path
        self.db = Database()
        self.boldify = do_boldify if boldify else null_boldify
        if os.path.exists(self.goal_model_path):
            self.ind_slot_dist, self.ind_slot_value_dist, self.domain_ordering_dist, self.book_dist = pickle.load(
                open(self.goal_model_path, 'rb'))
            print('Loading goal model is done')
        else:
            self._build_goal_model()
            print('Building goal model is done')

        # remove some slot
        del self.ind_slot_dist['police']['reqt']['postcode']
        del self.ind_slot_value_dist['police']['reqt']['postcode']
        del self.ind_slot_dist['hospital']['reqt']['postcode']
        del self.ind_slot_value_dist['hospital']['reqt']['postcode']
        del self.ind_slot_dist['hospital']['reqt']['address']
        del self.ind_slot_value_dist['hospital']['reqt']['address']

    def _build_goal_model(self):
        dialogs = json.load(open(self.corpus_path))

        # domain ordering
        def _get_dialog_domains(dialog):
            return list(filter(lambda x: x in domains and len(dialog['goal'][x]) > 0, dialog['goal']))

        domain_orderings = []
        for d in dialogs:
            d_domains = _get_dialog_domains(dialogs[d])
            first_index = []
            for domain in d_domains:
                message = [dialogs[d]['goal']['message']] if type(dialogs[d]['goal']['message']) == str else \
                dialogs[d]['goal']['message']
                for i, m in enumerate(message):
                    if domain_keywords[domain].lower() in m.lower() or domain.lower() in m.lower():
                        first_index.append(i)
                        break
            domain_orderings.append(tuple(map(lambda x: x[1], sorted(zip(first_index, d_domains), key=lambda x: x[0]))))
        domain_ordering_cnt = Counter(domain_orderings)
        self.domain_ordering_dist = deepcopy(domain_ordering_cnt)
        for order in domain_ordering_cnt.keys():
            self.domain_ordering_dist[order] = domain_ordering_cnt[order] / sum(domain_ordering_cnt.values())

        # independent goal slot distribution
        ind_slot_value_cnt = dict([(domain, {}) for domain in domains])
        domain_cnt = Counter()
        book_cnt = Counter()

        for d in dialogs:
            for domain in domains:
                if dialogs[d]['goal'][domain] != {}:
                    domain_cnt[domain] += 1
                if 'info' in dialogs[d]['goal'][domain]:
                    for slot in dialogs[d]['goal'][domain]['info']:
                        if 'invalid' in slot:
                            continue
                        if 'info' not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]['info'] = {}
                        if slot not in ind_slot_value_cnt[domain]['info']:
                            ind_slot_value_cnt[domain]['info'][slot] = Counter()
                        if 'care' in dialogs[d]['goal'][domain]['info'][slot]:
                            continue
                        ind_slot_value_cnt[domain]['info'][slot][dialogs[d]['goal'][domain]['info'][slot]] += 1
                if 'reqt' in dialogs[d]['goal'][domain]:
                    for slot in dialogs[d]['goal'][domain]['reqt']:
                        if 'reqt' not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]['reqt'] = Counter()
                        ind_slot_value_cnt[domain]['reqt'][slot] += 1
                if 'book' in dialogs[d]['goal'][domain]:
                    book_cnt[domain] += 1
                    for slot in dialogs[d]['goal'][domain]['book']:
                        if 'invalid' in slot:
                            continue
                        if 'book' not in ind_slot_value_cnt[domain]:
                            ind_slot_value_cnt[domain]['book'] = {}
                        if slot not in ind_slot_value_cnt[domain]['book']:
                            ind_slot_value_cnt[domain]['book'][slot] = Counter()
                        if 'care' in dialogs[d]['goal'][domain]['book'][slot]:
                            continue
                        ind_slot_value_cnt[domain]['book'][slot][dialogs[d]['goal'][domain]['book'][slot]] += 1

        self.ind_slot_value_dist = deepcopy(ind_slot_value_cnt)
        self.ind_slot_dist = dict([(domain, {}) for domain in domains])
        self.book_dist = {}
        for domain in domains:
            if 'info' in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]['info']:
                    if 'info' not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]['info'] = {}
                    if slot not in self.ind_slot_dist[domain]['info']:
                        self.ind_slot_dist[domain]['info'][slot] = {}
                    self.ind_slot_dist[domain]['info'][slot] = sum(ind_slot_value_cnt[domain]['info'][slot].values()) / \
                                                               domain_cnt[domain]
                    slot_total = sum(ind_slot_value_cnt[domain]['info'][slot].values())
                    for val in self.ind_slot_value_dist[domain]['info'][slot]:
                        self.ind_slot_value_dist[domain]['info'][slot][val] = ind_slot_value_cnt[domain]['info'][slot][
                                                                                  val] / slot_total
            if 'reqt' in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]['reqt']:
                    if 'reqt' not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]['reqt'] = {}
                    self.ind_slot_dist[domain]['reqt'][slot] = ind_slot_value_cnt[domain]['reqt'][slot] / domain_cnt[
                        domain]
                    self.ind_slot_value_dist[domain]['reqt'][slot] = ind_slot_value_cnt[domain]['reqt'][slot] / \
                                                                     domain_cnt[domain]
            if 'book' in ind_slot_value_cnt[domain]:
                for slot in ind_slot_value_cnt[domain]['book']:
                    if 'book' not in self.ind_slot_dist[domain]:
                        self.ind_slot_dist[domain]['book'] = {}
                    if slot not in self.ind_slot_dist[domain]['book']:
                        self.ind_slot_dist[domain]['book'][slot] = {}
                    self.ind_slot_dist[domain]['book'][slot] = sum(ind_slot_value_cnt[domain]['book'][slot].values()) / \
                                                               domain_cnt[domain]
                    slot_total = sum(ind_slot_value_cnt[domain]['book'][slot].values())
                    for val in self.ind_slot_value_dist[domain]['book'][slot]:
                        self.ind_slot_value_dist[domain]['book'][slot][val] = ind_slot_value_cnt[domain]['book'][slot][
                                                                                  val] / slot_total
            self.book_dist[domain] = book_cnt[domain] / len(dialogs)

        pickle.dump((self.ind_slot_dist, self.ind_slot_value_dist, self.domain_ordering_dist, self.book_dist),
                    open(self.goal_model_path, 'wb'))

    def _get_domain_goal(self, domain):
        cnt_slot = self.ind_slot_dist[domain]
        cnt_slot_value = self.ind_slot_value_dist[domain]
        pro_book = self.book_dist[domain]

        while True:
            # domain_goal = defaultdict(lambda: {})
            # domain_goal = {'info': {}, 'fail_info': {}, 'reqt': {}, 'book': {}, 'fail_book': {}}
            domain_goal = {'info': {}}
            # inform
            if 'info' in cnt_slot:
                for slot in cnt_slot['info']:
                    if random.random() < cnt_slot['info'][slot] + pro_correction['info']:
                        domain_goal['info'][slot] = nomial_sample(cnt_slot_value['info'][slot])

                if domain in ['hotel', 'restaurant', 'attraction'] and 'name' in domain_goal['info'] and len(
                        domain_goal['info']) > 1:
                    if random.random() < cnt_slot['info']['name']:
                        domain_goal['info'] = {'name': domain_goal['info']['name']}
                    else:
                        del domain_goal['info']['name']

                if domain in ['taxi', 'train'] and 'arriveBy' in domain_goal['info'] and 'leaveAt' in domain_goal[
                    'info']:
                    if random.random() < (
                            cnt_slot['info']['leaveAt'] / (cnt_slot['info']['arriveBy'] + cnt_slot['info']['leaveAt'])):
                        del domain_goal['info']['arriveBy']
                    else:
                        del domain_goal['info']['leaveAt']

                if domain in ['taxi', 'train'] and 'arriveBy' not in domain_goal['info'] and 'leaveAt' not in \
                        domain_goal['info']:
                    if random.random() < (cnt_slot['info']['arriveBy'] / (
                            cnt_slot['info']['arriveBy'] + cnt_slot['info']['leaveAt'])):
                        domain_goal['info']['arriveBy'] = nomial_sample(cnt_slot_value['info']['arriveBy'])
                    else:
                        domain_goal['info']['leaveAt'] = nomial_sample(cnt_slot_value['info']['leaveAt'])

                if domain in ['taxi', 'train'] and 'departure' not in domain_goal['info']:
                    domain_goal['info']['departure'] = nomial_sample(cnt_slot_value['info']['departure'])

                if domain in ['taxi', 'train'] and 'destination' not in domain_goal['info']:
                    domain_goal['info']['destination'] = nomial_sample(cnt_slot_value['info']['destination'])

                if domain in ['taxi', 'train'] and \
                        'departure' in domain_goal['info'] and \
                        'destination' in domain_goal['info'] and \
                        domain_goal['info']['departure'] == domain_goal['info']['destination']:
                    if random.random() < (cnt_slot['info']['departure'] / (
                            cnt_slot['info']['departure'] + cnt_slot['info']['destination'])):
                        domain_goal['info']['departure'] = nomial_sample(cnt_slot_value['info']['departure'])
                    else:
                        domain_goal['info']['destination'] = nomial_sample(cnt_slot_value['info']['destination'])
                if domain_goal['info'] == {}:
                    continue
            # request
            if 'reqt' in cnt_slot:
                reqt = [slot for slot in cnt_slot['reqt']
                        if random.random() < cnt_slot['reqt'][slot] + pro_correction['reqt'] and slot not in
                        domain_goal['info']]
                if len(reqt) > 0:
                    domain_goal['reqt'] = reqt

            # book
            if 'book' in cnt_slot and random.random() < pro_book + pro_correction['book']:
                if 'book' not in domain_goal:
                    domain_goal['book'] = {}

                for slot in cnt_slot['book']:
                    if random.random() < cnt_slot['book'][slot] + pro_correction['book']:
                        domain_goal['book'][slot] = nomial_sample(cnt_slot_value['book'][slot])

                # makes sure that there are all necessary slots for booking
                if domain == 'restaurant' and 'time' not in domain_goal['book']:
                    domain_goal['book']['time'] = nomial_sample(cnt_slot_value['book']['time'])

                if domain == 'hotel' and 'stay' not in domain_goal['book']:
                    domain_goal['book']['stay'] = nomial_sample(cnt_slot_value['book']['stay'])

                if domain in ['hotel', 'restaurant'] and 'day' not in domain_goal['book']:
                    domain_goal['book']['day'] = nomial_sample(cnt_slot_value['book']['day'])

                if domain in ['hotel', 'restaurant'] and 'people' not in domain_goal['book']:
                    domain_goal['book']['people'] = nomial_sample(cnt_slot_value['book']['people'])

                if domain == 'train' and len(domain_goal['book']) <= 0:
                    domain_goal['book']['people'] = nomial_sample(cnt_slot_value['book']['people'])

            # fail_book
            if 'book' in domain_goal and random.random() < 0.5:
                if domain == 'hotel':
                    domain_goal['fail_book'] = deepcopy(domain_goal['book'])
                    if 'stay' in domain_goal['book'] and random.random() < 0.5:
                        # increase hotel-stay
                        domain_goal['fail_book']['stay'] = str(int(domain_goal['book']['stay']) + 1)
                    elif 'day' in domain_goal['book']:
                        # push back hotel-day by a day
                        domain_goal['fail_book']['day'] = days[(days.index(domain_goal['book']['day']) - 1) % 7]

                elif domain == 'restaurant':
                    domain_goal['fail_book'] = deepcopy(domain_goal['book'])
                    if 'time' in domain_goal['book'] and random.random() < 0.5:
                        hour, minute = domain_goal['book']['time'].split(':')
                        domain_goal['fail_book']['time'] = str((int(hour) + 1) % 24) + ':' + minute
                    elif 'day' in domain_goal['book']:
                        if random.random() < 0.5:
                            domain_goal['fail_book']['day'] = days[(days.index(domain_goal['book']['day']) - 1) % 7]
                        else:
                            domain_goal['fail_book']['day'] = days[(days.index(domain_goal['book']['day']) + 1) % 7]

            # fail_info
            if 'info' in domain_goal and len(self.db.query(domain, domain_goal['info'].items())) == 0:
                num_trial = 0
                while num_trial < 100:
                    adjusted_info = self._adjust_info(domain, domain_goal['info'])
                    if len(self.db.query(domain, adjusted_info.items())) > 0:
                        if domain == 'train':
                            domain_goal['info'] = adjusted_info
                        else:
                            domain_goal['fail_info'] = domain_goal['info']
                            domain_goal['info'] = adjusted_info

                        break
                    num_trial += 1

                if num_trial >= 100:
                    continue

            # at least there is one request and book
            if 'reqt' in domain_goal or 'book' in domain_goal:
                break

        return domain_goal

    def get_user_goal(self):
        domain_ordering = ()
        while len(domain_ordering) <= 0:
            domain_ordering = nomial_sample(self.domain_ordering_dist)
        # domain_ordering = ('restaurant',)

        user_goal = {dom: self._get_domain_goal(dom) for dom in domain_ordering}
        assert len(user_goal.keys()) > 0

        # using taxi to communte between places, removing destination and departure.
        if 'taxi' in domain_ordering:
            places = [dom for dom in domain_ordering[: domain_ordering.index('taxi')] if 'address' in self.ind_slot_dist[dom]['reqt'].keys()]
            if len(places) >= 1:
                del user_goal['taxi']['info']['destination']
                user_goal[places[-1]]['reqt'] = list(set(user_goal[places[-1]].get('reqt', [])).union({'address'}))
                if places[-1] == 'restaurant' and 'book' in user_goal['restaurant']:
                    user_goal['taxi']['info']['arriveBy'] = user_goal['restaurant']['book']['time']
                    if 'leaveAt' in user_goal['taxi']['info']:
                        del user_goal['taxi']['info']['leaveAt']
            if len(places) >= 2:
                del user_goal['taxi']['info']['departure']
                user_goal[places[-2]]['reqt'] = list(set(user_goal[places[-2]].get('reqt', [])).union({'address'}))

        # match area of attraction and restaurant
        if 'restaurant' in domain_ordering and \
                'attraction' in domain_ordering and \
                'fail_info' not in user_goal['restaurant'] and \
                domain_ordering.index('restaurant') > domain_ordering.index('attraction') and \
                'area' in user_goal['restaurant']['info'] and 'area' in user_goal['attraction']['info']:
            adjusted_restaurant_goal = deepcopy(user_goal['restaurant']['info'])
            adjusted_restaurant_goal['area'] = user_goal['attraction']['info']['area']
            if len(self.db.query('restaurant', adjusted_restaurant_goal.items())) > 0 and random.random() < 0.5:
                user_goal['restaurant']['info']['area'] = user_goal['attraction']['info']['area']

        # match day and people of restaurant and hotel
        if 'restaurant' in domain_ordering and 'hotel' in domain_ordering and \
                'book' in user_goal['restaurant'] and 'book' in user_goal['hotel']:
            if random.random() < 0.5:
                user_goal['restaurant']['book']['people'] = user_goal['hotel']['book']['people']
                if 'fail_book' in user_goal['restaurant']:
                    user_goal['restaurant']['fail_book']['people'] = user_goal['hotel']['book']['people']
            if random.random() < 1.0:
                user_goal['restaurant']['book']['day'] = user_goal['hotel']['book']['day']
                if 'fail_book' in user_goal['restaurant']:
                    user_goal['restaurant']['fail_book']['day'] = user_goal['hotel']['book']['day']
                    if user_goal['restaurant']['book']['day'] == user_goal['restaurant']['fail_book']['day'] and \
                            user_goal['restaurant']['book']['time'] == user_goal['restaurant']['fail_book']['time'] and \
                            user_goal['restaurant']['book']['people'] == user_goal['restaurant']['fail_book']['people']:
                        del user_goal['restaurant']['fail_book']

        # match day and people of hotel and train
        if 'hotel' in domain_ordering and 'train' in domain_ordering and \
                'book' in user_goal['hotel'] and 'info' in user_goal['train']:
            if user_goal['train']['info']['destination'] == 'cambridge' and \
                'day' in user_goal['hotel']['book']:
                user_goal['train']['info']['day'] = user_goal['hotel']['book']['day']
            elif user_goal['train']['info']['departure'] == 'cambridge' and \
                'day' in user_goal['hotel']['book'] and 'stay' in user_goal['hotel']['book']:
                user_goal['train']['info']['day'] = days[
                    (days.index(user_goal['hotel']['book']['day']) + int(
                        user_goal['hotel']['book']['stay'])) % 7]
            # In case, we have no query results with adjusted train goal, we simply drop the train goal.
            if len(self.db.query('train', user_goal['train']['info'].items())) == 0:
                del user_goal['train']
                domain_ordering = tuple(list(domain_ordering).remove('train'))

        for domain in user_goal:
            if not user_goal[domain]['info']:
                user_goal[domain]['info'] = {'none':'none'}

        user_goal['domain_ordering'] = domain_ordering

        return user_goal

    def _adjust_info(self, domain, info):
        # adjust one of the slots of the info
        adjusted_info = deepcopy(info)
        slot = random.choice(list(info.keys()))
        adjusted_info[slot] = random.choice(list(self.ind_slot_value_dist[domain]['info'][slot].keys()))
        return adjusted_info

    def build_message(self, user_goal, boldify=null_boldify):
        message = []
        state = deepcopy(user_goal)

        for dom in user_goal['domain_ordering']:
            dom_msg = []
            state = deepcopy(user_goal[dom])
            num_acts_in_unit = 0

            if not (dom == 'taxi' and len(state['info']) == 1):
                # intro
                m = [templates[dom]['intro']]

            # info
            def fill_info_template(user_goal, domain, slot, info):
                if slot != 'area' or not ('restaurant' in user_goal and
                                          'attraction' in user_goal and
                                          info in user_goal['restaurant'].keys() and
                                          info in user_goal['attraction'].keys() and
                                          'area' in user_goal['restaurant'][info] and
                                          'area' in user_goal['attraction'][info] and
                                          user_goal['restaurant'][info]['area'] == user_goal['attraction'][info]['area']):
                    return templates[domain][slot].format(self.boldify(user_goal[domain][info][slot]))
                else:
                    restaurant_index = user_goal['domain_ordering'].index('restaurant')
                    attraction_index = user_goal['domain_ordering'].index('attraction')
                    if restaurant_index > attraction_index and domain == 'restaurant':
                        return templates[domain][slot].format(self.boldify('same area as the attraction'))
                    elif attraction_index > restaurant_index and domain == 'attraction':
                        return templates[domain][slot].format(self.boldify('same area as the restaurant'))
                return templates[domain][slot].format(self.boldify(user_goal[domain][info][slot]))

            info = 'info'
            if 'fail_info' in user_goal[dom]:
                info = 'fail_info'
            if dom == 'taxi' and len(state[info]) == 1:
                taxi_index = user_goal['domain_ordering'].index('taxi')
                places = [dom for dom in user_goal['domain_ordering'][: taxi_index] if
                          dom in ['attraction', 'hotel', 'restaurant']]
                if len(places) >= 2:
                    random.shuffle(places)
                    m.append(templates['taxi']['commute'])
                    if 'arriveBy' in state[info]:
                        m.append('The taxi should arrive at the {} from the {} by {}.'.format(self.boldify(places[0]),
                                                                                              self.boldify(places[1]),
                                                                                              self.boldify(state[info]['arriveBy'])))
                    elif 'leaveAt' in state[info]:
                        m.append('The taxi should leave from the {} to the {} after {}.'.format(self.boldify(places[0]),
                                                                                                self.boldify(places[1]),
                                                                                                self.boldify(state[info]['leaveAt'])))
                    message.append(' '.join(m))
            else:
                while len(state[info]) > 0:
                    num_acts = random.randint(1, min(len(state[info]), 3))
                    slots = random.sample(list(state[info].keys()), num_acts)
                    sents = [fill_info_template(user_goal, dom, slot, info) for slot in slots if slot not in ['parking', 'internet']]
                    if 'parking' in slots:
                        sents.append(templates[dom]['parking ' + state[info]['parking']])
                    if 'internet' in slots:
                        sents.append(templates[dom]['internet ' + state[info]['internet']])
                    m.extend(sents)
                    message.append(' '.join(m))
                    m = []
                    for slot in slots:
                        del state[info][slot]

            # fail_info
            if 'fail_info' in user_goal[dom]:
            # if 'fail_info' in user_goal[dom]:
                adjusted_slot = list(filter(lambda x: x[0][1] != x[1][1],
                                            zip(user_goal[dom]['info'].items(), user_goal[dom]['fail_info'].items())))[0][0][0]
                if adjusted_slot in ['internet', 'parking']:
                    message.append(templates[dom]['fail_info ' + adjusted_slot + ' ' + user_goal[dom]['info'][adjusted_slot]])
                else:
                    message.append(templates[dom]['fail_info ' + adjusted_slot].format(self.boldify(user_goal[dom]['info'][adjusted_slot])))

            # reqt
            if 'reqt' in state:
                slot_strings = []
                for slot in state['reqt']:
                    if slot in ['internet', 'parking', 'food']:
                        continue
                    slot_strings.append(slot if slot not in request_slot_string_map else request_slot_string_map[slot])
                if len(slot_strings) > 0:
                    message.append(templates[dom]['request'].format(self.boldify(', '.join(slot_strings))))
                if 'internet' in state['reqt']:
                    message.append('Make sure to ask if the hotel includes free wifi.')
                if 'parking' in state['reqt']:
                    message.append('Make sure to ask if the hotel includes free parking.')
                if 'food' in state['reqt']:
                    message.append('Make sure to ask about what food it serves.')

            def get_same_people_domain(user_goal, domain, slot):
                if slot not in ['day', 'people']:
                    return None
                domain_index = user_goal['domain_ordering'].index(domain)
                previous_domains = user_goal['domain_ordering'][:domain_index]
                for prev in previous_domains:
                    if prev in ['restaurant', 'hotel', 'train'] and 'book' in user_goal[prev] and \
                            slot in user_goal[prev]['book'] and user_goal[prev]['book'][slot] == \
                            user_goal[domain]['book'][slot]:
                        return prev
                return None

            # book
            book = 'book'
            if 'fail_book' in user_goal[dom]:
                book = 'fail_book'
            if 'book' in state:
                slot_strings = []
                for slot in ['people', 'time', 'day', 'stay']:
                    if slot in state[book]:
                        if slot == 'people':
                            same_people_domain = get_same_people_domain(user_goal, dom, slot)
                            if same_people_domain is None:
                                slot_strings.append('for {} people'.format(self.boldify(state[book][slot])))
                            else:
                                slot_strings.append(self.boldify(
                                    'for the same group of people as the {} booking'.format(same_people_domain)))
                        elif slot == 'time':
                            slot_strings.append('at {}'.format(self.boldify(state[book][slot])))
                        elif slot == 'day':
                            same_people_domain = get_same_people_domain(user_goal, dom, slot)
                            if same_people_domain is None:
                                slot_strings.append('on {}'.format(self.boldify(state[book][slot])))
                            else:
                                slot_strings.append(
                                    self.boldify('on the same day as the {} booking'.format(same_people_domain)))
                        elif slot == 'stay':
                            slot_strings.append('for {} nights'.format(self.boldify(state[book][slot])))
                        del state[book][slot]

                assert len(state[book]) <= 0, state[book]

                if len(slot_strings) > 0:
                    message.append(templates[dom]['book'].format(' '.join(slot_strings)))

            # fail_book
            if 'fail_book' in user_goal[dom]:
                adjusted_slot = list(filter(lambda x: x[0][1] != x[1][1], zip(user_goal[dom]['book'].items(),
                                                                              user_goal[dom]['fail_book'].items())))[0][0][0]

                if adjusted_slot in ['internet', 'parking']:
                    message.append(
                        templates[dom]['fail_book ' + adjusted_slot + ' ' + user_goal[dom]['book'][adjusted_slot]])
                else:
                    message.append(templates[dom]['fail_book ' + adjusted_slot].format(
                        self.boldify(user_goal[dom]['book'][adjusted_slot])))

        if boldify == do_boldify:
            for i, m in enumerate(message):
                message[i] = message[i].replace('wifi', "<b>wifi</b>")
                message[i] = message[i].replace('internet', "<b>internet</b>")
                message[i] = message[i].replace('parking', "<b>parking</b>")

        return message

