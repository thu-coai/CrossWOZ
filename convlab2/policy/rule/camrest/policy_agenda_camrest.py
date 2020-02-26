#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

__time__ = '2019/1/31 10:24'

import json
import os
import random
import logging

from convlab2.policy.policy import Policy
from convlab2.task.camrest.goal_generator import GoalGenerator

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'dontcare'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]


class UserPolicyAgendaCamrest(Policy):
    """ The rule-based user policy model by agenda. Derived from the UserPolicy class """

    def __init__(self):
        """
        Constructor for User_Policy_Agenda class.
        """
        self.max_turn = 40
        self.max_initiative = 4

        self.goal_generator = GoalGenerator(corpus_path=os.path.join(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                'data/camrest/CamRest676_v2.json'))

        self.__turn = 0
        self.goal = None
        self.agenda = None

        Policy.__init__(self)

    def init_session(self):
        """ Build new Goal and Agenda for next session """
        self.__turn = 0
        self.goal = Goal(self.goal_generator)
        self.domain_goals = self.goal.domain_goals
        self.agenda = Agenda(self.goal)

    def predict(self, state):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by user.
        """
        self.__turn += 2

        assert isinstance(state, list)

        sys_action = {}
        for intent, slot, value in state:
            k = intent
            sys_action.setdefault(k, [])
            sys_action[k].append([slot, value])

        if self.__turn > self.max_turn:
            self.agenda.close_session()
        else:
            sys_action = self._transform_sysact_in(sys_action)
            self.agenda.update(sys_action, self.goal)
            if self.goal.task_complete():
                self.agenda.close_session()

        # A -> A' + user_action
        # action = self.agenda.get_action(random.randint(1, self.max_initiative))
        action = self.agenda.get_action(self.max_initiative)

        tuples = []
        for intent, svs in action.items():
            for slot, value in svs:
                tuples.append([intent, slot, value])

        return tuples

    def is_terminated(self):
        # Is there any action to say?
        return self.agenda.is_empty()

    def get_goal(self):
        return self.domain_goals

    def get_reward(self):
        return self._reward()

    def _reward(self):
        """
        Calculate reward based on task completion
        Returns:
            reward (float): Reward given by user.
        """
        if self.goal.task_complete():
            reward = 2.0 * self.max_turn
        elif self.agenda.is_empty():
            reward = -1.0 * self.max_turn
        else:
            reward = -1.0
        return reward

    @classmethod
    def _transform_sysact_in(cls, action):
        new_action = {}
        if not isinstance(action, dict):
            logging.warning('illegal da: {}'.format(action))
            return new_action

        for act in action.keys():
            if not isinstance(act, str):
                logging.warning('illegal act: {}'.format(act))
                continue

            new_action[act] = action[act]

        return new_action


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal_generator: GoalGenerator):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        self.domain_goals = goal_generator.get_user_goal()

        if 'reqt' in self.domain_goals.keys():
            self.domain_goals['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals['reqt']}

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        if 'reqt' in self.domain_goals:
            reqt_vals = self.domain_goals['reqt'].values()
            for val in reqt_vals:
                if val in NOT_SURE_VALS:
                    return False

        return True

    def next_domain_incomplete(self):
        # request
        # reqt
        if 'reqt' in self.domain_goals:
            requests = self.domain_goals['reqt']
            unknow_reqts = [key for (key, val) in requests.items() if val in NOT_SURE_VALS]
            if len(unknow_reqts) > 0:
                return 'reqt', ['name'] if 'name' in unknow_reqts else unknow_reqts

        return None, None

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'


class Agenda(object):
    def __init__(self, goal: Goal):
        """
        Build a new agenda from goal
        Args:
            goal (Goal): User goal.
        """

        def random_sample(data, minimum=0, maximum=1000):
            return random.sample(data, random.randint(min(len(data), minimum), min(len(data), maximum)))

        self.__cur_push_num = 0

        self.__stack = []


        # inform
        if 'info' in goal.domain_goals:
            for slot in random_sample(goal.domain_goals['info'].keys(),
                                      len(goal.domain_goals['info'])):
                self.__push('inform', slot, goal.domain_goals['info'][slot])

    def update(self, sys_action, goal: Goal):
        """
        update Goal by current agent action and current goal. { A' + G" + sys_action -> A" }
        Args:
            sys_action (tuple): Preorder system action.s
            goal (Goal): User Goal
        """
        self.__cur_push_num = 0

        for diaact in sys_action.keys():
            slot_vals = sys_action[diaact]
            if 'nooffer' in diaact:
                if self.update_domain(diaact, slot_vals, goal):
                    return

        for diaact in sys_action.keys():
            if 'nooffer' in diaact:
                continue

            slot_vals = sys_action[diaact]
            if self.update_domain(diaact, slot_vals, goal):
                return

        unk_type, data = goal.next_domain_incomplete()
        if unk_type == 'reqt' and not self._check_reqt_info() and not self._check_reqt():
            for slot in data:
                self._push_item('request', slot, DEF_VAL_UNK)

    def update_domain(self, diaact, slot_vals, goal: Goal):
        """
        Handel Domain-XXX
        :param diaact:      Dial-Act
        :param slot_vals:   slot value pairs
        :param goal:        Goal
        :return:            True:user want to close the session. False:session is continue
        """
        intent = diaact

        g_reqt = goal.domain_goals.get('reqt', dict({}))
        g_info = goal.domain_goals.get('info', dict({}))

        if intent in ['inform']:
            for [slot, value] in slot_vals:
                if slot in g_reqt:
                    if not self._check_reqt_info():
                        self._remove_item('request', slot)
                        if value in NOT_SURE_VALS:
                            g_reqt[slot] = '\"' + value + '\"'
                        else:
                            g_reqt[slot] = value
                else:
                    pass

        elif intent in ['request']:
            for [slot, _] in slot_vals:
                if slot in g_reqt:
                    pass
                else:

                    if random.random() < 0.5:
                        self._push_item('inform', slot, DEF_VAL_DNC)

        elif intent in ['nooffer']:
            if len(g_reqt.keys()) > 0:
                self.close_session()
                return True

        return False

    def close_session(self):
        """ Clear up all actions """
        self.__stack = []

    def get_action(self, initiative=1):
        """
        get multiple acts based on initiative
        Args:
            initiative (int): number of slots , just for 'inform'
        Returns:
            action (dict): user diaact
        """
        diaacts, slots, values = self.__pop(initiative)
        action = {}
        for (diaact, slot, value) in zip(diaacts, slots, values):
            if diaact not in action.keys():
                action[diaact] = []
            action[diaact].append([slot, value])

        return action

    def is_empty(self):
        """
        Is the agenda already empty
        Returns:
            (boolean): True for empty, False for not.
        """
        return len(self.__stack) <= 0

    def _remove_item(self, diaact, slot=DEF_VAL_UNK):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                self.__stack.remove(self.__stack[idx])
                break

    def _push_item(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self._remove_item(diaact, slot)
        self.__push(diaact, slot, value)
        self.__cur_push_num += 1

    def _check_item(self, diaact, slot=None):
        for idx in range(len(self.__stack)):
            if slot is None:
                if self.__stack[idx]['diaact'] == diaact:
                    return True
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    return True
        return False

    def _check_reqt(self):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == 'request':
                return True
        return False

    def _check_reqt_info(self):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == 'inform':
                return True
        return False

    def __check_next_diaact_slot(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact'], self.__stack[-1]['slot']
        return None, None

    def __check_next_diaact(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact']
        return None

    def __push(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self.__stack.append({'diaact': diaact, 'slot': slot, 'value': value})

    def __pop(self, initiative=1):
        diaacts = []
        slots = []
        values = []

        p_diaact, p_slot = self.__check_next_diaact_slot()
        if p_diaact == 'inform':
            for _ in range(10 if self.__cur_push_num == 0 else self.__cur_push_num):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact, next_slot = self.__check_next_diaact_slot()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            next_diaact != 'inform':
                        break
                except:
                    break
        else:
            for _ in range(initiative if self.__cur_push_num == 0 else self.__cur_push_num):
                try:
                    item = self.__stack.pop(-1)
                    diaacts.append(item['diaact'])
                    slots.append(item['slot'])
                    values.append(item['value'])

                    cur_diaact = item['diaact']

                    next_diaact = self.__check_next_diaact()
                    if next_diaact is None or \
                            next_diaact != cur_diaact or \
                            (cur_diaact == 'request' and item['slot'] == 'name'):
                        break
                except:
                    break

        return diaacts, slots, values

    def __str__(self):
        text = '\n-----agenda-----\n'
        text += '<stack top>\n'
        for item in reversed(self.__stack):
            text += str(item) + '\n'
        text += '<stack btm>\n'
        text += '-----agenda-----\n'
        return text


def test():
    user_simulator = UserPolicyAgendaCamrest()
    user_simulator.init_session()

    test_turn(user_simulator, {'inform': [['none', 'none']]})
    test_turn(user_simulator, {'inform': [['pricerange', 'expensive']]})
    test_turn(user_simulator, {'request': [['food', '?'], ['area', '?']]})
    test_turn(user_simulator, {'request': [['area', '?']]})
    test_turn(user_simulator, {})
    test_turn(user_simulator, {"inform": [['phone', '123456789']]})
    test_turn(user_simulator, {"inform": [['address', '987654321']]})

def test_turn(user_simulator, sys_action):
    print('input:', sys_action)
    action, session_over, reward = user_simulator.predict(None, sys_action)
    print('----------------------------------')
    print('sys_action :' + str(sys_action))
    print('user_action:' + str(action))
    print('over       :' + str(session_over))
    print('reward     :' + str(reward))
    print(user_simulator.goal)
    print(user_simulator.agenda)


def test_with_system():
    from convlab2.policy.camrest.rule_based_camrest_bot import RuleBasedCamrestBot, fake_state
    user_simulator = UserPolicyAgendaCamrest()
    user_simulator.init_session()
    state = fake_state()
    system_agent = RuleBasedCamrestBot()
    sys_action = system_agent.predict(state)
    action, session_over, reward = user_simulator.predict(None, sys_action)
    print("Sys:")
    print(json.dumps(sys_action, indent=4))
    print("User:")
    print(json.dumps(action, indent=4))

