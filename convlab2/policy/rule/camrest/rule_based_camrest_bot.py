import copy
import json
import random
from copy import deepcopy

from convlab2.policy.policy import Policy
from convlab2.util.camrest.dbquery import Database

# Alphabet used to generate Ref number
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class RuleBasedCamrestBot(Policy):
    ''' Rule-based bot. Implemented for Camrest dataset. '''

    choice = ""

    def __init__(self):
        Policy.__init__(self)
        self.last_state = {}
        self.db = Database()

    def init_session(self):
        self.last_state = {}

    def predict(self, state):
        """
        Args:
            State, please refer to util/state.py
        Output:
            DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
        # print('policy received state: {}'.format(state))

        self.kb_result = []

        DA = {}

        if 'user_action' in state and (len(state['user_action']) > 0):
            user_action = state['user_action']
        else:
            user_action = check_diff(self.last_state, state)

        # Debug info for check_diff function

        last_state_cpy = copy.deepcopy(self.last_state)
        state_cpy = copy.deepcopy(state)

        try:
            del last_state_cpy['history']
        except:
            pass

        try:
            del state_cpy['history']
        except:
            pass
        '''
        if last_state_cpy != state_cpy:
            print("Last state: ", last_state_cpy)
            print("State: ", state_cpy)
            print("Predicted action: ", user_action)
        '''

        self.last_state = state

        for user_act in user_action:
            self._update_DA(user_act, user_action, state, DA)

        # print("Sys action: ", DA)
        for intent in DA:
            if not DA[intent]:
                DA[intent] = [['none', 'none']]

        tuples = []
        for intent, svs in DA.items():
            for slot, value in svs:
                tuples.append([intent, slot, value])
        state['system_action'] = tuples
        return tuples

    def _update_DA(self, user_act, user_action, state, DA):
        """ Answer user's utterance about any domain other than taxi or train. """

        intent_type = user_act

        constraints = state.items()

        kb_result = self.db.query(constraints)
        self.kb_result = deepcopy(kb_result)

        # print("\tConstraint: " + "{}".format(constraints))
        # print("\tCandidate Count: " + "{}".format(len(kb_result)))
        # if len(kb_result) > 0:
        #     print("Candidate: " + "{}".format(kb_result[0]))

        # print(state['user_action'])
        # Respond to user's request
        if intent_type == 'request':
            if ("inform") not in DA:
                DA["inform"] = []
            for slot in user_action[user_act]:
                if len(kb_result) > 0:
                    kb_slot_name = slot[0]
                    if kb_slot_name in kb_result[0]:
                        DA["inform"].append([slot[0], kb_result[0][kb_slot_name]])
                    else:
                        DA["inform"].append([slot[0], "unknown"])
                # DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][slot[0].lower()]])

        else:
            # There's no result matching user's constraint
            # if len(state['kb_results_dict']) == 0:
            if len(kb_result) == 0:
                if ("nooffer") not in DA:
                    DA["nooffer"] = []

                for slot in state['belief_state']:
                    if state['belief_state'][slot] != "" and \
                            state['belief_state'][slot] != "dontcare":
                        slot_name = slot
                        DA["nooffer"].append([slot_name, state['belief_state'][slot]])

            # There's exactly one result matching user's constraint
            # elif len(state['kb_results_dict']) == 1:
            elif len(kb_result) == 1:

                # Inform user about this result
                if ("inform") not in DA:
                    DA["inform"] = []
                props = []
                for prop in state['belief_state']:
                    props.append(prop)
                property_num = len(props)
                if property_num > 0:
                    info_num = random.randint(0, 999999) % property_num + 1
                    random.shuffle(props)
                    for i in range(info_num):
                        slot_name = props[i]
                        # DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][props[i]]])
                        DA["inform"].append([slot_name, kb_result[0][props[i]]])

            # There are multiple resultes matching user's constraint
            else:

                # Recommend a choice from kb_list
                if ("inform") not in DA:
                    DA["inform"] = []

                idx = random.randint(0, 999999) % len(kb_result)
                # idx = 0 
                choice = kb_result[idx]
                props = []
                for prop in choice:
                    props.append([prop, choice[prop]])
                prop_num = min(random.randint(0, 999999) % 3, len(props))
                # prop_num = min(2, len(props))
                random.shuffle(props)
                for i in range(prop_num):
                    slot = props[i][0]
                    string = slot
                    if string not in ['location', 'type', 'id']:
                        DA["inform"].append([string, str(props[i][1])])


def check_diff(last_state, state):
    # print(state)
    user_action = {}
    if last_state == {}:
        for slot in state['belief_state']:
            if state['belief_state'][slot] != "":
                if ("inform") not in user_action:
                    user_action["inform"] = []
                if [slot, state['belief_state'][slot]] \
                        not in user_action["inform"]:
                    user_action["inform"].append([slot, state['belief_state'][slot]])

    else:
        for slot in last_state['belief_state']:
            if last_state['belief_state'][slot] != "":
                if ("inform") not in user_action:
                    user_action["inform"] = []
                if [slot, last_state['belief_state'][slot]] \
                        not in user_action["inform"]:
                    user_action["inform"].append([slot, last_state['belief_state'][slot]])

    return user_action


def deduplicate(lst):
    i = 0
    while i < len(lst):
        if lst[i] in lst[0: i]:
            lst.pop(i)
            i -= 1
        i += 1
    return lst


def generate_ref_num(length):
    """ Generate a ref num for booking. """
    string = ""
    while len(string) < length:
        string += alphabet[random.randint(0, 999999) % 36]
    return string


def fake_state():
    user_action = {
        "inform": [
            [
                "pricerange",
                "moderate"
            ]
        ]
    }
    from convlab2.util.camrest.state import default_state
    init_belief_state = default_state()['belief_state']
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {'user_action': user_action,
             'belief_state': init_belief_state,
             'kb_results_dict': kb_results}
    '''
    state = {'user_action': dict(),
             'belief_state: dict(),
             'kb_results_dict': kb_results
    }
    '''
    return state


def test_init_state():
    user_action = {
        "inform": [
            [
                "pricerange",
                "moderate"
            ]
        ]
    }
    current_slots = dict()
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {'user_action': user_action,
             'current_slots': current_slots,
             'kb_results_dict': []}
    return state


def test_run():
    policy = RuleBasedCamrestBot()
    system_act = policy.predict(fake_state())
    print(json.dumps(system_act, indent=4))
