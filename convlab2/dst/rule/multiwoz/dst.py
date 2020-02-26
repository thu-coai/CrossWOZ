import json
import os

from convlab2.util.multiwoz.state import default_state
from convlab2.dst.rule.multiwoz.dst_util import normalize_value
from convlab2.dst.dst import DST
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``tatk.util.multiwoz.state.default_state`` returns a default state.
        value_dict(dict):
            It helps check whether ``user_act`` has correct content.
    """

    def __init__(self):
        DST.__init__(self)
        self.state = default_state()
        path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        path = os.path.join(path, 'data/multiwoz/value_dict.json')
        self.value_dict = json.load(open(path))

    def update(self, user_act=None):
        """
        update belief_state, request_state
        :param user_act:
        :return:
        """
        self.state['user_action'] = user_act
        for intent, domain, slot, value in user_act:
            domain = domain.lower()
            intent = intent.lower()
            if domain in ['unk', 'general', 'booking']:
                continue
            if intent == 'inform':
                k = REF_SYS_DA[domain.capitalize()].get(slot, slot)
                if k is None:
                    continue
                try:
                    assert domain in self.state['belief_state']
                except:
                    raise Exception('Error: domain <{}> not in new belief state'.format(domain))
                domain_dic = self.state['belief_state'][domain]
                assert 'semi' in domain_dic
                assert 'book' in domain_dic
                if k in domain_dic['semi']:
                    nvalue = normalize_value(self.value_dict, domain, k, value)
                    self.state['belief_state'][domain]['semi'][k] = nvalue
                elif k in domain_dic['book']:
                    self.state['belief_state'][domain]['book'][k] = value
                elif k.lower() in domain_dic['book']:
                    self.state['belief_state'][domain]['book'][k.lower()] = value
                elif k == 'trainID' and domain == 'train':
                    self.state['belief_state'][domain]['book'][k] = normalize_value(self.value_dict, domain, k, value)
                else:
                    # raise Exception('unknown slot name <{}> of domain <{}>'.format(k, domain))
                    with open('unknown_slot.log', 'a+') as f:
                        f.write('unknown slot name <{}> of domain <{}>\n'.format(k, domain))
            elif intent == 'request':
                k = REF_SYS_DA[domain.capitalize()].get(slot, slot)
                if domain not in self.state['request_state']:
                    self.state['request_state'][domain] = {}
                if k not in self.state['request_state'][domain]:
                    self.state['request_state'][domain][k] = 0

        return self.state

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.multiwoz.state.default_state`` returns."""
        self.state = default_state()


if __name__ == '__main__':
    # from tatk.dst.rule.multiwoz import RuleDST

    dst = RuleDST()

    # Action is a dict. Its keys are strings(domain-type pairs, both uppercase and lowercase is OK) and its values are list of lists.
    # The domain may be one of ('Attraction', 'Hospital', 'Booking', 'Hotel', 'Restaurant', 'Taxi', 'Train', 'Police').
    # The type may be "inform" or "request".

    # For example, the action below has a key "Hotel-Inform", in which "Hotel" is domain and "Inform" is action type.
    # Each list in the value of "Hotel-Inform" is a slot-value pair. "Area" is slot and "east" is value. "Star" is slot and "4" is value.
    action = [
        ["Inform", "Hotel", "Area", "east"],
        ["Inform", "Hotel", "Stars", "4"]
    ]

    # method `update` updates the attribute `state` of tracker, and returns it.
    state = dst.update(action)
    assert state == dst.state
    assert state == {'user_action': [["Inform", "Hotel", "Area", "east"], ["Inform", "Hotel", "Stars", "4"]],
                     'system_action': [],
                     'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                                      'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                                'semi': {'name': '',
                                                         'area': 'east',
                                                         'parking': '',
                                                         'pricerange': '',
                                                         'stars': '4',
                                                         'internet': '',
                                                         'type': ''}},
                                      'attraction': {'book': {'booked': []},
                                                     'semi': {'type': '', 'name': '', 'area': ''}},
                                      'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                                     'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                                      'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                                      'taxi': {'book': {'booked': []},
                                               'semi': {'leaveAt': '',
                                                        'destination': '',
                                                        'departure': '',
                                                        'arriveBy': ''}},
                                      'train': {'book': {'booked': [], 'people': ''},
                                                'semi': {'leaveAt': '',
                                                         'destination': '',
                                                         'day': '',
                                                         'arriveBy': '',
                                                         'departure': ''}}},
                     'request_state': {},
                     'terminated': False,
                     'history': []}

    # Please call `init_session` before a new dialog. This initializes the attribute `state` of tracker with a default state, which `tatk.util.multiwoz.state.default_state` returns. But You needn't call it before the first dialog, because tracker gets a default state in its constructor.
    dst.init_session()
    action = [["Inform", "Train", "Arrive", "19:45"]]
    state = dst.update(action)
    assert state == {'user_action': [["Inform", "Train", "Arrive", "19:45"]],
                     'system_action': [],
                     'belief_state': {'police': {'book': {'booked': []}, 'semi': {}},
                                      'hotel': {'book': {'booked': [], 'people': '', 'day': '', 'stay': ''},
                                                'semi': {'name': '',
                                                         'area': '',
                                                         'parking': '',
                                                         'pricerange': '',
                                                         'stars': '',
                                                         'internet': '',
                                                         'type': ''}},
                                      'attraction': {'book': {'booked': []},
                                                     'semi': {'type': '', 'name': '', 'area': ''}},
                                      'restaurant': {'book': {'booked': [], 'people': '', 'day': '', 'time': ''},
                                                     'semi': {'food': '', 'pricerange': '', 'name': '', 'area': ''}},
                                      'hospital': {'book': {'booked': []}, 'semi': {'department': ''}},
                                      'taxi': {'book': {'booked': []},
                                               'semi': {'leaveAt': '',
                                                        'destination': '',
                                                        'departure': '',
                                                        'arriveBy': ''}},
                                      'train': {'book': {'booked': [], 'people': ''},
                                                'semi': {'leaveAt': '',
                                                         'destination': '',
                                                         'day': '',
                                                         'arriveBy': '19:45',
                                                         'departure': ''}}},
                     'request_state': {},
                     'terminated': False,
                     'history': []}
