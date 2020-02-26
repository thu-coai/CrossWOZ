from convlab2.dst.dst import DST
from convlab2.util.camrest.state import default_state


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``tatk.util.camrest.state.default_state`` returns a default state.
    """
    def __init__(self):
        super().__init__()
        self.state = default_state()

    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.camrest.state.default_state`` returns."""
        self.state = default_state()

    def update(self, user_act=None):
        """
        update belief_state, request_state
        :param user_act:
        :return:
        """
        self.state['user_action'] = user_act
        for intent, slot, value in user_act:
            if intent == "nooffer":
                continue
            elif intent == "inform":
                if slot not in self.state['belief_state']:
                    continue
                self.state['belief_state'][slot] = value
            elif intent == "request":
                self.state['request_state'][slot] = 0
        return self.state


if __name__ == '__main__':
    dst = RuleDST()

    # Action is a dict. Its keys are strings(type) and its values are list of lists.
    # The type may be one of ('inform', 'request', 'nooffer').

    # For example, the action below has a key "inform".
    # Each list in the value of "inform" is a slot-value pair. "name" is slot and "Chiquito Restaurant Bar" is value. "pricerange" is slot and "expensive" is value.
    # Note! Keys and slots must be lowercase.
    action = [['inform', 'name', 'Chiquito Restaurant Bar'],
              ['inform', 'pricerange', 'expensive'],
              ['inform', 'area', 'south'],
              ['inform', 'food', 'mexican']
              ]

    # method `update` updates the attribute `state` of tracker, and returns it.
    state = dst.update(action)
    assert state == dst.state
    assert state == {'user_action': [['inform', 'name', 'Chiquito Restaurant Bar'],
                                     ['inform', 'pricerange', 'expensive'],
                                     ['inform', 'area', 'south'],
                                     ['inform', 'food', 'mexican']],
                     'system_action': [],
                     'belief_state': {'address': '',
                                      'area': 'south',
                                      'food': 'mexican',
                                      'name': 'Chiquito Restaurant Bar',
                                      'phone': '',
                                      'pricerange': 'expensive'},
                     'request_state': {},
                     'terminated': False,
                     'history': []}

    # Please call `init_session` before a new dialog. This initializes the attribute `state` of tracker with a default state, which `tatk.util.camrest.state.default_state` returns. But You needn't call it before the first dialog, because tracker gets a default state in its constructor.
    dst.init_session()
    action = []  # empty dict is OK
    state = dst.update(action)
    assert state == {'user_action': [],
                     'system_action': [],
                     'belief_state': {'address': '',
                                      'area': '',
                                      'food': '',
                                      'name': '',
                                      'phone': '',
                                      'pricerange': ''},
                     'request_state': {},
                     'terminated': False,
                     'history': []}
