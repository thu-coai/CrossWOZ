def default_state():
    state = dict(user_action=[],
                 system_action=[],
                 belief_state={},
                 request_state={},
                 terminated=False,
                 history=[])
    state['belief_state'] = {'address': '',
                             'area': '',
                             'food': '',
                             'name': '',
                             'phone': '',
                             'pricerange': ''
                             }
    return state
