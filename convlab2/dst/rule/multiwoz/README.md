# RuleDST
RuleDST is a rule based dialog state tracker which trivially updates new values from NLU result to states.

# How to use
Example:

```python
from tatk.dst.rule.multiwoz import RuleDST


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

```
