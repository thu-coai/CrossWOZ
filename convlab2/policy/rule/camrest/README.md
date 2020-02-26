# Rule policy
Rule policy is a rule based **system** dialog policy. It takes a dialog state as input and generates system's dialog act. We implement it on camrest dataset.

# How to use
Example:

```python
from tatk.policy.rule.camrest.rule_based_camrest_bot import RuleBasedCamrestBot
sys_policy = RuleBasedCamrestBot()

# Policy takes dialog state as input. Please refer to tatk.util.camrest.state

state = {'user_action': [['inform', 'name', 'Chiquito Restaurant Bar'],
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

# Please call `init_session` before a new session, this clears policy's history info.
sys_policy.init_session()
    
# method `predict` takes state output from tracker, and generates system's dialog act.
sys_da = sys_policy.predict(state)
```

# Agenda Policy

Agenda policy is a rule based **user** dialog policy. It takes a system's dialog act as input and generates user's dialog act. It maintains a stack-like structure containing the pending user dialogue acts that are needed to elicit the information specified in the goal. We implement it on camrest dataset.

## How to use

```
user_simulator = UserPolicyAgendaCamrest()
user_simulator.init_session()
user_action, session_over = user_simulator.predict({'system_action': sys_action})
```

## Reference

```
@inproceedings{schatzmann2007agenda,
  title={Agenda-based user simulation for bootstrapping a POMDP dialogue system},
  author={Schatzmann, Jost and Thomson, Blaise and Weilhammer, Karl and Ye, Hui and Young, Steve},
  booktitle={Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Companion Volume, Short Papers},
  pages={149--152},
  year={2007},
  organization={Association for Computational Linguistics}
}
```

