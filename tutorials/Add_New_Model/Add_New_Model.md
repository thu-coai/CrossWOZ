# Add New Model

- [Add NLU model](#Add-NLU-model)
- [Add DST model](#Add-DST-model)
- [Add Policy model](#Add-Policy-model)
- [Add NLG model](#Add-NLG-model)
- [Add End2End model](#Add-End2End-model)

## Add NLU model

we will take BERTNLU as an example to show how to add new NLU model to **tatk**.

To add this model, you should place the data-independent part under `tatk/tatk/nlu/bert` directory. Those files that are relavant to data should be placed under `tatk/tatk/nlu/bert/camrest`.

### NLU interface

To make the new model consistent with **tatk**, we should follow the NLU interface definition in `tatk/nlu/nlu.py`. The key function is `predict` which takes an utterance(str) as input and return the dialog act. The dialog act format is depended on specific dataset. For camrest dataset, it looks like `{"inform": [["food","brazilian"],["area","north"]]}`


```python
"""Natural language understanding interface."""
from abc import ABCMeta, abstractmethod

class NLU(metaclass=ABCMeta):
    """NLU module interface."""

    @abstractmethod
    def predict(self, utterance):
        """
        Predict the dialog act of a natural language utterance.
        
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        pass
```

### Add New Model

In order to add new Model to **tatk**, we should inherit the `NLU` class above. Here is a piece from BERTNLU. This file should be place under `tatk/tatk/nlu/bert/camrest`. Thus we can use `from tatk.nlu.bert.camrest import BERTNLU` to import the new model.


```python
class BERTNLU(NLU):
    def __init__(self, mode, model_file):
        ## model initialization here, feel free to change the arguments
        self.model = BertNLU()

    def predict(self, utterance):
        return self.model.predict()
```



## Add DST model

we will take RuleDST as an example to show how to add new DST model to **tatk**.

To add this model, you should place the data-independent part under `tatk/tatk/dst/rule` directory. Those files that are relavant to data should be placed under `tatk/tatk/dst/rule/camrest`.

### DST interface

To make the new model consistent with **tatk**, we should follow the DST interface definition in `tatk/dst/state_tracker.py`. The key function is `update` which takes dialog_act(dict) as input, update the `state` attribute and return it. The state format is depended on specific dataset. For camrest dataset, it is defined in `tatk/tatk/util/camrest/state.py`.


```python
class DST(metaclass=ABCMeta):
    """Base class for dialog state tracker models."""

    @abstractmethod
    def update(self, dialog_act):
        """ DST
 pass

    @abstractmethod
    def init_session(self):
        pass
```

### Add New Model

In order to add new Model to **tatk**, we should inherit the `DST` class above. This file should be place under `tatk/tatk/dst/rule/camrest`. Thus we can use `from tatk.dst.rule.camrest import RuleDST` to import the new model.


```python
from tatk.util.camrest.state import default_state

class RuleDST(DST):
    def __init__(self):
        ## model initialization here, feel free to change the arguments
        self.state = default_state()

    def update(self, user_act=None):
        # modify self.state
        return copy.deepcopy(self.state)
    
    def init_session(self):
        """Initialize ``self.state`` with a default state, which ``tatk.util.camrest.state.default_state`` returns."""
        self.state = default_state()
```



## Add Policy Model

we will take Rule policy as an example to show how to add new Policy model to **tatk**.

To add this model, you should place the data-independent part under `tatk/tatk/policy/rule` directory. Those files that are relavant to data should be placed under `tatk/tatk/policy/rule/camrest`.

### DST interface

To make the new model consistent with **tatk**, we should follow the Policy interface definition in `tatk/policy/policy.py`. The key function is `predict` which takes state(dict) as input and outputs dialog act. The state format is depended on specific dataset. For camrest dataset, it is defined in `tatk/tatk/util/camrest/state.py`.


```python
class Policy(metaclass=ABCMeta):
    """Base class for policy model."""

    @abstractmethod
    def predict(self, state):
        """Predict the next agent action given dialog state.
        
        Args:
            state (tuple or dict):
                when the DST and Policy module are separated, the type of state is tuple.
                else when they are aggregated together, the type of state is dict (dialog act).
        Returns:
            action (dict):
                The next dialog action.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Init the class variables for a new session."""
        pass
```

### Add New Model

In order to add new Model to **tatk**, we should inherit the `Policy` class above. This file should be place under `tatk/tatk/policy/rule/camrest`. Thus we can use `from tatk.policy.rule.camrest import Rule` to import the new model.


```python
class Rule(Policy):
    def __init__(self, is_train=False, character='sys'):
        ## model initialization here, feel free to change the arguments
        self.policy = RulePolicy()
        
    def predict(self, state):
        action = self.policy.predict(state)
        return action

    def init_session(self):
        pass
```



## Add NLG Model

we will take TemplateNLG as an example to show how to add new NLG model to **tatk**.

To add this model, you should place the data-independent part under `tatk/tatk/nlg/template_nlg` directory. Those files that are relavant to data should be placed under `tatk/tatk/nlg/template_nlg/camrest`.

### NLG interface

To make the new model consistent with **tatk**, we should follow the NLU interface definition in `tatk/nlg/nlg.py`. The key function is `generate` which takes the dialog act as input and return an utterance(str). The dialog act format is depended on specific dataset. For camrest dataset, it looks like `{"inform": [["food","brazilian"],["area","north"]]}`


```python
class NLG(metaclass=ABCMeta):
    """Base class for NLG model."""

    @abstractmethod
    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (dict):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            response (str):
                A natural langauge utterance.
        """
        pass
```

### Add New Model

In order to add new Model to **tatk**, we should inherit the `NLG` class above. This file should be place under `tatk/tatk/nlg/template_nlg/camrest`. Thus we can use `from tatk.nlg.template_nlg.camrest import TemplateNLG` to import the new model.


```python
class TemplateNLG(NLG):
    def __init__(self, is_user, mode="manual"):
        ## model initialization here, feel free to change the arguments
        self.template = Template(is_user)

    def generate(self, dialog_acts):
        return self.template.generate()
```



## Add End2End Model

we will take Sequicity as an example to show how to add new End-to-End model to **tatk**.

To add this model, you should place the data-independent part under `tatk/tatk/e2e/sequicity` directory. Those files that are relavant to data should be placed under `tatk/tatk/e2e/sequicity/camrest`.

### End2End interface

To make the new model consistent with **tatk**, we should follow the `Agent` interface definition in `tatk/dialog_agent/agent.py`. The key function is `response` which takes an utterance(str) as input and return an utterance(str). 


```python
class Agent(metaclass=ABCMeta):
    """Interface for dialog agent classes."""

    @abstractmethod
    def response(self, observation):
        """Generate agent response given user input.

        The data type of input and response can be either str or dict, condition on the form of agent.

        Example:
            If the agent is a pipeline agent with NLU, DST and Policy, then type(input) == str and
            type(response) == dict.
        Args:
            observation (str or dict):
                The input to the agent.
        Returns:
            response (str or dict):
                The response generated by the agent.
        """
        pass

    @abstractmethod
    def init_session(self):
        """Reset the class variables to prepare for a new session."""
        pass
```

### Add New Model

In order to add new Model to **tatk**, we should inherit the `Agent` class above. This file should be place under `tatk/tatk/e2e/sequicity/camrest`. Thus we can use `from tatk.e2e.sequicity.multiwoz import Sequicity` to import the new model.


```python
class Sequicity(Agent):
    def __init__(self, model_file=None):
        self.init_session()
        
    def response(self, usr):
        return self.generate(usr)
        
    def init_session(self):
        self.belief_span = init()
```
