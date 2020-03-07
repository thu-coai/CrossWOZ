"""Policy Interface"""
from convlab2.util.module import Module


class Policy(Module):
    """Base class for policy model."""

    def predict(self, state):
        """Predict the next agent action given dialog state.
        update state['system_action'] with predict system action
        
        Args:
            state (tuple or dict):
                when the DST and Policy module are separated, the type of state is tuple.
                else when they are aggregated together, the type of state is dict (dialog act).
        Returns:
            action (list of list):
                The next dialog action.
        """
        return []
