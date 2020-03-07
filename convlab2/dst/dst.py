"""Dialog State Tracker Interface"""
from convlab2.util.module import Module


class DST(Module):
    """Base class for dialog state tracker models."""

    def update(self, action):
        """ Update the internal dialog state variable.
        update state['user_action'] with input action

        Args:
            action (str or list of tuples):
                The type is str when DST is word-level (such as NBT), and list of tuples when it is DA-level.
        Returns:
            new_state (dict):
                Updated dialog state, with the same form of previous state.
        """
        pass
