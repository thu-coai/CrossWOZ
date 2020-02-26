# -*- coding: utf-8 -*-
"""Vector Interface"""


class Vector():

    def __init__(self):
        pass

    def generate_dict(self):
        """init the dict for mapping state/action into vector"""

    def state_vectorize(self, state):
        """vectorize a state

        Args:
            state (tuple):
                Dialog state
        Returns:
            state_vec (np.array):
                Dialog state vector
        """
        raise NotImplementedError

    def action_devectorize(self, action_vec):
        """recover an action
        
        Args:
            action_vec (np.array):
                Dialog act vector
        Returns:
            action (tuple):
                Dialog act
        """
        raise NotImplementedError
