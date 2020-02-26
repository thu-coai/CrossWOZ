"""Natural Language Generation Interface"""
from convlab2.util.module import Module


class NLG(Module):
    """Base class for NLG model."""

    def generate(self, action):
        """Generate a natural language utterance conditioned on the dialog act.
        
        Args:
            action (list of list):
                The dialog action produced by dialog policy module, which is in dialog act format.
        Returns:
            utterance (str):
                A natural langauge utterance.
        """
        return ''
