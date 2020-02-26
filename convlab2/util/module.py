"""module interface."""
from abc import ABC


class Module(ABC):

    def train(self, *args, **kwargs):
        """Model training entry point"""
        pass

    def test(self, *args, **kwargs):
        """Model testing entry point"""
        pass

    def from_cache(self, *args, **kwargs):
        """restore internal state for multi-turn dialog"""
        return None

    def to_cache(self, *args, **kwargs):
        """save internal state for multi-turn dialog"""
        return None

    def init_session(self):
        """Init the class variables for a new session."""
        pass
