from tests.nlg.test_nlg import BaseTestMultiwoz, BaseTestTemplateNLG
from convlab2.nlg.template.multiwoz.nlg import TemplateNLG


class TestTemplateMultiwoz(BaseTestMultiwoz, BaseTestTemplateNLG):
    @classmethod
    def setup_class(cls):
        BaseTestMultiwoz.setup_class()
        BaseTestTemplateNLG.setup_class()

    def test_nlg(self):
        self._test_nlg(TemplateNLG)
