from tests.nlg.test_nlg import BaseTestTemplateNLG, BaseTestCamrest
from convlab2.nlg.template.camrest.nlg import TemplateNLG


class TestTemplateCamrest(BaseTestTemplateNLG, BaseTestCamrest):
    @classmethod
    def setup_class(cls):
        BaseTestTemplateNLG.setup_class()
        BaseTestCamrest.setup_class()

    def test_nlg(self):
        self._test_nlg(TemplateNLG)
