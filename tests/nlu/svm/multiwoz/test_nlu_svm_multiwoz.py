from convlab2.nlu.svm.multiwoz.nlu import SVMNLU
from tests.nlu.test_nlu import BaseTestNLUMultiwoz


class TestSVMNLU(BaseTestNLUMultiwoz):
    def test_usr(self):
        model_file = self.model_urls['svm_multiwoz_usr']
        self.nlu = SVMNLU('usr', model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        model_file = self.model_urls['svm_multiwoz_sys']
        self.nlu = SVMNLU('sys', model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        model_file = self.model_urls['svm_multiwoz_all']
        self.nlu = SVMNLU('all', model_file)
        super()._test_predict(self.all_utterances)
