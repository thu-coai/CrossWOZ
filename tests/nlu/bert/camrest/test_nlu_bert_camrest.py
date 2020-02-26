import subprocess
from convlab2.nlu.bert.camrest.nlu import BERTNLU
from tests.nlu.test_nlu import BaseTestNLUCamrest


class TestBERTNLU(BaseTestNLUCamrest):
    def test_usr(self):
        assert 0 == subprocess.call('cd tatk/nlu/bert/camrest && python preprocess.py {mode}'.format(mode='usr'),
                                    shell=True)
        model_file = self.model_urls['bert_camrest_usr']
        self.nlu = BERTNLU('usr', model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        assert 0 == subprocess.call('cd tatk/nlu/bert/camrest && python preprocess.py {mode}'.format(mode="sys"),
                                    shell=True)
        model_file = self.model_urls['bert_camrest_sys']
        self.nlu = BERTNLU('sys', model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        assert 0 == subprocess.call('cd tatk/nlu/bert/camrest && python preprocess.py {mode}'.format(mode="all"),
                                    shell=True)
        model_file = self.model_urls['bert_camrest_all']
        self.nlu = BERTNLU('all', model_file)
        super()._test_predict(self.all_utterances)
