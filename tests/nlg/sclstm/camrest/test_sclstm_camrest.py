from convlab2.nlg.sclstm.camrest.sc_lstm import SCLSTM
from tests.nlg.test_nlg import BaseTestCamrest, BaseTestSCLSTM


class TestSCLSTMMultiwoz(BaseTestSCLSTM, BaseTestCamrest):
    @classmethod
    def setup_class(cls):
        BaseTestSCLSTM.setup_class()
        BaseTestCamrest.setup_class()

    def test_nlg(self):
        self._test_nlg(SCLSTM)
