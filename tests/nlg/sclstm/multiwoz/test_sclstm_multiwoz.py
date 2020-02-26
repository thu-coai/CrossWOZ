from convlab2.nlg.sclstm.multiwoz.sc_lstm import SCLSTM
from tests.nlg.test_nlg import BaseTestMultiwoz, BaseTestSCLSTM


class TestSCLSTMMultiwoz(BaseTestSCLSTM, BaseTestMultiwoz):
    @classmethod
    def setup_class(cls):
        BaseTestSCLSTM.setup_class()
        BaseTestMultiwoz.setup_class()

    def test_nlg(self):
        self._test_nlg(SCLSTM)
