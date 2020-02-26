from convlab2.nlu import NLU
import pytest


def test_nlu():
    with pytest.raises(TypeError):
        NLU()
    assert hasattr(NLU, 'predict')


class BaseTestNLU:
    """

    Note:
        Instance of BaseTestNLU(or its subclass) should have attribute `nlu`."""
    @staticmethod
    def is_iterable(obj):
        try:
            _ = (_ for _ in obj)
        except TypeError:
            return False
        else:
            return True

    @classmethod
    def setup_class(cls):
        cls.model_urls = {
            "bert_multiwoz_all": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip",
            "bert_multiwoz_sys": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_sys.zip",
            "bert_multiwoz_usr": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_usr.zip",
            "svm_multiwoz_all": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_all.zip",
            "svm_multiwoz_sys": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_sys.zip",
            "svm_multiwoz_usr": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_multiwoz_usr.zip",
            "bert_camrest_all": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_camrest_all.zip",
            "bert_camrest_sys": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_camrest_sys.zip",
            "bert_camrest_usr": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_camrest_usr.zip",
            "svm_camrest_all": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_all.zip",
            "svm_camrest_sys": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_sys.zip",
            "svm_camrest_usr": r"https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_usr.zip",
        }

    def _check_result(self, result):
        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, str)
            assert self.is_iterable(value)
            for item in value:
                assert self.is_iterable(item)
                slot, v = item
                assert isinstance(slot, str)
                assert isinstance(v, str)

    def _test_predict(self, utterances):
        assert hasattr(self, "nlu")
        assert hasattr(self.nlu, "predict")
        for utterance in utterances:
            result = self.nlu.predict(utterance)
            self._check_result(result)


class BaseTestNLUMultiwoz(BaseTestNLU):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.usr_utterances = [
            "I need to book a hotel in the east that has 4 stars .",
            "That does n't matter as long as it has free wifi and parking .",
            "Could you book the Wartworth for one night , 1 person ?",
            "Friday and Can you book it for me and get a reference number ?",
            "I am looking to book a train that is leaving from Cambridge to Bishops Stortford on Friday .",
            "I want to get there by 19:45 at the latest .",
            "Yes please . I also need the travel time , departure time , and price .",
            "Yes . Sorry , but suddenly my plans changed . " +
            "Can you change the Wartworth booking to Monday for 3 people and 4 nights ?",
            "Thank you very much , goodbye .",
            "",
            "测试一下中文",
            "，。：‘.",
        ]
        cls.sys_utterances = [
            "I can help you with that . What is your price range ?",
            "If you 'd like something cheap , I recommend the Allenbell . " +
            "For something moderately priced , I would recommend the Warkworth House .",
            "What day will you be staying ?",
            "Booking was successful . \n Reference number is : BMUKPTG6 .   Can I help you with anything else today ?",
            "There are a number of trains leaving throughout the day .   What time would you like to travel ?",
            "Okay ! The latest train you can take leaves at 17:29 , and arrives by 18:07 . " +
            "Would you like for me to book that for you ?",
            "Reference number is : UIFV8FAS . The price is 10.1 GBP and the trip will take about 38 minutes ." +
            " May I be of any other assistance ?"
            "I have made that change and your reference number is YF86GE4J",
            "You 're welcome . Have a nice day !",
            "",
            "测试一下中文",
            "，。：‘.",
        ]
        cls.all_utterances = cls.usr_utterances + cls.sys_utterances
        import random
        random.shuffle(cls.all_utterances)


class BaseTestNLUCamrest(BaseTestNLU):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        cls.usr_utterances = [
            "I am looking for a restaurant in the north part of town that serves brazilian food.",
            "How about Spanish or Portuguese food?",
            "I would like to try indian food in the north part of town. ",
            "I would like their phone number and address, please.",
            "Thank you goodbye",
            "I want to find a restaurant in the east part of town. It serves chinese food. " +
            "What is the address, phone number, and postcode?",
            "Hi. I'm trying to find a restaurant that serves Venetian food.  I'd like to stay in the west part " +
            "of town.  Can you find me a phone number?",
            "That's okay! How about Indian food? Could I have the address and " +
            "phone number of an Indian food restaurant please. ",
            "",
            "测试中文",
            "，。：‘.",
        ]
        cls.sys_utterances = [
            "There are no mathcing records found for that request",
            "There are no matching records found for that request.",
            "Royal Spice is an Indian restaurant in the north of town, and is in the cheap price range.",
            "Their phone number is 01733 553355 and their address is Victoria Avenue Chesterton C.B 4, 1 E.H",
            "Thank you for using the Cambridge Restaurant Phone System. Enjoy your meal!",
            "Yu Garden is a Chinese restaurant in the east part of town. Their address is 529 Newmarket " +
            "Road Fen Ditton, phone number 01223 248882 and postcode C.B 5, 8 P.A.",
            "I apologize, there are no restaurants that serve venetian food in the west part of town. " +
            "Would you like to try a different part of town?",
            "Cocum is an Indian restaurant located at 71 Castle Street City Centre, " +
            "and it's phone number is 01223 366668.",
            "",
            "测试中文",
            "，。：‘.",
        ]
        cls.all_utterances = cls.usr_utterances + cls.sys_utterances
        import random
        random.shuffle(cls.all_utterances)
