import pytest
from convlab2.nlg.nlg import NLG


def test_nlg():
    with pytest.raises(TypeError):
        NLG()
    assert hasattr(NLG, 'generate')


class BaseTestNLG:
    """Base class for testing nlg."""

    @staticmethod
    def _test_generate(nlg, acts):
        assert hasattr(nlg, 'generate')
        for act in acts:
            result = nlg.generate(act)
            assert isinstance(result, str)

    @classmethod
    def setup_class(cls):
        pass


class BaseTestTemplateNLG(BaseTestNLG):
    """Base class for testing template nlg.

    Note:
        BaseTestTemplateNLG or its subclass should have attributes(instance attributes or class attributes):
            `usr_acts` and `sys_acts`
    """

    @classmethod
    def setup_class(cls):
        cls.modes = 'auto', 'manual', 'auto_manual'

    def _test_nlg(self, nlg_class):
        for mode in self.modes:
            is_user = True
            acts = self.usr_acts
            nlg = nlg_class(is_user, mode)
            self._test_generate(nlg, acts)

            is_user = False
            acts = self.sys_acts
            nlg = nlg_class(is_user, mode)
            self._test_generate(nlg, acts)


class BaseTestSCLSTM(BaseTestNLG):
    """Base class for testing sclstm nlg.

    Note:
        BaseTestSCLSTM or its subclass should have attributes(instance attributes or class attributes):
            `usr_acts` and `sys_acts`
    """
    def _test_nlg(self, nlg_class):
        for is_user in (True, False):
            nlg = nlg_class(is_user=is_user)
            if is_user:
                acts = self.usr_acts
            else:
                acts = self.sys_acts
            self._test_generate(nlg, acts)


class BaseTestMultiwoz:
    """Base class for testing multiwoz. This class has attributes `usr_acts` and `sys_acts`."""

    @classmethod
    def setup_class(cls):
        cls.usr_acts = [
            {
                "Hotel-Inform": [
                    [
                        "Area",
                        "east"
                    ],
                    [
                        "Stars",
                        "4"
                    ]
                ]
            },
            {
                "Hotel-Inform": [
                    [
                        "Parking",
                        "yes"
                    ],
                    [
                        "Internet",
                        "yes"
                    ]
                ]
            },
            {},
            {
                "Hotel-Inform": [
                    [
                        "Day",
                        "friday"
                    ]
                ],
                "Hotel-Request": [
                    [
                        "Ref",
                        "?"
                    ]
                ]
            },
            {
                "Train-Inform": [
                    [
                        "Dest",
                        "bishops stortford"
                    ],
                    [
                        "Day",
                        "friday"
                    ],
                    [
                        "Depart",
                        "cambridge"
                    ]
                ]
            },
            {
                "Train-Inform": [
                    [
                        "Arrive",
                        "19:45"
                    ]
                ]
            },
            {
                "Train-Request": [
                    [
                        "Leave",
                        "?"
                    ],
                    [
                        "Time",
                        "?"
                    ],
                    [
                        "Ticket",
                        "?"
                    ]
                ]
            },
            {
                "Hotel-Inform": [
                    [
                        "Stay",
                        "4"
                    ],
                    [
                        "Day",
                        "monday"
                    ],
                    [
                        "People",
                        "3"
                    ]
                ]
            },

        ]

        cls.sys_acts = [
            {
                "Hotel-Request": [
                    [
                        "Price",
                        "?"
                    ]
                ]
            },
            {
                "Hotel-Recommend": [
                    [
                        "Price",
                        "cheap"
                    ],
                    [
                        "Price",
                        "moderately priced"
                    ],
                    [
                        "Name",
                        "Allenbell"
                    ],
                    [
                        "Name",
                        "Warkworth House"
                    ]
                ]
            },
            {
                "Booking-Request": [
                    [
                        "Day",
                        "?"
                    ]
                ]
            },
            {
                "general-reqmore": [
                    [
                        "none",
                        "none"
                    ]
                ],
                "Booking-Book": [
                    [
                        "Ref",
                        "BMUKPTG6"
                    ]
                ]
            },
            {
                "Train-Inform": [
                    [
                        "Choice",
                        "a number"
                    ],
                    [
                        "Leave",
                        "throughout the day"
                    ]
                ],
                "Train-Request": [
                    [
                        "Leave",
                        "?"
                    ]
                ]
            },
            {
                "Train-Inform": [
                    [
                        "Leave",
                        "17:29"
                    ],
                    [
                        "Arrive",
                        "18:07"
                    ]
                ],
                "Train-OfferBook": [
                    [
                        "none",
                        "none"
                    ]
                ]
            },
            {
                "general-reqmore": [
                    [
                        "none",
                        "none"
                    ]
                ],
                "Train-OfferBooked": [
                    [
                        "Time",
                        "38 minutes"
                    ],
                    [
                        "Ref",
                        "UIFV8FAS"
                    ],
                    [
                        "Ticket",
                        "10.1 GBP"
                    ]
                ]
            },
            {
                "Booking-Book": [
                    [
                        "Ref",
                        "YF86GE4J"
                    ]
                ]
            },

        ]


class BaseTestCamrest:
    """Base class for testing camrest. This class has attributes `usr_acts` and `sys_acts`."""

    @classmethod
    def setup_class(cls):
        cls.usr_acts = [
            {
                "inform": [
                    [
                        "food",
                        "brazilian"
                    ],
                    [
                        "area",
                        "north"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "food",
                        "portuguese"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "food",
                        "indian"
                    ]
                ]
            },
            {
                "request": [
                    [
                        "phone",
                        "?"
                    ],
                    [
                        "address",
                        "?"
                    ]
                ]
            },
            {},
            {
                "request": [
                    [
                        "postcode",
                        "?"
                    ],
                    [
                        "phone",
                        "?"
                    ],
                    [
                        "address",
                        "?"
                    ]
                ],
                "inform": [
                    [
                        "food",
                        "chinese"
                    ],
                    [
                        "area",
                        "east"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "food",
                        "venetian"
                    ],
                    [
                        "area",
                        "west"
                    ]
                ],
                "request": [
                    [
                        "phone",
                        "?"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "food",
                        "dontcare"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "pricerange",
                        "moderate"
                    ],
                    [
                        "area",
                        "south"
                    ]
                ]
            },
            {
                "request": [
                    [
                        "phone",
                        "?"
                    ],
                    [
                        "address",
                        "?"
                    ]
                ]
            },
        ]
        cls.sys_acts = [
            {
                "nooffer": [
                    [
                        "none",
                        "none"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "name",
                        "Royal Spice"
                    ],
                    [
                        "pricerange",
                        "cheap"
                    ],
                    [
                        "food",
                        "Indian"
                    ],
                    [
                        "area",
                        "north"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "postcode",
                        "C.B 4, 1 E.H"
                    ],
                    [
                        "address",
                        "Victoria Avenue Chesterton"
                    ],
                    [
                        "phone",
                        "01733 553355"
                    ]
                ]
            },
            {},
            {
                "inform": [
                    [
                        "name",
                        "Yu Garden"
                    ],
                    [
                        "area",
                        "east"
                    ],
                    [
                        "food",
                        "Chinese"
                    ],
                    [
                        "address",
                        "529 Newmarket Road Fen Ditton"
                    ],
                    [
                        "postcode",
                        "C.B 5, 8 P.A"
                    ],
                    [
                        "phone",
                        "01223 248882"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "name",
                        "Pizza Hut Cherry Hinton"
                    ],
                    [
                        "name",
                        "Restaurant Alimentum"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "food",
                        "Modern european"
                    ],
                    [
                        "food",
                        "Italian"
                    ]
                ],
                "request": [
                    [
                        "food",
                        "?"
                    ]
                ]
            },
            {
                "inform": [
                    [
                        "address",
                        "G4 cambridge leisure park clifton way cherry hinton"
                    ],
                    [
                        "name",
                        "pizza hut cherry hinton"
                    ],
                    [
                        "phone",
                        "01223 413000"
                    ]
                ]
            },
        ]
