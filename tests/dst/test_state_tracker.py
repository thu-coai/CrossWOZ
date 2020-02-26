import pytest
import abc
from convlab2.dst.dst import DST
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA
from convlab2.util.multiwoz.state import default_state


def test_tracker():
    with pytest.raises(TypeError):
        DST()
    assert hasattr(DST, "update")
    assert hasattr(DST, "init_session")


class BaseTestTracker(abc.ABC):
    """

    Note: instance of BaseTestTracker's subclass should have attribute `tracker`
    """
    def setup_method(self):
        assert hasattr(self, 'dst')
        assert isinstance(self.dst, DST)

    @abc.abstractmethod
    def _check_result(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def _test_update(self, *args, **kwargs):
        raise NotImplementedError


class BaseTestMultiwozTracker(BaseTestTracker):
    @classmethod
    def setup_class(cls):
        # domain_slots will be used for checking request_state
        cls.domain_slots = {}
        for key, value in REF_SYS_DA.items():
            new_key = key.lower()
            new_value = set(value.values())
            cls.domain_slots[new_key] = new_value

        cls.default_belief_state = default_state()['belief_state']

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

    @classmethod
    def _check_format(cls, dict_, target_dict_):
        assert isinstance(dict_, dict) == isinstance(target_dict_, dict)
        if not isinstance(dict_, dict):
            return True
        assert set(dict_) == set(target_dict_)  # check keys
        for key in target_dict_:
            assert isinstance(key, str)
            target_value = target_dict_[key]
            value = dict_[key]
            assert cls._check_format(value, target_value)
        return True

    @classmethod
    def _check_action(cls, action):
        assert isinstance(action, dict)
        for key, value in action.items():
            assert isinstance(key, str)
            assert isinstance(value, list)
            for item in value:
                assert isinstance(item, list)
                slot, v = item
                assert isinstance(slot, str)
                assert isinstance(v, str)

    def _check_result(self, state):
        assert isinstance(state, dict)
        KEYS = "user_action", "system_action", "belief_state", "request_state", "terminated", "history"
        for key in KEYS:
            assert key in state
        user_action, system_action, belief_state, request_state, terminated, history = map(state.get, KEYS)

        # check user_action
        self._check_action(user_action)

        # check_system_action
        self._check_action(system_action)

        # check belief_state
        assert isinstance(belief_state, dict)
        assert isinstance(self.default_belief_state, dict)
        self._check_format(belief_state, self.default_belief_state)

        # check request_state
        assert isinstance(request_state, dict)
        for domain, slot_value_dict in request_state.items():
            assert isinstance(domain, str)
            assert domain in self.__class__.domain_slots
            assert isinstance(slot_value_dict, dict)
            for slot, value in slot_value_dict.items():
                assert isinstance(slot, str)
                assert isinstance(value, str)
                assert slot in self.__class__.domain_slots[domain]

        # check terminated
        assert isinstance(terminated, bool)

        # check history
        assert isinstance(history, list)
        for item in history:
            assert isinstance(item, list)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)

    def _test_update(self, actions):
        for action in actions:
            state = self.tracker.update(action)
            self._check_result(state)
