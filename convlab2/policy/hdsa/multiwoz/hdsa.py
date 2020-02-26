import os
import sys
#sys.path.append("/home/mawenchang/Convlab-2/")
from convlab2.policy.hdsa.multiwoz.predictor import HDSA_predictor
from convlab2.policy.hdsa.multiwoz.generator import HDSA_generator
from convlab2.policy import Policy

DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "hdsa.zip")


class HDSA(Policy):

    def __init__(self, archive_file=DEFAULT_ARCHIVE_FILE, model_file=None, use_cuda=False):
        self.predictor = HDSA_predictor(archive_file, model_file, use_cuda)
        self.generator = HDSA_generator(archive_file, model_file, use_cuda)

    def init_session(self):
        self.generator.init_session()

    def predict(self, state):

        act, kb = self.predictor.predict(state)
        response = self.generator.generate(state, act, kb)

        return response


if __name__ == '__main__':

    state = {'belief_state': {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "west",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    },
             'history': [['null', 'I want to find a restaurant west of town .']],
             'request_state': {},
             'user_action': {'Restaurant-Inform': [['area', 'west']]}}

    cur_model = HDSA(model_file="https://convlab.blob.core.windows.net/models/hdsa.zip")
    response = cur_model.predict(state)
    import pprint as pp
    pp.pprint(response)

