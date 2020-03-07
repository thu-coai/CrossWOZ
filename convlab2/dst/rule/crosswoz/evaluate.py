import json
import zipfile
from collections import Counter
from pprint import pprint
from convlab2.dst.rule.crosswoz.dst import RuleDST
from convlab2.util.crosswoz.state import default_state
from copy import deepcopy


def calculateJointState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        res.append(predicts==labels)
    return sum(res) / len(res) if len(res) else 0.


def calculateSlotState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for x, y in zip(predicts, labels):
            for w, z in zip(predicts[x].values(),labels[y].values()):
                res.append(w==z)
    return sum(res) / len(res) if len(res) else 0.


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def test_sys_state(data, goal_type):
    ruleDST = RuleDST()
    state_predict_golden = []
    for task_id, item in data.items():
        if goal_type and item['type']!=goal_type:
            continue
        ruleDST.init_session()
        for i, turn in enumerate(item['messages']):
            if turn['role'] == 'sys':
                usr_da = item['messages'][i - 1]['dialog_act']
                if i > 2:
                    for domain, svs in item['messages'][i - 2]['sys_state'].items():
                        for slot, value in svs.items():
                            if slot != 'selectedResults':
                                ruleDST.state['belief_state'][domain][slot] = value
                ruleDST.update(usr_da)
                new_state = deepcopy(ruleDST.state['belief_state'])
                golden_state = deepcopy(turn['sys_state_init'])
                for x in golden_state:
                    golden_state[x].pop('selectedResults')
                state_predict_golden.append({
                        'predict': new_state,
                        'golden': golden_state
                })
    print('joint state', calculateJointState(state_predict_golden))
    print('slot state', calculateSlotState(state_predict_golden))


if __name__ == '__main__':
    test_data_path = '../../../../data/crosswoz/test.json.zip'
    test_data = read_zipped_json(test_data_path, 'test.json')
    for goal_type in ['单领域', '独立多领域', '独立多领域+交通', '不独立多领域', '不独立多领域+交通', None]:
        print(goal_type)
        test_sys_state(test_data, goal_type=goal_type)
