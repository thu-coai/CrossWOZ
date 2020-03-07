import json
import zipfile
from collections import Counter
from pprint import pprint
from convlab2.policy.rule.crosswoz.rule_simulator import Simulator
from copy import deepcopy


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for quad in predicts:
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if quad not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, F1


def calculateJointState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        res.append(predicts==labels)
    return sum(res)/len(res)


def calculateSlotState(predict_golden):
    res = []
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for x, y in zip(predicts, labels):
            res.append(x==y)
    return sum(res) / len(res)


def begin_active_tuple_num(data):
    active_tuples = []
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages']):
            active_t = []
            for j, st in enumerate(turn['user_state']):
                if st[-1]:
                    active_t.append(j)
            active_tuples.append(tuple(active_t))
            break
    c = Counter(active_tuples)
    s = sum(c.values())
    pprint({x[0]:x[1]/s for x in c.items()})


def begin_da_type(data):
    da_type = []
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages']):
            for intent, domain, slot, value in turn['dialog_act']:
                if intent == 'General':
                    intent = '+'.join([intent,domain,slot,value])
                da_type.append(intent)
            break
    c = Counter(da_type)
    s = sum(c.values())
    print('da_type')
    pprint({x[0]: x[1] / s for x in c.items()})


def end_usr_da_type(data):
    da_type = []
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages'][::-1]):
            if turn['role']=='sys':
                continue
            da_key = ''
            for intent, domain, slot, value in turn['dialog_act']:
                if intent == 'General':
                    intent = '+'.join([intent,domain,slot,value])
                da_key+=intent
                # da_type.append(intent)
            da_type.append(da_key)
            break
    c = Counter(da_type)
    s = sum(c.values())
    print('da_type')
    pprint({x[0]: x[1] / s for x in c.items()})


def eval_begin_da_predict(data):
    predict_golden = []
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages']):
            active_t = []
            for j, st in enumerate(turn['user_state']):
                if st[-1]:
                    active_t.append(st)
            da = [['General', 'greet', 'none', 'none']]
            for intent, domain, slot, value, expressed in active_t:
                if not value:
                    da.append(['Request', domain, slot, ''])
                elif isinstance(value, str):
                    da.append(['Inform', domain, slot, value])
                elif isinstance(value, list):
                    for v in value:
                        da.append(['Inform', domain, slot, v])
                else:
                    assert 0
            predict_golden.append({'predict':sorted(da), 'golden': turn['dialog_act']})
            break
    print('First turn da prediction given state precision/recall/f1',calculateF1(predict_golden))


def eval_simulator_performance(data, goal_type=None):
    begin_da_predict_golden = []
    state_da_predict_golden = []
    state_predict_golden = []
    simulator = Simulator()
    for task_id, item in data.items():
        if goal_type and item['type']!=goal_type:
            continue
        for i, turn in enumerate(item['messages']):
            if turn['role']=='usr':
                if i==0:
                    simulator.init_session(goal=item['goal'])
                    begin_da_predict_golden.append({
                        'predict': simulator.begin_da(),
                        'golden': turn['dialog_act']
                    })
                else:
                    last_turn = item['messages'][i - 2]
                    usr_da = item['messages'][i - 2]['dialog_act']
                    sys_da = item['messages'][i - 1]['dialog_act']
                    simulator.init_session(goal=item['goal'], state=deepcopy(last_turn['user_state']))
                    simulator.state_update(prev_user_da=usr_da, prev_sys_da=sys_da)
                    cur_da = simulator.state_predict()
                    new_state = deepcopy(simulator.state)
                    state_da_predict_golden.append({
                        'predict': cur_da,
                        'golden': turn['dialog_act']
                    })
                    state_predict_golden.append({
                        'predict': new_state,
                        'golden': turn['user_state']
                    })

    print('begin da', calculateF1(begin_da_predict_golden))
    print('state da', calculateF1(state_da_predict_golden))
    print('all da', calculateF1(begin_da_predict_golden+state_da_predict_golden))
    print('joint state', calculateJointState(state_predict_golden))
    print('slot state', calculateSlotState(state_predict_golden))


def eval_state_predict(data):
    def state_update(prev_state, cur_state):
        update = []
        for prev_ele, cur_ele in zip(prev_state, cur_state):
            if cur_ele != prev_ele:
                update.append(cur_ele)
        id = 1
        for ele in cur_state[::-1]:
            if ele[-1]:
                id = ele[0]
                break
        return update, id

    simulator = Simulator()
    for task_id, item in data.items():
        for i, turn in enumerate(item['messages']):
            if turn['role']=='usr' and i > 0:
                last_turn = item['messages'][i-2]
                usr_da = item['messages'][i-2]['dialog_act']
                sys_da = item['messages'][i-1]['dialog_act']
                simulator.init_session(goal=item['goal'],state=deepcopy(last_turn['user_state']))
                simulator.state_update(prev_user_da=usr_da, prev_sys_da=sys_da)
                cur_da = simulator.state_predict()
                new_state = simulator.state
                # print('old state:')
                # pprint(last_turn['user_state'])
                # if 'NoOffer' in [x[0] for x in item['messages'][i-1]['dialog_act']]:
                print(item['messages'][i-2]['content'])
                print(item['messages'][i-1]['content'])
                print(turn['content'])
                print('usr da')
                pprint(usr_da)
                print('sys da')
                pprint(sys_da)
                print('predict state update:')
                pprint(state_update(last_turn['user_state'], new_state))
                print('golden state:')
                pprint(state_update(last_turn['user_state'], turn['user_state']))
                print('predict usr da')
                pprint(cur_da)
                print('golden usr da')
                pprint(turn['dialog_act'])
                print('-'*100)


if __name__ == '__main__':
    # train_data_path = '../../data/raw_data/train.json.zip'
    # train_data = read_zipped_json(train_data_path, 'train.json')
    # begin_active_tuple_num(train_data)
    # begin_da_type(train_data)
    # end_usr_da_type(train_data)
    # eval_begin_da_predict(train_data)
    # print(len(train_data))
    # eval_state_predict(train_data)
    test_data_path = '../../../../data/crosswoz/test.json.zip'
    test_data = read_zipped_json(test_data_path, 'test.json')
    for goal_type in ['单领域', '独立多领域', '独立多领域+交通', '不独立多领域', '不独立多领域+交通', None]:
        print(goal_type)
        eval_simulator_performance(test_data, goal_type=goal_type)
