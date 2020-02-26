import json
from pprint import pprint
import zipfile


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def get_goal_type(data, mode):
    goal_types = []
    for no, sess in data.items():
        goal_type = sess['type']
        for i, turn in enumerate(sess['messages']):
            if mode == 'usr' and turn['role'] == 'sys':
                continue
            elif mode == 'sys' and turn['role'] == 'usr':
                continue
            goal_types.append(goal_type)
    return goal_types


def calculateF1(predict_golden, goal_type=None, intent=None, domain=None, slot=None):
    if domain=='General':
        domain = None
        intent = 'General'
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for quad in predicts:
            if intent and quad[0] != intent:
                continue
            if domain and quad[1] != domain:
                continue
            if quad in labels:
                TP += 1
            else:
                FP += 1
        for quad in labels:
            if intent and quad[0] != intent:
                continue
            if domain and quad[1] != domain:
                continue
            if quad not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, F1


if __name__ == '__main__':
    predict_golden = json.load(open('output/all_context/output.json',encoding='utf-8'))
    print('all', calculateF1(predict_golden))
    goal_types = get_goal_type(read_zipped_json('../../../../data/crosswoz/test.json.zip', 'test.json',),mode='all')
    type_predict_golden = {}
    for goal_type, d in zip(goal_types,predict_golden):
        type_predict_golden.setdefault(goal_type, [])
        type_predict_golden[goal_type].append(d)
    for goal_type in type_predict_golden:
        print(goal_type,len(type_predict_golden[goal_type]))
        print([float('%.2f' % (x*100)) for x in calculateF1(type_predict_golden[goal_type])])
    intents = ['Inform', 'Request', 'General', 'Recommend', 'Select', 'NoOffer']
    domains = ['景点', '酒店', '餐馆', '出租', '地铁', 'General']
    intent_predict_golden = dict.fromkeys(intents)
    domain_predict_golden = dict.fromkeys(domains)
    for intent in intents:
        intent_predict_golden[intent] = [float('%.2f' % (x*100)) for x in calculateF1(predict_golden,intent=intent)]
    for domain in domains:
        domain_predict_golden[domain] = [float('%.2f' % (x*100)) for x in calculateF1(predict_golden,domain=domain)]
    pprint(intent_predict_golden)
    pprint(domain_predict_golden)
