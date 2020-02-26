"""
Evaluate SVMNLU models on Multiwoz test dataset

Metric:
    dataset level Precision/Recall/F1

Usage:
    PYTHONPATH=../../../.. python evaluate.py [usr|sys|all]
"""
import json
import random
import sys
import zipfile

import numpy
import torch

from convlab2.nlu.svm.multiwoz import SVMNLU

seed = 2019
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)


def da2triples(dialog_act):
    triples = []
    for intent, svs in dialog_act.items():
        for slot, value in svs:
            triples.append([intent, slot, value])
    return triples


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:")
        print("\t python evaluate.py mode")
        print("\t mode=usr|sys|all")
        sys.exit()
    mode = sys.argv[1]
    if mode== 'usr':
        model = SVMNLU(mode='usr')
    elif mode== 'sys':
        model = SVMNLU(mode='sys')
    elif mode== 'all':
        model = SVMNLU(mode='all')
    else:
        raise Exception("Invalid mode")

    archive = zipfile.ZipFile('../../../../data/multiwoz/test.json.zip', 'r')
    test_data = json.load(archive.open('test.json'))
    TP, FP, FN = 0, 0, 0
    sen_num = 0
    sess_num = 0
    for no, session in test_data.items():
        sess_num += 1
        if sess_num%10==0:
            print('Session [%d|%d]' % (sess_num, len(test_data)))
            precision = 1.0 * TP / (TP + FP)
            recall = 1.0 * TP / (TP + FN)
            F1 = 2.0 * precision * recall / (precision + recall)
            print('Model on {} session {} sentences:'.format(sess_num, sen_num))
            print('\t Precision: %.2f' % (100 * precision))
            print('\t Recall: %.2f' % (100 * recall))
            print('\t F1: %.2f' % (100 * F1))
        for i, turn in enumerate(session['log']):
            if i % 2 == 0 and mode == 'sys':
                continue
            elif i % 2 == 1 and mode == 'usr':
                continue
            sen_num += 1
            labels = da2triples(turn['dialog_act'])
            predicts = model.predict(turn['text'])
            for triple in predicts:
                if triple in labels:
                    TP += 1
                else:
                    FP += 1
            for triple in labels:
                if triple not in predicts:
                    FN += 1
    print(TP,FP,FN)
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / (TP + FN)
    F1 = 2.0 * precision * recall / (precision + recall)
    print('Model on {} session {} sentences data_key={}'.format(len(test_data), sen_num, mode))
    print('\t Precision: %.2f' % (100 * precision))
    print('\t Recall: %.2f' % (100 * recall))
    print('\t F1: %.2f' % (100 * F1))
