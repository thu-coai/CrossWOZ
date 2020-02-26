"""
reorder generated goals
"""
from copy import deepcopy
import pprint
import json
import re
import random


def goals_reorder(goal_list):
    # pprint.pprint(goal_list)
    id_old2new = {}
    single_ids = []
    cross_ids = {}
    move_ids = []
    for goal in goal_list:
        if goal['生成方式'] == '单领域生成':
            single_ids.append(goal['id'])
            # print('单领域生成:',goal['id'])
        elif '周边' in goal['生成方式']:
            searchObj = re.search(r'id=(\d)', goal['生成方式'])
            src = int(searchObj.group(1))
            cross_ids[goal['id']] = src
            # print('跨领域生成:', goal['id'])
        elif goal['领域'] == '地铁' or goal['领域'] == '出租':
            start_id, end_id = 0, 0
            for slot, value in goal['约束条件']:
                if slot == '出发地':
                    start_id = int(value[-1])
                elif slot == '目的地':
                    end_id = int(value[-1])
                else:
                    assert 0
            move_ids.append((goal['id'],start_id,end_id))
            # print('地铁/出租', goal['id'])
        else:
            assert 0

    # pprint.pprint(goal_list)

    order = []
    for x in single_ids:
        order.append(x)
        for k, s, e in move_ids[:]:
            if s in order and e in order:
                order.append(k)
                move_ids.remove((k, s, e))
        for tar, src in list(cross_ids.items())[:]:
            if src==x:
                order.append(tar)
                cross_ids.pop(tar)
                for k, s, e in move_ids[:]:
                    if s in order and e in order:
                        order.append(k)
                        move_ids.remove((k, s, e))
    # print(order)
    assert len(order) == len(goal_list)
    id_old2new = dict([(j,i+1) for i,j in enumerate(order)])
    # print(id_old2new)
    for goal in goal_list:
        goal['id'] = id_old2new[goal['id']]
        if '周边' in goal['生成方式']:
            searchObj = re.search(r'id=(\d)', goal['生成方式'])
            src = int(searchObj.group(1))
            goal['生成方式'] = re.sub('\d', str(id_old2new[src]),goal['生成方式'])
            for i in range(len(goal['约束条件'])):
                if goal['约束条件'][i][0] == '名称':
                    assert 'id' in goal['约束条件'][i][1]
                    goal['约束条件'][i][1] = re.sub('\d', str(id_old2new[src]),goal['约束条件'][i][1])

        elif goal['领域'] == '地铁' or goal['领域'] == '出租':
            start_id = id_old2new[int(goal['约束条件'][0][1][-1])]
            end_id = id_old2new[int(goal['约束条件'][1][1][-1])]
            start_id, end_id = min(start_id, end_id), max(start_id, end_id)
            goal['约束条件'][0][1] = goal['约束条件'][0][1][:-1] + str(start_id)
            goal['约束条件'][1][1] = goal['约束条件'][0][1][:-1] + str(end_id)
    goal_list = sorted(goal_list,key=lambda x:x['id'])
    return goal_list


if __name__ == '__main__':
    goals = json.load(open('result/goal_4.json', encoding='utf-8'))
    for goal in goals:
        if goal['timestamp'] == "2019-05-20 11:23:35.702161":
            # print(goal)
            goals_reorder(goal['goals'])