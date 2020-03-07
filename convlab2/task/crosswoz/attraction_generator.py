# -*- coding: utf-8 -*-
import json
import random
from collections import Counter
from copy import deepcopy

import numpy as np


class AttractionGenerator:
    def __init__(self, database):
        self.database = database.values()
        self.constraints2prob = {
            "名称": 0.1,
            "门票": 0.5,
            "游玩时间": 0.5,
            "评分": 0.5
        }
        self.constraints2weight = {
            "名称": dict.fromkeys([x['名称'] for x in self.database],1),
            "门票": {'免费': 10, '不免费': 1, '20元以下': 2, '20-50元': 4, '50-100元': 3, '100-150元': 3, '150-200元': 3, '200元以上': 2},
            "游玩时间": dict(Counter([x['游玩时间'] for x in self.database])),
            "评分": {'4分以上': 0.2, '4.5分以上': 0.6, '5分': 0.2}
        }
        self.min_constraints = 1
        self.max_constraints = 3
        self.min_require = 1
        self.max_require = 3
        self.all_attrs = ['名称', '地址', '电话', '门票', '游玩时间', '评分',
                          '周边景点', '周边餐馆', '周边酒店',
                          # '官网', '介绍', '开放时间'
                          ]

    def generate(self, goal_num=0, exist_goal=None, random_seed=None):
        name_flag = False
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        goal = {
            "领域": "景点",
            "id": goal_num,
            "约束条件": [],
            "需求信息": [],
            "生成方式": ""
        }
        # generate method
        if exist_goal:
            goal['生成方式'] = 'id={}的周边{}'.format(exist_goal["id"], "景点")
            goal['约束条件'].append(['名称', '出现在id={}的周边{}里'.format(exist_goal["id"], "景点")])
            name_flag = True
        else:
            goal['生成方式'] = '单领域生成'

        # generate constraints
        random_req = deepcopy(self.all_attrs)
        random_req.remove('名称')
        # if constraint == name ?
        if not exist_goal and random.random() < self.constraints2prob['名称']:
            v = self.constraints2weight['名称']
            goal['约束条件'] = [['名称', random.choices(list(v.keys()),list(v.values()))[0]]]
            name_flag = True
        else:
            rest_constraints = list(self.constraints2prob.keys())
            rest_constraints.remove('名称')
            random.shuffle(rest_constraints)
            # cons_num = random.randint(self.min_constraints, self.max_constraints)
            cons_num = random.choices([1, 2, 3], [20, 60, 20])[0]
            for k in rest_constraints:
                if cons_num > 0:
                    v = self.constraints2weight[k]
                    value = random.choices(list(v.keys()), list(v.values()))[0]
                    goal['约束条件'].append([k, value])
                    random_req.remove(k)
                    cons_num -= 1
                else:
                    break

        # generate required information
        if not name_flag:
            goal['需求信息'].append(['名称', ""])

        random.shuffle(random_req)
        # req_num = random.randint(self.min_require, self.max_require)
        req_num = random.choices([1, 2], [30, 70])[0]
        for k in random_req:
            if req_num > 0:
                goal['需求信息'].append([k,""])
                req_num -= 1
                if k == '名称':
                    name_flag = True
            else:
                break

        return goal
