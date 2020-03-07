# -*- coding: utf-8 -*-
import json
import random
from collections import Counter
from copy import deepcopy
from itertools import chain

import numpy as np


class HotelGenerator:
    def __init__(self, database):
        self.database = database.values()
        self.constraints2prob = {
            '名称':0.1,
            '酒店类型': 0.7,
            '酒店设施': 0.5,
            '价格': 0.3,
            '评分': 0.5
        }

        lp = [x['价格'] for x in self.database if x['价格'] is not None]
        lp = [int(i) for i in lp]
        raw_faci = [x['酒店设施'] for x in self.database]
        all_faci = list(chain(*raw_faci))
        all_faci = [x for x in all_faci if '\n' not in x]
            
        self.constraints2weight = {
            "名称": dict.fromkeys([x['名称'] for x in self.database], 1),
            "酒店类型": dict(Counter([x['酒店类型'] for x in self.database])),
            "酒店设施": dict(Counter(all_faci)),
            "价格": {
                '100-200元': np.sum(list(map(lambda x: 100 <= x <= 200, lp))),
                '200-300元': np.sum(list(map(lambda x: 200 <= x <= 300, lp))),
                '300-400元': np.sum(list(map(lambda x: 300 <= x <= 400, lp))),
                '400-500元': np.sum(list(map(lambda x: 400 <= x <= 500, lp))),
                '500-600元': np.sum(list(map(lambda x: 500 <= x <= 600, lp))),
                '600-700元': np.sum(list(map(lambda x: 600 <= x <= 700, lp))),
                '700-800元': np.sum(list(map(lambda x: 700 <= x <= 800, lp))),
                '800-900元': np.sum(list(map(lambda x: 800 <= x <= 900, lp))),
                '900-1000元': np.sum(list(map(lambda x: 900 <= x <= 1000, lp))),
                '1000元以上': np.sum(list(map(lambda x: x > 1000, lp)))
            },
            "评分": {'4分以上': 0.2, '4.5分以上': 0.6, '5分': 0.2}
        }
        self.faci_cons_expect = 2
        self.min_faci_cons = 1
        self.max_faci_cons = 2
        self.min_constraints = 1
        self.max_constraints = 3
        self.min_require = 1
        self.max_require = 3
        self.order_prob = 0.1
        self.all_attrs =  ['酒店设施', '名称', '酒店类型', '地址', '电话', '价格', '评分',
                           '周边景点', '周边餐馆'
                           # '预订须知', '附加费用', '酒店简介', '儿童政策', '入离时间',
                           ]

    def generate(self, goal_num=0, exist_goal=None, random_seed=None):
        name_flag = False
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        goal = {
            "领域": "酒店",
            "id": goal_num,
            "约束条件": [],
            "需求信息": [],
            "预订信息": [],
            "生成方式":""
        }
        random_req = deepcopy(self.all_attrs)
        random_req.remove('名称')
        faci_cons = list(self.constraints2weight['酒店设施'].keys())
        rest_faci_cons = deepcopy(faci_cons)
        web_serv = ['公共区域和部分房间提供wifi', '酒店各处提供wifi', '部分房间提供wifi', '公共区域提供wifi', '所有房间提供wifi']
        web_flag = 1
        # generate method
        if exist_goal:
            goal['生成方式'] = 'id={}的周边{}'.format(exist_goal["id"], "酒店")
            goal['约束条件'].append(['名称', '出现在id={}的周边{}里'.format(exist_goal["id"], "酒店")])
            name_flag = True
        else:
            goal['生成方式'] = '单领域生成'
        # generate constraints
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
                    if k != '酒店设施':
                        value = random.choices(list(v.keys()),list(v.values()))[0]
                        goal['约束条件'].append([k, value])
                        random_req.remove(k)
                    else:
                        value_list = []
                        random.shuffle(faci_cons)
                        faci_cons_num = random.randint(self.min_faci_cons, self.max_faci_cons)
                        while faci_cons_num > 0:
                            value = random.choices(list(v.keys()), list(v.values()))[0]
                            if value not in value_list:
                                if value not in web_serv:
                                    value_list.append(value)
                                    rest_faci_cons.remove(value)
                                else:
                                    if web_flag:
                                        value_list.append(value)
                                        web_flag = 0
                                faci_cons_num -= 1
                        for value in value_list:
                            goal['约束条件'].append([k+'-'+value,'是'])
                        random_req.remove(k)
                    cons_num -= 1
                else:
                    break

        # generate required information
        if not name_flag:
            goal['需求信息'].append(['名称', ""])

        random.shuffle(random_req)
        req_num = random.choices([1, 2], [30, 70])[0]
        for k in random_req:
            if req_num > 0:
                assert k != '名称'
                if k != '酒店设施':
                    goal['需求信息'].append([k,""])
                else:
                    random.shuffle(rest_faci_cons)
                    faci_req_list = []
                    faci_cons_num = random.randint(self.min_faci_cons, self.max_faci_cons)
                    for i in rest_faci_cons:
                        if faci_cons_num > 0:
                            if i not in web_serv:
                                faci_req_list.append(i)
                            else:
                                if web_flag:
                                    faci_req_list.append(i)
                                    web_flag = 0
                            faci_cons_num -= 1
                    for value in faci_req_list:
                        goal['需求信息'].append([k+'-'+value,""])

                req_num -= 1
            else:
                break


        # generate book information
        # all_date = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
        # max_duration = 9
        # max_person = 9
        # if random.random() < self.order_prob:
        #     person = random.randint(1, max_person)
        #     date = all_date[random.randint(0, 6)]
        #     duration = random.randint(1, max_duration)
        #     bookInfo = [["人数", person], ["开始时间", date], ["天数", duration]]
        #     goal['预订信息'] = bookInfo
        #     goal['需求信息'].append(["预订订单号", ""])

        return goal
