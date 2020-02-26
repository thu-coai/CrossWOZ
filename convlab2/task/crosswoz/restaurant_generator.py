# -*- coding: utf-8 -*-
import json
import random
from copy import deepcopy
from collections import Counter

import numpy as np


class RestaurantGenerator:
    def __init__(self, database):
        self.database = database.values()
        self.constraints2prob = {
            "名称": 0.1,
            "推荐菜": 0.6,
            "人均消费": 0.5,
            "评分": 0.5
        }
        self.constraints2weight = {
            "名称": dict.fromkeys([x['名称'] for x in self.database], 1),
            "推荐菜": {1: 5, 2: 1},
            "人均消费": {"50元以下": 1,
                     "50-100元": 15,
                     "100-150元": 15,
                     "150-500元": 5,
                     "500-1000元": 2,
                     "1000元以上": 1
                     },
            "评分": {'4分以上': 0.2, '4.5分以上': 0.6, '5分': 0.2}
        }
        self.min_constraints = 1
        self.max_constraints = 3
        self.min_require = 1
        self.max_require = 3
        self.order_prob = 0.1
        self.twodish_prob = 0.15
        self.all_attrs = ['名称', '地址', '电话', '营业时间', '推荐菜', '人均消费', '评分',
                          '周边景点', '周边餐馆', '周边酒店',
                          # '交通', '介绍',
                          ]
        self.cooccur = {}  # check if the list is empty
        for res in self.database:
            for dish in res['推荐菜']:
                self.cooccur[dish] = self.cooccur.get(dish, set()).union(set(res['推荐菜']))
                self.cooccur[dish].remove(dish)
        all_dish = [dish for res in self.database for dish in res['推荐菜']]
        all_dish = Counter(all_dish)
        for k,v in all_dish.items():
            if v==1:
                del self.cooccur[k]
        self.time2weight = {}
        for hour in range(0, 23):
            for minute in [':00', ':30']:
                timePoint = str(hour) + minute
                if hour in [11, 12, 17, 18]:  # 饭点
                    self.time2weight[timePoint] = 20
                elif hour in list(range(0, 7)):  # 深夜/清晨
                    self.time2weight[timePoint] = 1
                else:  # 白天非饭点
                    self.time2weight[timePoint] = 5

    def generate(self, goal_num=0, exist_goal=None, random_seed=None):
        name_flag = False
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        goal = {
            "领域": "餐馆",
            "id": goal_num,
            "约束条件": [],
            "需求信息": [],
            '预订信息': [],
            "生成方式": ""
        }
        # generate method
        if exist_goal:
            goal['生成方式'] = 'id={}的周边{}'.format(exist_goal["id"], "餐馆")
            goal['约束条件'].append(['名称', '出现在id={}的周边{}里'.format(exist_goal["id"], "餐馆")])
            name_flag = True
        else:
            goal['生成方式'] = '单领域生成'
        # generate constraints
        random_req = deepcopy(self.all_attrs)
        random_req.remove('名称')
        # if constraint == name ?
        if not exist_goal and random.random() < self.constraints2prob['名称']:
            v = self.constraints2weight['名称']
            goal['约束条件'] = [['名称', random.choices(list(v.keys()), list(v.values()))[0]]]
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
                    if k == '推荐菜':
                        value = random.choices(list(self.cooccur.keys()))
                        if random.random() < self.twodish_prob and self.cooccur[value[0]]:
                            value.append(random.choice(list(self.cooccur[value[0]])))
                    else:
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
        req_num = random.choices([1, 2], [30, 70])[0]
        for k in random_req:
            if req_num > 0:
                goal['需求信息'].append([k, ""])
                req_num -= 1
                if k == '名称':
                    name_flag = True
            else:
                break



        # if random.random() < self.order_prob:
        #     people_num = random.randint(1, 9)
        #     week_day = random.choice(['周日', '周一', '周二', '周三', '周四', '周五', '周六', ])
        #     book_time = random.choices(list(self.time2weight.keys()), list(self.time2weight.values()))[0]
        #     goal['预订信息'] = [["人数", people_num], ["日期", week_day], ["时间", book_time]]
        #     goal['需求信息'].append(["预订订单号", ""])

        return goal
