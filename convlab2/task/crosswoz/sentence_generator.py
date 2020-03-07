# -*- coding: utf-8 -*-
import random

import numpy as np


class SentenceGenerator:
    def generate(self, goals, random_seed=None):
        sens = []
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        # print(goals)
        # for ls in goals:
        for goal in goals:
            # print(goal)
            sen = ''
            # if "周边" in goal["生成方式"]:
            #     sen += goal["生成方式"] + "。" + "通过它的周边推荐，"
            domain = goal["领域"]
            if domain == "酒店":
                for constraint in goal["约束条件"]:
                    if constraint[0] == "名称":
                        if '周边' in constraint[1]:
                            origin_id = int(constraint[1].split('id=')[1][0])
                            sen += ('你要去id=%d附近的酒店(id=%d)住宿。' % (origin_id, goal['id']))
                        else:
                            sen += ('你要去名叫%s的酒店(id=%d)住宿。' % (constraint[1], goal['id']))
                if sen == '':
                    sen += "你要去一个酒店(id=%d)住宿。" % goal['id']

                for constraint in goal["约束条件"]:
                    if constraint[0] == "酒店类型":
                        sen += ('你希望酒店是%s的。' % constraint[1])
                    elif "酒店设施" in constraint[0]:
                        sen += ('你希望酒店提供%s。' % constraint[0].split('-')[1])
                    elif constraint[0] == "价格":
                        sen += ('你希望酒店的最低价格是%s的。' % constraint[1])
                    elif constraint[0] == "评分":
                        sen += ('你希望酒店的评分是%s。' % constraint[1])
                    elif constraint[0] == "预订信息":
                        sen += ""
                # if goal["预订信息"]:
                #     sen += "你希望预订在%s入住，共%s人，住%s天。" % (goal["预订信息"][1][1], goal["预订信息"][0][1], goal["预订信息"][2][1])
            elif domain == "景点":

                for constraint in goal["约束条件"]:
                    if constraint[0] == "名称":
                        if '周边' in constraint[1]:
                            origin_id = int(constraint[1].split('id=')[1][0])
                            sen += ('你要去id=%d附近的景点(id=%d)游玩。' % (origin_id, goal['id']))
                        else:
                            sen += ('你要去名叫%s的景点(id=%d)游玩。' % (constraint[1], goal['id']))
                if sen == '':
                    sen += "你要去一个景点(id=%d)游玩。" % goal['id']

                for constraint in goal["约束条件"]:
                    if constraint[0] == "门票":
                        sen += ('你希望景点的票价是%s的。' % constraint[1])
                    elif constraint[0] == "游玩时间":
                        sen += ('你希望游玩的时长是%s。' % constraint[1])
                    elif constraint[0] == "评分":
                        sen += ('你希望景点的评分是%s。' % constraint[1])
            elif domain == "餐馆":
                for constraint in goal["约束条件"]:
                    if constraint[0] == "名称":
                        if '周边' in constraint[1]:
                            origin_id = int(constraint[1].split('id=')[1][0])
                            sen += ('你要去id=%d附近的餐馆(id=%d)用餐。' % (origin_id, goal['id']))
                        else:
                            sen += ('你要去名叫%s的餐馆(id=%d)用餐。' % (constraint[1], goal['id']))
                if sen == '':
                    sen += "你要去一个餐馆(id=%d)用餐。" % goal['id']

                for constraint in goal["约束条件"]:
                    if constraint[0] == "推荐菜":
                        sen += ('你想吃的菜肴是%s。' % '、'.join(constraint[1]))
                    elif constraint[0] == "人均消费":
                        sen += ('你希望餐馆的人均消费是%s的。' % constraint[1])
                    elif constraint[0] == "评分":
                        sen += ('你希望餐馆的评分是%s。' % constraint[1])
                # if goal["预订信息"]:
                #     sen += "你希望预订在%s%s共%s人一起用餐。" % (goal["预订信息"][1][1], goal["预订信息"][2][1], goal["预订信息"][0][1])
            elif domain == "出租":
                sen += '你想呼叫从%s到%s的出租车。' % (goal["约束条件"][0][1], goal["约束条件"][1][1])
            elif domain == "地铁":
                sen += '你想乘坐从%s到%s的地铁。' % (goal["约束条件"][0][1], goal["约束条件"][1][1])
            sen += '你想知道这个%s的%s。' % (domain, '、'.join(["酒店设施是否包含%s" % item[0].split('-')[1]
                                                       if "酒店设施" in item[0]
                                                       else item[0]
                                                       for item in goal['需求信息']]))
            sens.append(sen)
        return sens
