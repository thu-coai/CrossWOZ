"""
usage:
from convlab2.task.crosswoz.goal_generator import GoalGenerator
GoalGenerator.generate()
"""
import json
import random
from copy import deepcopy
import numpy as np
from collections import Counter
import datetime
import os
from pprint import pprint

from convlab2.task.crosswoz.attraction_generator import AttractionGenerator
from convlab2.task.crosswoz.hotel_generator import HotelGenerator
from convlab2.task.crosswoz.metro_generator import MetroGenerator
from convlab2.task.crosswoz.restaurant_generator import RestaurantGenerator
from convlab2.task.crosswoz.sentence_generator import SentenceGenerator
from convlab2.task.crosswoz.taxi_generator import TaxiGenerator
from convlab2.task.crosswoz.reorder import goals_reorder

goal_num = 0
goal_max = 5


class GoalGenerator:
    @staticmethod
    def generate():
        goal_list = generate_method(
            database_dir=os.path.abspath(os.path.join(os.path.abspath(__file__),'../../../../data/crosswoz/database/')),
            single_domain=False, cross_domain=True,
            multi_target=False, transportation=True)
        goal_list = goals_reorder(goal_list)

        semantic_tuples = []
        for sub_goal in goal_list:
            sub_goal_id = sub_goal['id']
            domain = sub_goal['领域']
            for slot, value in sub_goal['约束条件']:
                semantic_tuples.append([sub_goal_id, domain, slot, value, False])
            for slot, value in sub_goal['需求信息']:
                if '周边' in slot or slot=='推荐菜':
                    value = []
                semantic_tuples.append([sub_goal_id, domain, slot, value, False])
        return semantic_tuples


def call_count():
    global goal_num
    goal_num += 1
    return goal_num


class SingleDomainGenerator():
    def __init__(self, database, domain_index=None):
        self.database = database
        self.attraction_generator = AttractionGenerator(database['attraction'])
        self.restaurant_generator = RestaurantGenerator(database['restaurant'])
        self.hotel_generator = HotelGenerator(database['hotel'])
        self.generators = [self.attraction_generator, self.restaurant_generator, self.hotel_generator]
        if domain_index:
            self.generators = [self.generators[domain_index - 1]]

    def generate(self, multi_target=False):
        goal = []
        # 单领域单目标，保证一定会有一个目标生成
        if not multi_target and len(self.generators) == 1:
            goal.append(self.generators[0].generate(call_count()))
        else:
            random.shuffle(self.generators)
            for generator in self.generators:
                # 多领域单独生成，每个领域中目标以一定概率独立生成
                if random.random() < 0.8:
                    goal.append(generator.generate(call_count()))
                    # 多领域多目标生成，控制总数不超过5
                    if len(goal) == goal_max:
                        break
                    if multi_target and random.random() < 0.2:
                        goal.append(generator.generate(call_count()))

            if len(goal) == 0:
                goal.append(self.generators[0].generate(call_count()))
        assert 0 < len(goal) <= goal_max
        return goal


class CrossDomainGenerator():
    def __init__(self, database):
        self.database = database
        self.attraction_generator = AttractionGenerator(database['attraction'])
        self.restaurant_generator = RestaurantGenerator(database['restaurant'])
        self.hotel_generator = HotelGenerator(database['hotel'])
        self.generators = [self.hotel_generator, self.attraction_generator, self.restaurant_generator]
        '''
        transfer probabolity matrix 
        [hotel attraction restaurant] to [do-not-trans hotel attraction restaurant]
        '''
        self.trans_matrix = [[0, 0.45, 0.45], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3]]

    def generate(self, exist_goal):
        goal = []
        if exist_goal["领域"] == "酒店":
            index = 0
        elif exist_goal["领域"] == "景点":
            index = 1
        else:
            index = 2

        trans_exist = [-1, -1, -1]
        exist_goal_required_info = exist_goal["需求信息"]
        for item in exist_goal_required_info:
            if item[0] == "周边酒店":
                trans_exist[0] = 1
            if item[0] == "周边景点":
                trans_exist[1] = 1
            if item[0] == "周边餐馆":
                trans_exist[2] = 1

        if random.random() < self.trans_matrix[index][0] * trans_exist[0]:
            goal.append(self.generators[0].generate(call_count(), exist_goal))
        if random.random() < self.trans_matrix[index][1] * trans_exist[1]:
            goal.append(self.generators[1].generate(call_count(), exist_goal))
        if random.random() < self.trans_matrix[index][2] * trans_exist[2]:
            goal.append(self.generators[2].generate(call_count(), exist_goal))
        return goal


def load_json(database_dir, filename):
    list_data = json.load(open(os.path.join(database_dir, filename), encoding='utf-8'))
    return {x[0]: x[1] for x in list_data}


def generate_method(database_dir, single_domain=False, cross_domain=False, multi_target=False, transportation=False):
    """
    single_domain: 单领域生成还是多领域单独生成
    cross_domain: 是否需要跨领域跳转
    multi_target: （单个领域内）是否多目标
    transportation: 是否进行出租、地铁生成
    """
    database = {
        'attraction': load_json(database_dir, os.path.join(database_dir, 'attraction_db.json')),
        'hotel': load_json(database_dir, os.path.join(database_dir, 'hotel_db.json')),
        'restaurant': load_json(database_dir, os.path.join(database_dir, 'restaurant_db.json')),
    }
    global goal_num
    goal_num = 0

    # 单领域单目标
    if single_domain and not multi_target:
        # print('method-单领域单目标生成')
        domain_index = random.randint(1, 3)
        single_domain_generator = SingleDomainGenerator(database, domain_index=domain_index)
        single_domain_goal = single_domain_generator.generate(multi_target=False)
        return single_domain_goal

    # 多领域单独生成
    elif not single_domain and not cross_domain and not multi_target:
        # print('method-多领域单独生成')
        single_domain_generator = SingleDomainGenerator(database)
        single_domain_goals = single_domain_generator.generate(multi_target=False)
        # 确保总数至少有一个目标生成
        assert len(single_domain_goals) > 0

        goal_list = single_domain_goals

        # 生成出租、地铁
        if transportation and 1 < len(goal_list) < goal_max:
            if random.random() < 0.3:
                # goal1, goal2 = random.choices(single_domain_goals, k=2)
                metro_generator = MetroGenerator()
                taxi_generator = TaxiGenerator()
                if random.random() < 0.1 and len(goal_list) + 2 <= goal_max:
                    goal_list.append(metro_generator.generate(goal_list, call_count()))
                    goal_list.append(taxi_generator.generate(goal_list, call_count()))
                else:
                    if random.random() < 0.5:
                        goal_list.append(metro_generator.generate(goal_list, call_count()))
                    else:
                        goal_list.append(taxi_generator.generate(goal_list, call_count()))
        return goal_list

    # 多领域可跨领域生成
    elif not single_domain and cross_domain and not multi_target:
        # print('method-多领域可跨领域生成')
        # 首先进行单领域单独生成
        single_domain_generator = SingleDomainGenerator(database)
        single_domain_goals = single_domain_generator.generate(multi_target=False)

        # # 进行跨领域生成（跳转）
        cross_domain_generator = CrossDomainGenerator(database)
        cross_domain_goals = []
        copy_single_domain_goals = deepcopy(single_domain_goals)
        random.shuffle(copy_single_domain_goals)
        for goal in copy_single_domain_goals:
            if len(single_domain_goals) + len(cross_domain_goals) == goal_max:
                break
            for cross_goal in cross_domain_generator.generate(goal):
                if len(single_domain_goals) + len(cross_domain_goals) == goal_max:
                    break
                cross_domain_goals.append(cross_goal)

        goal_list = single_domain_goals + cross_domain_goals
        # 生成出租、地铁
        if transportation and 1 < len(goal_list) < goal_max:
            if random.random() < 0.3:
                # goal1, goal2 = random.choices(single_domain_goals, k=2)
                metro_generator = MetroGenerator()
                taxi_generator = TaxiGenerator()
                if random.random() < 0.1 and len(goal_list) + 2 <= goal_max:
                    metro = metro_generator.generate(goal_list, call_count())
                    taxi = taxi_generator.generate(goal_list, call_count())
                    goal_list.append(metro)
                    goal_list.append(taxi)
                else:
                    if random.random() < 0.5:
                        goal_list.append(metro_generator.generate(goal_list, call_count()))
                    else:
                        goal_list.append(taxi_generator.generate(goal_list, call_count()))
        return goal_list

    # 多领域多目标单独生成
    elif not single_domain and not cross_domain and multi_target:
        # print('method-多领域多目标单独生成')
        single_domain_generator = SingleDomainGenerator(database)
        single_domain_goals = single_domain_generator.generate(multi_target=True)

        goal_list = single_domain_goals

        # 生成出租、地铁
        if transportation and 1 < len(goal_list) < goal_max:
            if random.random() < 0.3:
                # goal1, goal2 = random.choices(single_domain_goals, k=2)
                metro_generator = MetroGenerator()
                taxi_generator = TaxiGenerator()
                if random.random() < 0.1 and len(goal_list) + 2 <= goal_max:
                    goal_list.append(metro_generator.generate(goal_list, call_count()))
                    goal_list.append(taxi_generator.generate(goal_list, call_count()))
                else:
                    if random.random() < 0.5:
                        goal_list.append(metro_generator.generate(goal_list, call_count()))
                    else:
                        goal_list.append(taxi_generator.generate(goal_list, call_count()))
        return goal_list

    # 多领域多目标可跨领域生成
    elif not single_domain and cross_domain and multi_target:
        # print('method-多领域多目标可跨领域生成')
        # 首先进行单领域多目标独立生成
        single_domain_generator = SingleDomainGenerator(database)
        single_domain_goals = single_domain_generator.generate(multi_target=True)

        # 进行跨领域生成（跳转）
        cross_domain_generator = CrossDomainGenerator(database)
        cross_domain_goals = []
        copy_single_domain_goals = deepcopy(single_domain_goals)
        random.shuffle(copy_single_domain_goals)
        for goal in copy_single_domain_goals:
            if len(single_domain_goals) + len(cross_domain_goals) == goal_max:
                break
            for cross_goal in cross_domain_generator.generate(goal):
                if len(single_domain_goals) + len(cross_domain_goals) == goal_max:
                    break
                cross_domain_goals.append(cross_goal)

        goal_list = single_domain_goals + cross_domain_goals
        # 生成出租、地铁
        if transportation and 1 < len(goal_list) < goal_max:
            if random.random() < 0.3:
                # goal1, goal2 = random.choices(single_domain_goals, k=2)
                metro_generator = MetroGenerator()
                taxi_generator = TaxiGenerator()
                if random.random() < 0.1 and len(goal_list) + 2 <= goal_max:
                    metro = metro_generator.generate(goal_list, call_count())
                    taxi = taxi_generator.generate(goal_list, call_count())
                    goal_list.append(metro)
                    goal_list.append(taxi)
                else:
                    if random.random() < 0.5:
                        goal_list.append(metro_generator.generate(goal_list, call_count()))
                    else:
                        goal_list.append(taxi_generator.generate(goal_list, call_count()))
        return goal_list

    # 异常处理
    else:
        raise LookupError('current method is not supported')
        return []


def generate_sentence(goal_list=[]):
    sentence_generator = SentenceGenerator()
    return sentence_generator.generate(goal_list)

# 给定all_goal_list，将目标按照单领域单目标，多领域单独和多领域可跨领域目标进行划分
def method_split(all_goal_list):
    single_domain, multi_domain, cross_domain = [], [], []
    for goal_list in all_goal_list:
        cross_flag = False
        if len(goal_list['goals']) == 1:
            single_domain.append(goal_list)
        else:
            for goal in goal_list['goals']:
                if goal['领域'] in ['餐馆', '酒店', '景点'] and goal['生成方式'] != '单领域生成':
                    cross_domain.append(goal_list)
                    cross_flag = True
                    break
            if not cross_flag:
                multi_domain.append(goal_list)
    return [single_domain, multi_domain, cross_domain]

# 给定all_goal_list，统计各领域相关信息
def domain_count(all_goal_list):
    result = {'餐馆': {'每个目标平均小目标数': 0, '约束条件': [], '需求信息': [], '目标含有本领域小目标的比例': 0, '含有本领域多个小目标的目标比例': 0},
              '酒店': {'每个目标平均小目标数': 0, '约束条件': [], '需求信息': [], '目标含有本领域小目标的比例': 0, '含有本领域多个小目标的目标比例': 0},
              '景点': {'每个目标平均小目标数': 0, '约束条件': [], '需求信息': [], '目标含有本领域小目标的比例': 0, '含有本领域多个小目标的目标比例': 0},
              '地铁': {'每个目标平均小目标数': 0, '约束条件': [], '需求信息': [], '目标含有本领域小目标的比例': 0, '含有本领域多个小目标的目标比例': 0},
              '出租': {'每个目标平均小目标数': 0, '约束条件': [], '需求信息': [], '目标含有本领域小目标的比例': 0, '含有本领域多个小目标的目标比例': 0}}
    for goal_list in all_goal_list:
        domain_count = {'餐馆': 0, '酒店': 0, '景点': 0, '地铁': 0, '出租': 0}
        for goal in goal_list['goals']:
            domain = goal['领域']
            result[domain]['每个目标平均小目标数'] += 1
            domain_count[domain] += 1
            result[domain]['约束条件'].append(len(goal['约束条件']))
            result[domain]['需求信息'].append(len(goal['需求信息']))
        for domain in domain_count:
            if domain_count[domain] > 0:
                result[domain]['目标含有本领域小目标的比例'] += 1
                if domain_count[domain] > 1:
                    result[domain]['含有本领域多个小目标的目标比例'] += 1
    # total_goal_num = sum([result[domain]['每个目标平均小目标数'] for domain in result])
    for domain in result:
        if result[domain]['每个目标平均小目标数']:
            result[domain]['约束条件'] = sum(result[domain]['约束条件']) / result[domain]['每个目标平均小目标数']
            result[domain]['需求信息'] = sum(result[domain]['需求信息']) / result[domain]['每个目标平均小目标数']
        else:
            result[domain]['约束条件'] = 0
            result[domain]['需求信息'] = 0
        result[domain]['每个目标平均小目标数'] = result[domain]['每个目标平均小目标数'] / len(all_goal_list)
        result[domain]['目标含有本领域小目标的比例'] = result[domain]['目标含有本领域小目标的比例'] / len(all_goal_list)
        result[domain]['含有本领域多个小目标的目标比例'] = result[domain]['含有本领域多个小目标的目标比例'] / len(all_goal_list)
    return result

# 给定all_goal_list，统计领域直接生成和跨领域生成的目标数
def cross_domain_ratio(all_goal_list):
    result = {'单领域生成': 0, '跨领域生成': 0}
    for goal_list in all_goal_list:
        for goal in goal_list['goals']:
            if goal['生成方式'] == '单领域生成':
                result['单领域生成'] += 1
            else:
                result['跨领域生成'] += 1
    total_goal_num = sum([result[item] for item in result])
    result['单领域生成'] = result['单领域生成'] / total_goal_num
    result['跨领域生成'] = result['跨领域生成'] / total_goal_num
    return result

# 给定all_goal_list，统计小目标分布及平均数
def num_count(all_goal_list):
    result = {'小目标数的分布': [], '平均小目标数': 0}
    for goal_list in all_goal_list:
        result['小目标数的分布'].append(len(goal_list['goals']))
    result['平均小目标数'] = sum(result['小目标数的分布']) / len(all_goal_list)
    result['小目标数的分布'] = Counter(result['小目标数的分布'])
    return result

# 给定all_goal_list，统计所有信息
def total_count(all_goal_list):
    split_data = method_split(all_goal_list)
    result = {
        '单领域目标生成': {
            '先验概率': len(split_data[0]) / len(all_goal_list)
        },
        '多领域单独': {
            '先验概率': len(split_data[1]) / len(all_goal_list)
        },
        '多领域可跨领域': {
            '先验概率': len(split_data[2]) / len(all_goal_list)
        }
    }
    index = 0
    for method in result:
        if len(split_data[index]) != 0:
            result[method].update(num_count(split_data[index]))
            result[method].update(domain_count(split_data[index]))
        index += 1

    return result


if __name__ == '__main__':
    pprint(GoalGenerator.generate())
