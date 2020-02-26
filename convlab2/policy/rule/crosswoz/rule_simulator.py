from convlab2.task.crosswoz.goal_generator import GoalGenerator
from convlab2.policy.policy import Policy
from pprint import pprint
import random
from copy import deepcopy
import re


class Simulator(Policy):
    def __init__(self):
        self.goal_generator = GoalGenerator()
        self.goal, self.state, self.sub_goals, self.sub_goals_state = None, None, None, None
        self.original_goal, self.goal_type = None, None
        self.max_turn = 40
        self.turn_num = 0
        self.da_seq = []

    def infer_goal_type(self, goal):
        sub_goals = list({x[0]: x[1] for x in goal}.values())
        if goal[-1][0] == 1:
            return '单领域'
        elif '出租' not in sub_goals and '地铁' not in sub_goals:
            goal_content = ' '.join([x[3] for x in goal if not isinstance(x[3], list)])
            if "出现在id" in goal_content:
                return '不独立多领域'
            else:
                return '独立多领域'
        else:
            goal_content = ' '.join([x[3] for x in goal if not isinstance(x[3], list) and x[1] not in ['出租', '地铁']])
            if "出现在id" in goal_content:
                return '不独立多领域+交通'
            else:
                return '独立多领域+交通'

    def init_session(self, goal=None, state=None, turn_num=0, da_seq=list()):
        self.goal = self.goal_generator.generate() if not goal else goal
        self.goal_type = self.infer_goal_type(self.goal)
        self.original_goal = deepcopy(self.goal)
        self.state = deepcopy(self.goal) if not state else state
        self.sub_goals = [None]+[[] for _ in range(self.goal[-1][0])]
        for t in self.goal:
            sub_goal_id = t[0]
            self.sub_goals[sub_goal_id].append(t)
        self.sub_goals_state = [None] + [[] for _ in range(self.goal[-1][0])]
        for t in self.state:
            sub_goal_id = t[0]
            self.sub_goals_state[sub_goal_id].append(t)
        self.turn_num = turn_num
        self.da_seq = da_seq

    def predict(self, sys_act):
        if self.turn_num==0:
            da = self.begin_da()
            self.turn_num+=1
            self.da_seq.append(da)
            return da
        else:
            self.da_seq.append(sys_act)
            self.turn_num+=1
            self.state_update(self.da_seq[-2], self.da_seq[-1])
            da = self.state_predict()
            self.da_seq.append(da)
            self.turn_num += 1
            return da

    def begin_da(self):
        active_tuples_num = random.choices([2, 3, 4], [30, 50, 20])[0]
        active_tuples_num = min(active_tuples_num, len(self.sub_goals[1]))
        active_tuples = self.state[:active_tuples_num]
        for i in range(active_tuples_num):
            self.state[i][-1] = True

        # given state predict da: f1 97.8% (by stat4state:test_begin_da_predict)
        da = [['General', 'greet', 'none', 'none']]
        for intent, domain, slot, value, expressed in active_tuples:
            if not value:
                da.append(['Request', domain, slot, ''])
            elif isinstance(value, str):
                da.append(['Inform', domain, slot, value])
            elif isinstance(value, list):
                for v in value:
                    da.append(['Inform', domain, slot, v])
            else:
                assert 0
        return sorted(da)

    def state_update(self, prev_user_da, prev_sys_da):
        if self.is_terminated():
            return deepcopy(self.state)
        # cur sub goal state
        cur_sub_goal_state = None
        cur_sub_goal = None
        cur_domain = None
        for sub_goal_state in self.sub_goals_state[1:]:
            for sub_goal_id, domain, slot, value, expressed in sub_goal_state:
                if not expressed or not value or 'id' in value:
                    break
            else:
                continue
            cur_sub_goal_state = sub_goal_state
            cur_sub_goal = self.sub_goals[cur_sub_goal_state[0][0]]
            cur_domain = cur_sub_goal_state[0][1]
            break
        assert cur_sub_goal_state
        # print('cur sub goal state:')
        # pprint(cur_sub_goal_state)

        # prev user request
        user_request = []
        user_inform = []
        user_select = []
        sys_inform = []
        sys_recommend = []
        sys_nooffer = False
        for intent, domain, slot, value in prev_user_da:
            if intent=='Request':
                user_request.append([domain, slot])
            elif intent=='Inform':
                user_inform.append([domain, slot, value])
            elif intent=='Select':
                user_select.append([domain, slot, value])

        for intent, domain, slot, value in prev_sys_da:
            if intent=='Inform':
                sys_inform.append([domain, slot, value])
            elif intent=='Recommend':
                sys_recommend.append([domain, slot, value])
            elif intent=='NoOffer':
                sys_nooffer = True

        if sys_nooffer:
            # DONE: nooffer =>  1) accept recommendation (need change constraints) or 2) alter the constraint (need re-express)

            alter_recommend = ''
            for domain, slot, value in sys_inform:
                if domain==cur_domain and slot=='名称':
                    alter_recommend = value
                    break
            for domain, slot, value in sys_recommend:
                if domain == cur_domain and slot == '名称':
                    alter_recommend = value
                    break

            # pprint(cur_sub_goal)
            if alter_recommend:
                # 1) accept recommendation (need change constraints)
                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    if (x[1],x[2]) in [(z[0],z[1]) for z in sys_inform]:
                        # slot in sys dialog act
                        if isinstance(x[3],list):
                            y[3] = []
                        for domain, slot, value in sys_inform:
                            if domain == cur_domain and slot == x[2]:
                                if slot=='推荐菜' or '周边' in slot:
                                    if value=='无':
                                        y[3] = value
                                        y[4] = True
                                        break
                                    else:
                                        # list type
                                        y[3].append(value)
                                        y[4] = True
                                elif x[3] and 'id' not in x[3]:
                                    # original constraints x[3]
                                    if value=='无':
                                        y[3] = value
                                        y[4] = True
                                        break
                                    elif slot=='评分':
                                        original = float(re.match('(\d\.\d|\d)', x[3])[0])
                                        if re.match('(\d\.\d|\d)', value):
                                            cur = float(re.match('(\d\.\d|\d)', value)[0])
                                            if cur<original:
                                                y[3] = value
                                                y[4] = True
                                                break
                                        else:
                                            y[3] = value
                                            y[4] = True
                                            break
                                    elif slot=='价格' or slot=='人均消费' or slot=='门票':
                                        if not (re.match('\d+', x[3]) and re.match('\d+', value)):
                                            y[3] = value
                                            y[4] = True
                                            break
                                        if re.match('(\d+)-(\d+)', x[3]):
                                            low = float(re.match('(\d+)-(\d+)', x[3])[1])
                                            high = float(re.match('(\d+)-(\d+)', x[3])[2])
                                        else:
                                            assert re.match('\d+', x[3])
                                            low = float(re.match('\d+', x[3])[0])
                                            high = 10000
                                        cur = float(re.match('\d+', value)[0])
                                        if cur < low or cur > high:
                                            y[3] = value
                                            y[4] = True
                                            break
                                    else:
                                        y[3] = value
                                        y[4] = True
                                        break
                                else:
                                    # request or cross constraint
                                    y[3] = value
                                    y[4] = True
                                    break

                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    if x[2] == '名称' and (not x[3] or 'id' in x[3]):
                        candidates = []
                        for domain, slot, value in sys_recommend:
                            if domain == cur_domain and slot == x[2]:
                                candidates.append(value)
                        if candidates:
                            y[3] = random.choice(candidates)
                            y[4] = True
            else:
                # 2) alter the constraint (need re-express)
                normal_constraints = []
                cross_constraints = []
                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    original_constraint = x[3]
                    expressed = y[4]
                    if original_constraint and expressed:
                        if 'id' in original_constraint:
                            cross_constraints.append(x)
                        else:
                            normal_constraints.append(x)
                random.shuffle(normal_constraints)
                expressed_constraints = normal_constraints+cross_constraints
                if len(expressed_constraints) > 0:
                    if isinstance(expressed_constraints[0][3], str):
                        expressed_constraints[0][3] = ''
                    else:
                        expressed_constraints[0][3] = []
                # clean expressed tag, init as goal, need re-express whole sub-goal
                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    y[3] = x[3]
                    y[4] = False

        else:
            # 1) check if all constraints have been expressed
            all_constraints_expressed = True
            for i, (sub_goal_id, domain, slot, value, expressed) in enumerate(cur_sub_goal):
                # constraint in goal but have not expressed
                if value and not cur_sub_goal_state[i][-1]:
                    all_constraints_expressed = False
                    break

            if not all_constraints_expressed:
                # 2) if all constraints have not been expressed, do nothing
                pass
            else:
                # 2) if all constraints have been expressed, fill request slot
                # TODO: explore different strategy:
                # a) fill all that inform;
                # b) fill request slot;
                # c) fill not filled slot;
                # d) fill what user reqeust
                # current: b
                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    if (not x[3] or 'id' in x[3]) and (x[1],x[2]) in [(z[0],z[1]) for z in sys_inform]:
                        # require slot & in sys dialog act
                        if isinstance(x[3],list):
                            y[3] = []
                        for domain, slot, value in sys_inform:
                            if domain == cur_domain and slot == x[2]:
                                if (slot=='推荐菜' or '周边' in slot) and value!='无':
                                    # list type
                                    y[3].append(value)
                                else:
                                    y[3] = value
                                y[4] = True

                for x, y in zip(cur_sub_goal, cur_sub_goal_state):
                    if x[2] == '名称' and (not x[3] or 'id' in x[3]):
                        candidates = []
                        for domain, slot, value in sys_recommend:
                            if domain == cur_domain and slot == x[2]:
                                candidates.append(value)
                        if candidates:
                            y[3] = random.choice(candidates)
                            y[4] = True

        return deepcopy(self.state)

    def state_predict(self):
        if self.is_terminated():
            da = [['General', 'thank', 'none', 'none']]
            if random.random() < 0.05:
                da.append(['General', 'bye', 'none', 'none'])
            return sorted(da)
        else:
            # DONE: predict according to state
            # cur sub goal state
            cur_sub_goal_state = None
            # cur_sub_goal = None
            # cur_domain = None
            for sub_goal_state in self.sub_goals_state[1:]:
                for sub_goal_id, domain, slot, value, expressed in sub_goal_state:
                    if not expressed or not value or 'id' in value:
                        break
                else:
                    continue
                cur_sub_goal_state = sub_goal_state
                # cur_sub_goal = self.sub_goals[cur_sub_goal_state[0][0]]
                # cur_domain = cur_sub_goal_state[0][1]
                break
            assert cur_sub_goal_state
            # priority: cross constraint > normal constraint > require
            cross_constraints = []
            normal_constraints = []
            requires = []
            for t in cur_sub_goal_state:
                if not t[4]:
                    # have not been expressed
                    if t[3]:
                        # constraints
                        if isinstance(t[3], str):
                            if 'id' in t[3]:
                                ref_id = int(t[3][t[3].index('id')+3])
                                # find ref entity
                                for x in self.sub_goals_state[ref_id]:
                                    if x[2] == '名称':
                                        if t[2] == '名称':
                                            cross_constraints.append([t, ['Select', t[1], '源领域', x[1]]])
                                            cross_constraints.append([t, ['Inform', x[1], '名称', x[3]]])
                                        else:
                                            cross_constraints.append([t, ['Inform', t[1], t[2], x[3]]])
                                        break
                            else:
                                normal_constraints.append([t, ['Inform', t[1], t[2], t[3]]])
                        else:
                            for v in t[3]:
                                normal_constraints.append([t, ['Inform', t[1], t[2], v]])
                    else:
                        # requirements
                        requires.append([t, ['Request', t[1], t[2], '']])
                elif not t[3]:
                    requires.append([t, ['Request', t[1], t[2], '']])
            # DONE: update state, issue da
            da_max_num = random.choices([1, 2, 3, 4], [40, 25, 15, 10])[0]
            da = []
            for t, act in cross_constraints:
                t[4] = True
                if t[2] in ['出发地','目的地']:
                    t[3] = act[3]
                da.append(act)
            queue = normal_constraints + requires
            for t, act in queue:
                if len(da) > da_max_num:
                    break
                else:
                    t[4] = True
                    da.append(act)

            return sorted(da)

    def is_terminated(self):
        for sub_goal_id, domain, slot, value, expressed in self.state:
            if not expressed or not value or 'id' in value:
                break
        else:
            return True
        return False
    
    def get_goal(self):
        return self.goal

    def get_reward(self):
        if self.is_terminated():
            return 2.0 * self.max_turn
        else:
            return -1.0


if __name__ == '__main__':
    simulator = Simulator()
    simulator.init_session()
    pprint(simulator.goal)
    pprint(simulator.predict([]))
    pprint(simulator.predict([['Inform','酒店','名称','北京首都宾馆']]))
    pprint(simulator.state)
