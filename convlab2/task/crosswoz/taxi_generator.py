# -*- coding: utf-8 -*-
import random

import numpy as np


class TaxiGenerator:
    def generate(self, goal_list, goal_num, random_seed=None):
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
        goal = {
            "领域": "出租",
            "id": goal_num,
            "约束条件": [],
            "需求信息": [],
            "生成方式": ""
        }
        goal1, goal2 = random.sample(goal_list, k=2)
        goal["约束条件"].append(["出发地", "id=%d" % goal1["id"]])
        goal["约束条件"].append(["目的地", "id=%d" % goal2["id"]])
        goal["需求信息"].append(["车型", ""])
        goal["需求信息"].append(["车牌", ""])
        # goal["需求信息"].append(["距离", ""])
        # goal["需求信息"].append(["电话", ""])

        if goal1["领域"] == goal2["领域"]:
            goal["生成方式"] = "同领域"
        else:
            goal["生成方式"] = "不同领域"

        return goal
