# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import json
import logging
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.util.train_util import init_logging_handler
from convlab2.task.camrest.goal_generator import GoalGenerator
from convlab2.policy.vhus.camrest.usermanager import UserDataManager
from convlab2.policy.vhus.train import VHUS_Trainer

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    init_logging_handler(cfg['log_dir'])
    manager = UserDataManager()
    goal_gen = GoalGenerator()
    env = VHUS_Trainer(cfg, manager, goal_gen)
    
    logging.debug('start training')
    
    best = float('inf')
    for e in range(cfg['epoch']):
        env.imitating(e)
        best = env.imit_test(e, best)
