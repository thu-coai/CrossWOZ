# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:27:34 2019

@author: truthless
"""

class Environment():
    
    def __init__(self, sys_nlg, usr, sys_nlu, sys_dst):
        self.sys_nlg = sys_nlg
        self.usr = usr
        self.sys_nlu = sys_nlu
        self.sys_dst = sys_dst
        
    def reset(self):
        self.usr.init_session()
        self.sys_dst.init_session()
        return self.sys_dst.state
        
    def step(self, action):
        model_response = self.sys_nlg.generate(action) if self.sys_nlg else action
        observation = self.usr.response(model_response)
        dialog_act = self.sys_nlu.predict(observation) if self.sys_nlu else observation
        state = self.sys_dst.update(dialog_act)
        
        reward = self.usr.get_reward()
        terminated = self.usr.is_terminated()
        
        return state, reward, terminated
