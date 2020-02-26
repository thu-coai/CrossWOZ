import os
import json
import numpy as np
from convlab2.policy.vec import Vector
from convlab2.util.crosswoz.state import default_state
from convlab2.util.crosswoz.lexicalize import delexicalize_da, lexicalize_da
from convlab2.util.crosswoz.dbquery import Database


class CrossWozVector(Vector):
    def __init__(self, sys_da_voc_json, usr_da_voc_json):
        self.sys_da_voc = json.load(open(sys_da_voc_json))
        self.usr_da_voc = json.load(open(usr_da_voc_json))
        self.database = Database()
        
        self.generate_dict()
        
    def generate_dict(self):
        self.sys_da2id = dict((a, i) for i, a in enumerate(self.sys_da_voc))
        self.id2sys_da = dict((i, a) for i, a in enumerate(self.sys_da_voc))
        
        # 155
        self.sys_da_dim = len(self.sys_da_voc)
        
        
        self.usr_da2id = dict((a, i) for i, a in enumerate(self.usr_da_voc))
        self.id2usr_da = dict((i, a) for i, a in enumerate(self.usr_da_voc))
        
        # 142
        self.usr_da_dim = len(self.usr_da_voc)

        # 26
        self.belief_state_dim = 0
        for domain, svs in default_state()['belief_state'].items():
            self.belief_state_dim += len(svs)

        self.db_res_dim = 4

        self.state_dim = self.sys_da_dim + self.usr_da_dim + self.belief_state_dim + self.db_res_dim + 1 # terminated

    def state_vectorize(self, state):
        self.belief_state = state['belief_state']
        self.cur_domain = state['cur_domain']

        da = state['user_action']
        da = delexicalize_da(da)
        usr_act_vec = np.zeros(self.usr_da_dim)
        for a in da:
            if a in self.usr_da2id:
                usr_act_vec[self.usr_da2id[a]] = 1.

        da = state['system_action']
        da = delexicalize_da(da)
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.
                
        belief_state_vec = np.zeros(self.belief_state_dim)
        i = 0
        for domain, svs in state['belief_state'].items():
            for slot, value in svs.items():                
                if value:
                    belief_state_vec[i] = 1.
                i += 1

        self.db_res = self.database.query(state['belief_state'], state['cur_domain'])
        db_res_num = len(self.db_res)
        db_res_vec = np.zeros(4)
        if db_res_num == 0:
            db_res_vec[0] = 1.
        elif db_res_num == 1:
            db_res_vec[1] = 1.
        elif 1 < db_res_num < 5:
            db_res_vec[2] = 1.
        else:
            db_res_vec[3] = 1.
            
        terminated = 1. if state['terminated'] else 0.
            
        # print('state dim', self.state_dim)
        state_vec = np.r_[usr_act_vec, sys_act_vec, belief_state_vec, db_res_vec, terminated]
        # print('actual state vec dim', len(state_vec))
        return state_vec
    
    def action_devectorize(self, action_vec):
        """
        must call state_vectorize func before
        :param action_vec:
        :return:
        """
        da = []
        for i, idx in enumerate(action_vec):
            if idx == 1:
                da.append(self.id2sys_da[i])
        lexicalized_da = lexicalize_da(da=da, cur_domain=self.cur_domain, entities=self.db_res)
        return lexicalized_da
    
    def action_vectorize(self, da):
        da = delexicalize_da(da)
        sys_act_vec = np.zeros(self.sys_da_dim)
        for a in da:
            if a in self.sys_da2id:
                sys_act_vec[self.sys_da2id[a]] = 1.
        return sys_act_vec


if __name__ == '__main__':
    vec = CrossWozVector('../../../data/crosswoz/sys_da_voc.json','../../../data/crosswoz/usr_da_voc.json')
    print(vec.sys_da_dim, vec.usr_da_dim, vec.belief_state_dim, vec.db_res_dim, vec.state_dim)
