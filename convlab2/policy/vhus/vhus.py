# -*- coding: utf-8 -*-
import os
import torch
import zipfile
from convlab2.util.file_util import cached_path
from convlab2.policy.policy import Policy
from convlab2.policy.vhus.util import padding

class UserPolicyVHUSAbstract(Policy):

    def __init__(self, archive_file, model_file):
        self.user = None
        self.goal_gen = None
        self.manager = None
        
    def init_session(self):
        self.time_step = -1
        self.topic = 'NONE'
        self.goal = self.goal_gen.get_user_goal()
        self.goal_input = torch.LongTensor(self.manager.get_goal_id(self.manager.usrgoal2seq(self.goal)))
        self.goal_len_input = torch.LongTensor([len(self.goal_input)]).squeeze()
        self.sys_da_id_stack = []  # to save sys da history
        self.terminated = False

    def predict(self, state):
        """Predict an user act based on state and preorder system action.

        Args:
            state (tuple):
                Dialog state.
        Returns:
            usr_action (tuple):
                User act.
            session_over (boolean):
                True to terminate session, otherwise session continues.
        """
        sys_action = state
		
        sys_seq_turn = self.manager.sysda2seq(self.manager.ref_data2stand(sys_action), self.goal)
        self.sys_da_id_stack += self.manager.get_sysda_id([sys_seq_turn])
        sys_seq_len = torch.LongTensor([max(len(sen), 1) for sen in self.sys_da_id_stack])
        max_sen_len = sys_seq_len.max().item()
        sys_seq = torch.LongTensor(padding(self.sys_da_id_stack, max_sen_len))
        usr_a, terminated = self.user.select_action(self.goal_input, self.goal_len_input, sys_seq, sys_seq_len)
        usr_action = self.manager.usrseq2da(self.manager.id2sentence(usr_a), self.goal)
        self.terminated = terminated
        return usr_action

    def load(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for VHUS Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_simulator.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        
        user_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_simulator.mdl')
        if os.path.exists(user_mdl):
            self.user.load_state_dict(torch.load(user_mdl))
            print('<<user simulator>> loaded checkpoint from file: {}'.format(user_mdl))

    def get_goal(self):
        return self.goal

    def is_terminated(self):
        # Is there any action to say?
        return self.terminated
