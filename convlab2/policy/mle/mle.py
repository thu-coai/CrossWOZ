# -*- coding: utf-8 -*-
import torch
import os
import zipfile
from convlab2.policy.policy import Policy
from convlab2.util.file_util import cached_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLEAbstract(Policy):

    def __init__(self, archive_file, model_file):
        self.vector = None
        self.policy = None

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        s_vec = torch.Tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=DEVICE), False).cpu()
        action = self.vector.action_devectorize(a.numpy())
        state['system_action'] = action
        return action

    def init_session(self):
        """
        Restore after one session
        """
        pass

    def load(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MLE Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_mle.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            print('<<dialog policy>> loaded checkpoint from file: {}'.format(policy_mdl))
