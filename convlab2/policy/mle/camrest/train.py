import os
import torch
import logging
import json
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_camrest import CamrestVector
from convlab2.policy.mle.train import MLE_Trainer_Abstract
from convlab2.policy.mle.multiwoz.loader import ActPolicyDataLoaderCamrest
from convlab2.util.train_util import init_logging_handler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLE_Trainer(MLE_Trainer_Abstract):
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
        vector = CamrestVector(voc_file, voc_opp_file)
        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.da_dim).to(device=DEVICE)
        self.policy.eval()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])
        
if __name__ == '__main__':
    manager = ActPolicyDataLoaderCamrest()
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    init_logging_handler(cfg['log_dir'])
    agent = MLE_Trainer(manager, cfg)
    
    logging.debug('start training')
    
    best = float('inf')
    for e in range(cfg['epoch']):
        agent.imitating(e)
        best = agent.imit_test(e, best)
