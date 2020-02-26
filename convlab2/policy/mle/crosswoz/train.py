import os
import torch
import logging
import torch.nn as nn
import json
import pickle
import sys
import random
import numpy as np

root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)

from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_crosswoz import CrossWozVector
from convlab2.policy.mle.crosswoz.loader import PolicyDataLoaderCrossWoz
from convlab2.util.train_util import to_device, init_logging_handler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLE_Trainer():
    def __init__(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']

        voc_file = os.path.join(root_dir, 'data/crosswoz/sys_da_voc.json')
        voc_opp_file = os.path.join(root_dir, 'data/crosswoz/usr_da_voc.json')
        vector = CrossWozVector(voc_file, voc_opp_file)
        self.policy = MultiDiscretePolicy(vector.state_dim, cfg['h_dim'], vector.sys_da_dim).to(device=DEVICE)
        self.policy.eval()
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg['lr'])
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()

    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)

        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()

            if (i + 1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.eval()

    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy>> validation, epoch {}, loss_a:{}'.format(epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = a_loss
            self.save(self.save_dir, 'best')

        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()

        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy>> test, epoch {}, loss_a:{}'.format(epoch, a_loss))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            # print(real)
            # print(predict)
            # print()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN

        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            # TODO: fix batch F1
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN

        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename='save/best'):
        policy_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_mle.pol.mdl')
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))


if __name__ == '__main__':
    random_seed = 2019
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    manager = PolicyDataLoaderCrossWoz()
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    init_logging_handler(cfg['log_dir'])
    agent = MLE_Trainer(manager, cfg)
    agent.load()

    logging.debug('start training')

    best = float('inf')
    for e in range(cfg['epoch']):
        agent.imitating(e)
        best = agent.imit_test(e, best)
    # agent.test() # 5731 1483 1880 0.7731534569983137
