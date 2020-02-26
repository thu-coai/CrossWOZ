# -*- coding: utf-8 -*-
"""
@author: truthless
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn

from convlab2.util.train_util import to_device
from convlab2.policy.vhus.usermodule import VHUS
from convlab2.policy.vhus.util import padding_data, kl_gaussian

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_iter(x, y, z, batch_size=64):
    data_len = len(x)
    num_batch = ((data_len - 1) // batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]
    z_shuffle = np.array(z)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], z_shuffle[start_id:end_id]

class VHUS_Trainer():
    def __init__(self, config, manager, goal_gen):
        
        voc_goal_size, voc_usr_size, voc_sys_size = manager.get_voc_size()
        self.user = VHUS(config, voc_goal_size, voc_usr_size, voc_sys_size).to(device=DEVICE)
        self.goal_gen = goal_gen
        self.manager = manager
        
        self.print_per_batch = config['print_per_batch']
        self.save_dir = config['save_dir']
        self.save_per_epoch = config['save_per_epoch']
        seq_goals, seq_usr_dass, seq_sys_dass = manager.data_loader_seg()
        train_goals, train_usrdas, train_sysdas, \
        test_goals, test_usrdas, test_sysdas, \
        val_goals, val_usrdas, val_sysdas = manager.train_test_val_split_seg(
            seq_goals, seq_usr_dass, seq_sys_dass)
        self.data_train = (train_goals, train_usrdas, train_sysdas, config['batchsz'])
        self.data_valid = (val_goals, val_usrdas, val_sysdas, config['batchsz'])
        self.data_test = (test_goals, test_usrdas, test_sysdas, config['batchsz'])
        self.alpha = config['alpha']
        self.optim = torch.optim.Adam(self.user.parameters(), lr=config['lr'])
        self.nll_loss = nn.NLLLoss(ignore_index=0) # PAD=0
        self.bce_loss = nn.BCEWithLogitsLoss()        
            
    def user_loop(self, data):
        batch_input = to_device(padding_data(data))
        a_weights, t_weights, argu = self.user(batch_input['goals'], batch_input['goals_length'], \
                                         batch_input['posts'], batch_input['posts_length'], batch_input['origin_responses'])
        
        loss_a, targets_a = 0, batch_input['origin_responses'][:, 1:] # remove sos_id
        for i, a_weight in enumerate(a_weights):
            loss_a += self.nll_loss(a_weight, targets_a[:, i])
        loss_a /= i
        loss_t = self.bce_loss(t_weights, batch_input['terminated'])
        loss_a += self.alpha * kl_gaussian(argu)
        return loss_a, loss_t
        
    def imitating(self, epoch):
        """
        train the user simulator by simple imitation learning (behavioral cloning)
        """
        self.user.train()
        a_loss, t_loss = 0., 0.
        data_train_iter = batch_iter(self.data_train[0], self.data_train[1], self.data_train[2], self.data_train[3])
        for i, data in enumerate(data_train_iter):
            self.optim.zero_grad()
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            loss = loss_a + loss_t
            loss.backward()
            self.optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                t_loss /= self.print_per_batch
                logging.debug('<<user simulator>> epoch {}, iter {}, loss_a:{}, loss_t:{}'.format(epoch, i, a_loss, t_loss))
                a_loss, t_loss = 0., 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.user.eval()
        
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the user simulator fit on the training dataset
        """        
        a_loss, t_loss = 0., 0.
        data_valid_iter = batch_iter(self.data_valid[0], self.data_valid[1], self.data_valid[2], self.data_valid[3])
        for i, data in enumerate(data_valid_iter):
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            
        a_loss /= i+1
        t_loss /= i+1
        logging.debug('<<user simulator>> validation, epoch {}, loss_a:{}, loss_t:{}'.format(epoch, a_loss, t_loss))
        loss = a_loss + t_loss
        if loss < best:
            logging.info('<<user simulator>> best model saved')
            best = loss
            self.save(self.save_dir, 'best')
            
        a_loss, t_loss = 0., 0.
        data_test_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        for i, data in enumerate(data_test_iter):
            loss_a, loss_t = self.user_loop(data)
            a_loss += loss_a.item()
            t_loss += loss_t.item()
            
        a_loss /= i+1
        t_loss /= i+1
        logging.debug('<<user simulator>> test, epoch {}, loss_a:{}, loss_t:{}'.format(epoch, a_loss, t_loss))
        return best
		
    def test(self):
        def sequential(da_seq):
            da = []
            cur_act = None
            for word in da_seq:
                if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '(', ')']:
                    continue
                if '-' in word:
                    cur_act = word
                else:
                    if cur_act is None:
                        continue
                    da.append(cur_act+'-'+word)
            return da
            
        def f1(pred, real):
            if not real:
                return 0, 0, 0
            TP, FP, FN = 0, 0, 0
            for item in real:
                if item in pred:
                    TP += 1
                else:
                    FN += 1
            for item in pred:
                if item not in real:
                    FP += 1
            return TP, FP, FN
    
        data_test_iter = batch_iter(self.data_test[0], self.data_test[1], self.data_test[2], self.data_test[3])
        a_TP, a_FP, a_FN, t_corr, t_tot = 0, 0, 0, 0, 0
        eos_id = self.user.usr_decoder.eos_id
        for i, data in enumerate(data_test_iter):
            batch_input = to_device(padding_data(data))
            a_weights, t_weights, argu = self.user(batch_input['goals'], batch_input['goals_length'], \
                                         batch_input['posts'], batch_input['posts_length'], batch_input['origin_responses'])
            usr_a = []
            for a_weight in a_weights:
                usr_a.append(a_weight.argmax(1).cpu().numpy())
            usr_a = np.array(usr_a).T.tolist()
            a = []
            for ua in usr_a:
                if eos_id in ua:
                    ua = ua[:ua.index(eos_id)]
                a.append(sequential(self.manager.id2sentence(ua)))
            targets_a = []
            for ua_sess in data[1]:
                for ua in ua_sess:
                    targets_a.append(sequential(self.manager.id2sentence(ua[1:-1])))
            TP, FP, FN = f1(a, targets_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN
                    
            t = t_weights.ge(0).cpu().tolist()
            targets_t = batch_input['terminated'].cpu().long().tolist()
            judge = np.array(t) == np.array(targets_t)
            t_corr += judge.sum()
            t_tot += judge.size

        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)
        print(t_corr, t_tot, t_corr/t_tot)
        
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.user.state_dict(), directory + '/' + str(epoch) + '_simulator.mdl')
        logging.info('<<user simulator>> epoch {}: saved network to mdl'.format(epoch))
