import os
import torch
import logging
import torch.nn as nn

from convlab2.util.train_util import to_device

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLE_Trainer_Abstract():
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        self.policy = None
        self.policy_optim = None
        
    def _init_data(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
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
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
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

