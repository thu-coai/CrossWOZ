import argparse
import logging
import random
import time
import json

import numpy as np
import torch
from nltk import word_tokenize
from torch.autograd import Variable
from torch.optim import Adam

from convlab2.e2e.sequicity.config import global_config as cfg
from convlab2.e2e.sequicity.metric import CamRestEvaluator, KvretEvaluator, MultiWozEvaluator
from convlab2.e2e.sequicity.reader import CamRest676Reader, KvretReader, MultiWozReader
from convlab2.e2e.sequicity.reader import get_glove_matrix
from convlab2.e2e.sequicity.reader import pad_sequences
from convlab2.e2e.sequicity.tsd_net import TSD, cuda_


class Model:
    def __init__(self, dataset):
        reader_dict = {
            'camrest': CamRest676Reader,
            'kvret': KvretReader,
            'multiwoz': MultiWozReader
        }
        model_dict = {
            'TSD':TSD
        }
        evaluator_dict = {
            'camrest': CamRestEvaluator,
            'kvret': KvretEvaluator,
            'multiwoz': MultiWozEvaluator
        }
        self.reader = reader_dict[dataset]()
        self.m = model_dict[cfg.m](embed_size=cfg.embedding_size,
                               hidden_size=cfg.hidden_size,
                               vocab_size=cfg.vocab_size,
                               layer_num=cfg.layer_num,
                               dropout_rate=cfg.dropout_rate,
                               z_length=cfg.z_length,
                               max_ts=cfg.max_ts,
                               beam_search=cfg.beam_search,
                               beam_size=cfg.beam_size,
                               eos_token_idx=self.reader.vocab.encode('EOS_M'),
                               vocab=self.reader.vocab,
                               teacher_force=cfg.teacher_force,
                               degree_size=cfg.degree_size)
        self.EV = evaluator_dict[dataset] # evaluator class
        if cfg.cuda: self.m = self.m.cuda()
        self.base_epoch = -1

    def _convert_batch(self, py_batch, prev_z_py=None):
        u_input_py = py_batch['user']
        u_len_py = py_batch['u_len']
        kw_ret = {}
        if cfg.prev_z_method == 'concat' and prev_z_py is not None:
            for i in range(len(u_input_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    u_input_py[i] = prev_z_py[i][:idx + 1] + u_input_py[i]
                else:
                    u_input_py[i] = prev_z_py[i] + u_input_py[i]
                u_len_py[i] = len(u_input_py[i])
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
        elif cfg.prev_z_method == 'separate' and prev_z_py is not None:
            for i in range(len(prev_z_py)):
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in prev_z_py[i] and prev_z_py[i].index(eob) != len(prev_z_py[i]) - 1:
                    idx = prev_z_py[i].index(eob)
                    prev_z_py[i] = prev_z_py[i][:idx + 1]
                for j, word in enumerate(prev_z_py[i]):
                    if word >= cfg.vocab_size:
                        prev_z_py[i][j] = 2 #unk
            prev_z_input_np = pad_sequences(prev_z_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in prev_z_py])
            prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
            kw_ret['prev_z_len'] = prev_z_len
            kw_ret['prev_z_input'] = prev_z_input
            kw_ret['prev_z_input_np'] = prev_z_input_np

        degree_input_np = np.array(py_batch['degree'])
        u_input_np = pad_sequences(u_input_py, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
        z_input_np = pad_sequences(py_batch['bspan'], padding='post').transpose((1, 0))
        m_input_np = pad_sequences(py_batch['response'], cfg.max_ts, padding='post', truncating='post').transpose(
            (1, 0))

        u_len = np.array(u_len_py)
        m_len = np.array(py_batch['m_len'])

        degree_input = cuda_(Variable(torch.from_numpy(degree_input_np).float()))
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        z_input = cuda_(Variable(torch.from_numpy(z_input_np).long()))
        m_input = cuda_(Variable(torch.from_numpy(m_input_np).long()))

        kw_ret['z_input_np'] = z_input_np

        return u_input, u_input_np, z_input, m_input, m_input_np,u_len, m_len,  \
               degree_input, kw_ret

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        train_time = 0
        for epoch in range(cfg.epoch_num):
            sw = time.time()
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            self.m.self_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    if cfg.truncated:
                        logging.debug('iter %d turn %d' % (iter_num, turn_num))
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)

                    loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                        m_input=m_input,
                                                                        degree_input=degree_input,
                                                                        u_input_np=u_input_np,
                                                                        m_input_np=m_input_np,
                                                                        turn_states=turn_states,
                                                                        u_len=u_len, m_len=m_len, mode='train', **kw_ret)
                    loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 5.0)
                    optim.step()
                    sup_loss += loss.item()
                    sup_cnt += 1
                    logging.debug(
                        'loss:{} pr_loss:{} m_loss:{} grad:{}'.format(loss.item(),
                                                                       pr_loss.item(),
                                                                       m_loss.item(),
                                                                       grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            train_time += time.time() - sw
            logging.info('Traning time: {}'.format(train_time))
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            logging.info('time for epoch %d: %f' % (epoch, time.time()-sw))
            valid_loss = valid_sup_loss + valid_unsup_loss

            if valid_loss <= prev_min_loss:
                self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        data_iterator = self.reader.mini_batch_iterator(data)
        mode = 'test' if not cfg.pretrain else 'pretrain_test'
        for batch_num, dial_batch in enumerate(data_iterator):
            turn_states = {}
            prev_z = None
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch, prev_z)
                m_idx, z_idx, turn_states = self.m(mode=mode, u_input=u_input, u_len=u_len, z_input=z_input,
                                                   m_input=m_input,
                                                   degree_input=degree_input, u_input_np=u_input_np,
                                                   m_input_np=m_input_np,
                                                   m_len=m_len, turn_states=turn_states,**kw_ret)
                self.reader.wrap_result(turn_batch, m_idx, z_idx, prev_z=prev_z)
                prev_z = z_idx
        ev = self.EV(result_path=cfg.result_path)
        res = ev.run_metrics()
        self.m.train()
        return res
    
    def interact(self):
        def z2degree(gen_z):
            gen_bspan = self.reader.vocab.sentence_decode(gen_z, eos='EOS_Z2')
            constraint_request = gen_bspan.split()
            constraints = constraint_request[:constraint_request.index('EOS_Z1')] if 'EOS_Z1' \
                in constraint_request else constraint_request
            for j, ent in enumerate(constraints):
                constraints[j] = ent.replace('_', ' ')
            degree = self.reader.db_search(constraints)
            degree_input_list = self.reader._degree_vec_mapping(len(degree))
            degree_input = cuda_(Variable(torch.Tensor(degree_input_list).unsqueeze(0)))
            return degree, degree_input
        
        def denormalize(uttr):
            uttr = uttr.replace(' -s', 's')
            uttr = uttr.replace(' -ly', 'ly')
            uttr = uttr.replace(' -er', 'er')
            return uttr
            
        self.m.eval()
        print('Start interaction.')
        kw_ret = dict({'func':z2degree})
        while True:
            usr = input('usr: ')
            if usr == 'END':
                break
            if usr == 'RESET':
                kw_ret = dict({'func':z2degree})
                continue
            usr = word_tokenize(usr.lower())
            usr_words = usr + ['EOS_U']
            u_len = np.array([len(usr_words)])
            usr_indices = self.reader.vocab.sentence_encode(usr_words)
            u_input_np = np.array(usr_indices)[:, np.newaxis]
            u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
            m_idx, z_idx, degree = self.m(mode='test', degree_input=None, z_input=None,
                                          u_input=u_input, u_input_np=u_input_np, u_len=u_len,
                                          m_input=None, m_input_np=None, m_len=None,
                                          turn_states=None, **kw_ret)
            venue = random.sample(degree, 1)[0] if degree else dict()
            l = [self.reader.vocab.decode(_) for _ in m_idx[0]]
            if 'EOS_M' in l:
                l = l[:l.index('EOS_M')]
            l_origin = []
            for word in l:
                if 'SLOT' in word:
                    word = word[:-5]
                    if word in venue.keys():
                        value = venue[word]
                        if value != '?':
                            l_origin.append(value)
                else:
                    l_origin.append(word)
            sys = ' '.join(l_origin)
            sys = denormalize(sys)
            print('sys:', sys)
            if cfg.prev_z_method == 'separate':
                eob = self.reader.vocab.encode('EOS_Z2')
                if eob in z_idx[0] and z_idx[0].index(eob) != len(z_idx[0]) - 1:
                    idx = z_idx[0].index(eob)
                    z_idx[0] = z_idx[0][:idx + 1]
                for j, word in enumerate(z_idx[0]):
                    if word >= cfg.vocab_size:
                        z_idx[0][j] = 2 #unk
                prev_z_input_np = pad_sequences(z_idx, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
                prev_z_len = np.array([len(_) for _ in z_idx])
                prev_z_input = cuda_(Variable(torch.from_numpy(prev_z_input_np).long()))
                kw_ret['prev_z_len'] = prev_z_len
                kw_ret['prev_z_input'] = prev_z_input
                kw_ret['prev_z_input_np'] = prev_z_input_np

    def predict(self, usr, kw_ret):
        def z2degree(gen_z):
            gen_bspan = self.reader.vocab.sentence_decode(gen_z, eos='EOS_Z2')
            constraint_request = gen_bspan.split()
            constraints = constraint_request[:constraint_request.index('EOS_Z1')] if 'EOS_Z1' \
                in constraint_request else constraint_request
            for j, ent in enumerate(constraints):
                constraints[j] = ent.replace('_', ' ')
            degree = self.reader.db_search(constraints)
            degree_input_list = self.reader._degree_vec_mapping(len(degree))
            degree_input = cuda_(Variable(torch.Tensor(degree_input_list).unsqueeze(0)))
            return degree, degree_input
            
        self.m.eval()

        kw_ret['func'] = z2degree
        if 'prev_z_input_np' in kw_ret:
            kw_ret['prev_z_len'] = np.array(kw_ret['prev_z_len'])
            kw_ret['prev_z_input_np'] = np.array(kw_ret['prev_z_input_np'])
            kw_ret['prev_z_input'] = cuda_(Variable(torch.Tensor(kw_ret['prev_z_input_np']).long()))

        usr = word_tokenize(usr.lower())

        usr_words = usr + ['EOS_U']
        u_len = np.array([len(usr_words)])
        usr_indices = self.reader.vocab.sentence_encode(usr_words)
        u_input_np = np.array(usr_indices)[:, np.newaxis]
        u_input = cuda_(Variable(torch.from_numpy(u_input_np).long()))
        m_idx, z_idx, degree = self.m(mode='test', degree_input=None, z_input=None,
                                        u_input=u_input, u_input_np=u_input_np, u_len=u_len,
                                        m_input=None, m_input_np=None, m_len=None,
                                        turn_states=None, **kw_ret)
        venue = random.sample(degree, 1)[0] if degree else dict()
        l = [self.reader.vocab.decode(_) for _ in m_idx[0]]
        if 'EOS_M' in l:
            l = l[:l.index('EOS_M')]
        l_origin = []
        for word in l:
            if 'SLOT' in word:
                word = word[:-5]
                if word in venue.keys():
                    value = venue[word]
                    if value != '?':
                        l_origin.append(value.replace(' ', '_'))
            else:
                l_origin.append(word)
        sys = ' '.join(l_origin)
        kw_ret['sys'] = sys
        if cfg.prev_z_method == 'separate':
            eob = self.reader.vocab.encode('EOS_Z2')
            if eob in z_idx[0] and z_idx[0].index(eob) != len(z_idx[0]) - 1:
                idx = z_idx[0].index(eob)
                z_idx[0] = z_idx[0][:idx + 1]
            for j, word in enumerate(z_idx[0]):
                if word >= cfg.vocab_size:
                    z_idx[0][j] = 2 #unk
            prev_z_input_np = pad_sequences(z_idx, cfg.max_ts, padding='post', truncating='pre').transpose((1, 0))
            prev_z_len = np.array([len(_) for _ in z_idx])
            kw_ret['prev_z_len'] = prev_z_len.tolist()
            kw_ret['prev_z_input_np'] = prev_z_input_np.tolist()
            if 'prev_z_input' in kw_ret:
                del kw_ret['prev_z_input']

        del kw_ret['func']

        return kw_ret

    def validate(self, data='dev'):
        self.m.eval()
        data_iterator = self.reader.mini_batch_iterator(data)
        sup_loss, unsup_loss = 0, 0
        sup_cnt, unsup_cnt = 0, 0
        for dial_batch in data_iterator:
            turn_states = {}
            for turn_num, turn_batch in enumerate(dial_batch):
                u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                m_len, degree_input, kw_ret \
                    = self._convert_batch(turn_batch)

                loss, pr_loss, m_loss, turn_states = self.m(u_input=u_input, z_input=z_input,
                                                                    m_input=m_input,
                                                                    turn_states=turn_states,
                                                                    degree_input=degree_input,
                                                                    u_input_np=u_input_np, m_input_np=m_input_np,
                                                                    u_len=u_len, m_len=m_len, mode='train',**kw_ret)
                sup_loss += loss.item()
                sup_cnt += 1
                logging.debug(
                    'loss:{} pr_loss:{} m_loss:{}'.format(loss.item(), pr_loss.item(), m_loss.item()))

        sup_loss /= (sup_cnt + 1e-8)
        unsup_loss /= (unsup_cnt + 1e-8)
        self.m.train()
        print('result preview...')
        self.eval()
        return sup_loss, unsup_loss

    def reinforce_tune(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        for epoch in range(self.base_epoch + cfg.rl_epoch_num + 1):
            mode = 'rl'
            if epoch <= self.base_epoch:
                continue
            epoch_loss, cnt = 0,0
            data_iterator = self.reader.mini_batch_iterator('train')
            optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()), weight_decay=1e-5)
            for iter_num, dial_batch in enumerate(data_iterator):
                turn_states = {}
                prev_z = None
                for turn_num, turn_batch in enumerate(dial_batch):
                    optim.zero_grad()
                    u_input, u_input_np, z_input, m_input, m_input_np, u_len, \
                    m_len, degree_input, kw_ret \
                        = self._convert_batch(turn_batch, prev_z)
                    loss_rl = self.m(u_input=u_input, z_input=z_input,
                                m_input=m_input,
                                degree_input=degree_input,
                                u_input_np=u_input_np,
                                m_input_np=m_input_np,
                                turn_states=turn_states,
                                u_len=u_len, m_len=m_len, mode=mode, **kw_ret)

                    if loss_rl is not None:
                        loss = loss_rl
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm(self.m.parameters(), 2.0)
                        optim.step()
                        epoch_loss += loss.item()
                        cnt += 1
                        logging.debug('{} loss {}, grad:{}'.format(mode,loss.item(),grad))

                    prev_z = turn_batch['bspan']

            epoch_sup_loss = epoch_loss / (cnt + 1e-8)
            logging.info('avg training loss in epoch %d sup:%f' % (epoch, epoch_sup_loss))

            valid_sup_loss, valid_unsup_loss = self.validate()
            logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))
            valid_loss = valid_sup_loss + valid_unsup_loss

            self.save_model(epoch)

            if valid_loss <= prev_min_loss:
                #self.save_model(epoch)
                prev_min_loss = valid_loss
            else:
                early_stop_count -= 1
                lr *= cfg.lr_decay
                if not early_stop_count:
                    break
                logging.info('early stop countdown %d, learning rate %f' % (early_stop_count, lr))

    def save_model(self, epoch, path=None):
        if not path:
            path = cfg.model_path
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cuda' if cfg.cuda else 'cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        initial_arr = self.m.u_encoder.embedding.weight.data.cpu().numpy()
        embedding_arr = torch.from_numpy(get_glove_matrix(self.reader.vocab, initial_arr))

        self.m.u_encoder.embedding.weight.data.copy_(embedding_arr)
        self.m.z_decoder.emb.weight.data.copy_(embedding_arr)
        self.m.m_decoder.emb.weight.data.copy_(embedding_arr)

    def count_params(self):

        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = sum([np.prod(p.size()) for p in module_parameters])

        print('total trainable params: %d' % param_cnt)


def main(arg_mode=None, arg_model=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('-mode')
    parser.add_argument('-model')
    # parser.add_argument('-cfg', nargs='*')
    parser.add_argument('-cfg')
    args = parser.parse_args()

    if arg_mode is not None:
        args.mode = arg_mode
    if arg_model is not None:
        args.model = arg_model

    # cfg.init_handler(args.model)
    c = json.load(open(args.cfg))
    cfg.init_handler(c['tsdf_init'])

    # if args.cfg:
    #     for pair in args.cfg:
    #         k, v = tuple(pair.split('='))
    #         dtype = type(getattr(cfg, k))
    #         if isinstance(None, dtype):
    #             raise ValueError()
    #         if dtype is bool:
    #             v = False if v == 'False' else True
    #         else:
    #             v = dtype(v)
    #         setattr(cfg, k, v)

    logging.debug(str(cfg))
    if cfg.cuda:
        logging.debug('Device: {}'.format(torch.cuda.current_device()))
    cfg.mode = args.mode

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # m = Model(args.model.split('-')[-1])
    m = Model(args.model)
    m.count_params()
    if args.mode == 'train':
        m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model()
        m.train()
    elif args.mode == 'test':
        m.load_model()
        m.eval()
    elif args.mode == 'rl':
        m.load_model()
        m.reinforce_tune()
    elif args.mode == 'interact':
        m.load_model()
        m.interact()
    elif args.mode == 'load':
        m.load_model()
        return m

if __name__ == '__main__':
    main()
