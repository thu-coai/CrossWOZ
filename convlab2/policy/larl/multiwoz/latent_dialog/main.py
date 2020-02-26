import os
import sys
import numpy as np
import torch as th
from torch import nn
from collections import defaultdict
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.base_modules import summary
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.decoders import TEACH_FORCE, GEN, DecoderRNN
from datetime import datetime
from convlab2.policy.larl.multiwoz.latent_dialog.utils import get_detokenize
from convlab2.policy.larl.multiwoz.latent_dialog.corpora import EOS, PAD
from convlab2.policy.larl.multiwoz.latent_dialog.data_loaders import DealDataLoaders, BeliefDbDataLoaders
from convlab2.policy.larl.multiwoz.latent_dialog import evaluators
from convlab2.policy.larl.multiwoz.latent_dialog.record import record, record_task, UniquenessSentMetric, UniquenessWordMetric
import logging


logger = logging.getLogger()


class LossManager(object):
    def __init__(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_loss(self, loss):
        for key, val in loss.items():
            # print('key = %s\nval = %s' % (key, val))
            if val is not None and type(val) is not bool:
                self.losses[key].append(val.item())

    def pprint(self, name, window=None, prefix=None):
        str_losses = []
        for key, loss in self.losses.items():
            if loss is None:
                continue
            aver_loss = np.average(loss) if window is None else np.average(loss[-window:])
            if 'nll' in key:
                str_losses.append('{} PPL {:.3f}'.format(key, np.exp(aver_loss)))
            else:
                str_losses.append('{} {:.3f}'.format(key, aver_loss))


        if prefix:
            return '{}: {} {}'.format(prefix, name, ' '.join(str_losses))
        else:
            return '{} {}'.format(name, ' '.join(str_losses))

    def clear(self):
        self.losses = defaultdict(list)
        self.backward_losses = []

    def add_backward_loss(self, loss):
        self.backward_losses.append(loss.item())

    def avg_loss(self):
        return np.mean(self.backward_losses)


class Reinforce(object):
    def __init__(self, dialog, ctx_gen, corpus, sv_config, sys_model, usr_model, rl_config, dialog_eval, ctx_gen_eval):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.corpus = corpus
        self.sv_config = sv_config
        self.sys_model = sys_model
        self.usr_model = usr_model
        self.rl_config = rl_config
        self.dialog_eval = dialog_eval
        self.ctx_gen_eval = ctx_gen_eval

        # training data for supervised learning
        train_dial, val_dial, test_dial = self.corpus.get_corpus()
        self.train_data = DealDataLoaders('Train', train_dial, self.sv_config)
        self.val_data = DealDataLoaders('Val', val_dial, self.sv_config)
        self.test_data = DealDataLoaders('Test', test_dial, self.sv_config)

        # training func for supervised learning
        self.train_func = train_single_batch

        # recording func
        self.record_func = record
        if self.rl_config.record_freq > 0:
            self.ppl_exp_file = open(os.path.join(self.rl_config.record_path, 'ppl.tsv'), 'w')
            self.rl_exp_file = open(os.path.join(self.rl_config.record_path, 'rl.tsv'), 'w')
            self.learning_exp_file = open(os.path.join(self.rl_config.record_path, 'learning.tsv'), 'w')

        # evaluation
        self.validate_func = validate
        self.evaluator = evaluators.BleuEvaluator('Deal')
        self.generate_func = generate


    def run(self):
        n = 0
        best_valid_loss = np.inf
        best_rl_reward = 0

        # BEFORE RUN, RECORD INITIAL PERFORMANCE
        self.record_func(n, self.sys_model, self.test_data, self.sv_config, self.usr_model, self.ppl_exp_file,
                         self.dialog_eval, self.ctx_gen_eval, self.rl_exp_file)

        for ctxs in self.ctx_gen.iter(self.rl_config.nepoch):
            n += 1
            if n % 20 == 0:
                print('='*15, '{}/{}'.format(n, self.ctx_gen.total_size(self.rl_config.nepoch)))

            # supervised learning
            if self.rl_config.sv_train_freq > 0 and n % self.rl_config.sv_train_freq == 0:
                # print('-'*15, 'Supervised Learning', '-'*15)
                self.train_func(self.sys_model, self.train_data, self.sv_config)
                # print('-'*40)

            # roll out and learn
            _, agree, rl_reward, rl_stats = self.dialog.run(ctxs, verbose=n % self.rl_config.record_freq == 0)

            # record model performance in terms of several evaluation metrics
            if self.rl_config.record_freq > 0 and n % self.rl_config.record_freq == 0:
                # TEST ON TRAINING DATA
                rl_stats = validate_rl(self.dialog_eval, self.ctx_gen, num_episode=400)
                self.learning_exp_file.write('{}\t{}\t{}\t{}\n'.format(n, rl_stats['sys_rew'],
                                                                       rl_stats['avg_agree'],
                                                                       rl_stats['sys_unique']))
                self.learning_exp_file.flush()
                aver_reward = rl_stats['sys_rew']

                # TEST ON HELD-HOLD DATA
                print('-'*15, 'Recording start', '-'*15)
                self.record_func(n, self.sys_model, self.test_data, self.sv_config, self.usr_model, self.ppl_exp_file,
                                 self.dialog_eval, self.ctx_gen_eval, self.rl_exp_file)

                # SAVE MODEL BASED on REWARD
                if aver_reward > best_rl_reward:
                    print('[INFO] Update on reward in Epsd {} ({} > {})'.format(n, aver_reward, best_rl_reward))
                    th.save(self.sys_model.state_dict(), self.rl_config.reward_best_model_path)
                    best_rl_reward = aver_reward
                else:
                    print('[INFO] No update on reward in Epsd {} ({} < {})'.format(n, aver_reward, best_rl_reward))

                print('-'*15, 'Recording end', '-'*15)

            # print('='*15, 'Episode {} end'.format(n), '='*15)
            if self.rl_config.nepisode > 0 and n > self.rl_config.nepisode:
                print('-'*15, 'Stop from config', '-'*15)
                break

        print("$$$ Load {}-model".format(self.rl_config.reward_best_model_path))
        self.sv_config.batch_size = 32
        self.sys_model.load_state_dict(th.load(self.rl_config.reward_best_model_path))

        validate(self.sys_model, self.val_data, self.sv_config)
        validate(self.sys_model, self.test_data, self.sv_config)

        with open(os.path.join(self.rl_config.record_path, 'valid_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.val_data, self.sv_config, self.evaluator, num_batch=None,
                               dest_f=f)

        with open(os.path.join(self.rl_config.record_path, 'test_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.test_data, self.sv_config, self.evaluator, num_batch=None,
                               dest_f=f)


class OfflineTaskReinforce(object):
    def __init__(self, agent, corpus, sv_config, sys_model, rl_config, generate_func):
        self.agent = agent
        self.corpus = corpus
        self.sv_config = sv_config
        self.sys_model = sys_model
        self.rl_config = rl_config
        # training func for supervised learning
        self.train_func = task_train_single_batch
        self.record_func = record_task
        self.validate_func = validate

        # prepare data loader
        train_dial, val_dial, test_dial = self.corpus.get_corpus()
        self.train_data = BeliefDbDataLoaders('Train', train_dial, self.sv_config)
        self.sl_train_data = BeliefDbDataLoaders('Train', train_dial, self.sv_config)
        self.val_data = BeliefDbDataLoaders('Val', val_dial, self.sv_config)
        self.test_data = BeliefDbDataLoaders('Test', test_dial, self.sv_config)

        # create log files
        if self.rl_config.record_freq > 0:
            self.learning_exp_file = open(os.path.join(self.rl_config.record_path, 'offline-learning.tsv'), 'w')
            self.ppl_val_file = open(os.path.join(self.rl_config.record_path, 'val-ppl.tsv'), 'w')
            self.rl_val_file = open(os.path.join(self.rl_config.record_path, 'val-rl.tsv'), 'w')
            self.ppl_test_file = open(os.path.join(self.rl_config.record_path, 'test-ppl.tsv'), 'w')
            self.rl_test_file = open(os.path.join(self.rl_config.record_path, 'test-rl.tsv'), 'w')
        # evaluation
        self.evaluator = evaluators.MultiWozEvaluator('SYS_WOZ')
        self.generate_func = generate_func

    def run(self):
        n = 0
        best_valid_loss = np.inf
        best_rewards = -1 * np.inf

        # BEFORE RUN, RECORD INITIAL PERFORMANCE
        test_loss = self.validate_func(self.sys_model, self.test_data, self.sv_config, use_py=True)
        t_success, t_match, t_bleu, t_f1 = self.generate_func(self.sys_model, self.test_data, self.sv_config,
                                                              self.evaluator, None, verbose=False)

        self.ppl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(test_loss), t_bleu, t_f1))
        self.ppl_test_file.flush()

        self.rl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, (t_success + t_match), t_success, t_match))
        self.rl_test_file.flush()

        self.sys_model.train()
        try:
            for epoch_id in range(self.rl_config.nepoch):
                self.train_data.epoch_init(self.sv_config, shuffle=True, verbose=epoch_id == 0, fix_batch=True)
                while True:
                    if n % self.rl_config.episode_repeat == 0:
                        batch = self.train_data.next_batch()

                    if batch is None:
                        break

                    n += 1
                    if n % 50 == 0:
                        print("Reinforcement Learning {}/{} eposide".format(n, self.train_data.num_batch*self.rl_config.nepoch))
                        self.learning_exp_file.write(
                            '{}\t{}\n'.format(n, np.mean(self.agent.all_rewards[-50:])))
                        self.learning_exp_file.flush()

                    # reinforcement learning
                    # make sure it's the same dialo
                    assert len(set(batch['keys'])) == 1
                    task_report, success, match = self.agent.run(batch, self.evaluator, max_words=self.rl_config.max_words, temp=self.rl_config.temperature)
                    reward = float(success) # + float(match)
                    stats = {'Match': match, 'Success': success}
                    self.agent.update(reward, stats)

                    # supervised learning
                    if self.rl_config.sv_train_freq > 0 and n % self.rl_config.sv_train_freq == 0:
                        self.train_func(self.sys_model, self.sl_train_data, self.sv_config)

                    # record model performance in terms of several evaluation metrics
                    if self.rl_config.record_freq > 0 and n % self.rl_config.record_freq == 0:
                         self.agent.print_dialog(self.agent.dlg_history, reward, stats)
                         print('-'*15, 'Recording start', '-'*15)
                         # save train reward
                         self.learning_exp_file.write('{}\t{}\n'.format(n, np.mean(self.agent.all_rewards[-self.rl_config.record_freq:])))
                         self.learning_exp_file.flush()

                         # PPL & reward on validation
                         valid_loss = self.validate_func(self.sys_model, self.val_data, self.sv_config, use_py=True)
                         v_success, v_match, v_bleu, v_f1 = self.generate_func(self.sys_model, self.val_data, self.sv_config, self.evaluator, None, verbose=False)
                         self.ppl_val_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(valid_loss), v_bleu, v_f1))
                         self.ppl_val_file.flush()
                         self.rl_val_file.write('{}\t{}\t{}\t{}\n'.format(n, (v_success + v_match), v_success, v_match))
                         self.rl_val_file.flush()

                         test_loss = self.validate_func(self.sys_model, self.test_data, self.sv_config, use_py=True)
                         t_success, t_match, t_bleu, t_f1 = self.generate_func(self.sys_model, self.test_data, self.sv_config, self.evaluator, None, verbose=False)
                         self.ppl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, np.exp(test_loss), t_bleu, t_f1))
                         self.ppl_test_file.flush()
                         self.rl_test_file.write('{}\t{}\t{}\t{}\n'.format(n, (t_success + t_match), t_success, t_match))
                         self.rl_test_file.flush()

                         # save model is needed
                         if v_success+v_match > best_rewards:
                             print("Model saved with success {} match {}".format(v_success, v_match))
                             th.save(self.sys_model.state_dict(), self.rl_config.reward_best_model_path)
                             best_rewards = v_success+v_match


                         self.sys_model.train()
                         print('-'*15, 'Recording end', '-'*15)
        except KeyboardInterrupt:
            print("RL training stopped from keyboard")

        print("$$$ Load {}-model".format(self.rl_config.reward_best_model_path))
        self.sv_config.batch_size = 32
        self.sys_model.load_state_dict(th.load(self.rl_config.reward_best_model_path))

        validate(self.sys_model, self.val_data, self.sv_config, use_py=True)
        validate(self.sys_model, self.test_data, self.sv_config, use_py=True)

        with open(os.path.join(self.rl_config.record_path, 'valid_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.val_data, self.sv_config, self.evaluator, num_batch=None, dest_f=f)

        with open(os.path.join(self.rl_config.record_path, 'test_file.txt'), 'w') as f:
            self.generate_func(self.sys_model, self.test_data, self.sv_config, self.evaluator, num_batch=None, dest_f=f)


def validate_rl(dialog_eval, ctx_gen, num_episode=200):
    print("Validate on training goals for {} episode".format(num_episode))
    reward_list = []
    agree_list = []
    sent_metric = UniquenessSentMetric()
    word_metric = UniquenessWordMetric()
    for _ in range(num_episode):
        ctxs = ctx_gen.sample()
        conv, agree, rewards = dialog_eval.run(ctxs)
        true_reward = rewards[0] if agree else 0
        reward_list.append(true_reward)
        agree_list.append(float(agree if agree is not None else 0.0))
        for turn in conv:
            if turn[0] == 'System':
                sent_metric.record(turn[1])
                word_metric.record(turn[1])
    results = {'sys_rew': np.average(reward_list),
               'avg_agree': np.average(agree_list),
               'sys_sent_unique': sent_metric.value(),
               'sys_unique': word_metric.value()}
    return results


def train_single_batch(model, train_data, config):
    batch_cnt = 0
    optimizer = model.get_optimizer(config, verbose=False)
    model.train()
    
    # decoding CE
    train_data.epoch_init(config, shuffle=True, verbose=False)
    for i in range(16):
        batch = train_data.next_batch()
        if batch is None:
            train_data.epoch_init(config, shuffle=True, verbose=False)
            batch = train_data.next_batch()
        optimizer.zero_grad()
        loss = model(batch, mode=TEACH_FORCE)
        model.backward(loss, batch_cnt)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


def task_train_single_batch(model, train_data, config):
    batch_cnt = 0
    optimizer = model.get_optimizer(config, verbose=False)
    model.train()

    # decoding CE
    train_data.epoch_init(config, shuffle=True, verbose=False)
    for i in range(16):
        batch = train_data.next_batch()
        if batch is None:
            train_data.epoch_init(config, shuffle=True, verbose=False)
            batch = train_data.next_batch()
        optimizer.zero_grad()
        loss = model(batch, mode=TEACH_FORCE)
        model.backward(loss, batch_cnt)
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()


def train(model, train_data, val_data, test_data, config, evaluator, gen=None):
    patience = 10
    valid_loss_threshold = np.inf
    best_valid_loss = np.inf
    batch_cnt = 0
    optimizer = model.get_optimizer(config)
    done_epoch = 0
    best_epoch = 0
    train_loss = LossManager()
    model.train()
    logger.info(summary(model, show_weights=False))
    saved_models = []
    last_n_model = config.last_n_model if hasattr(config, 'last_n_model') else 5

    logger.info('***** Training Begins at {} *****'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info('***** Epoch 0/{} *****'.format(config.max_epoch))
    while True:
        train_data.epoch_init(config, shuffle=True, verbose=done_epoch==0, fix_batch=config.fix_train_batch)
        while True:
            batch = train_data.next_batch()
            if batch is None:
                break
    
            optimizer.zero_grad()
            loss = model(batch, mode=TEACH_FORCE)
            model.backward(loss, batch_cnt)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            batch_cnt += 1
            train_loss.add_loss(loss)
    
            if batch_cnt % config.print_step == 0:
                # print('Print step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
                logger.info(train_loss.pprint('Train',
                                        window=config.print_step, 
                                        prefix='{}/{}-({:.3f})'.format(batch_cnt%config.ckpt_step, config.ckpt_step, model.kl_w)))
                sys.stdout.flush()
    
            if batch_cnt % config.ckpt_step == 0:
                logger.info('Checkpoint step at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
                logger.info('==== Evaluating Model ====')
                logger.info(train_loss.pprint('Train'))
                done_epoch += 1
                logger.info('done epoch {} -> {}'.format(done_epoch-1, done_epoch))

                # generation
                if gen is not None:
                    gen(model, val_data, config, evaluator, num_batch=config.preview_batch_num)

                # validation
                valid_loss = validate(model, val_data, config, batch_cnt)
                _ = validate(model, test_data, config, batch_cnt)

                # update early stopping stats
                if valid_loss < best_valid_loss:
                    if valid_loss <= valid_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch*config.patient_increase)
                        valid_loss_threshold = valid_loss
                        logger.info('Update patience to {}'.format(patience))
    
                    if config.save_model:
                        cur_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                        logger.info('!!Model Saved with loss = {},at {}.'.format(valid_loss, cur_time))
                        th.save(model.state_dict(), os.path.join(config.saved_path, '{}-model'.format(done_epoch)))
                        best_epoch = done_epoch
                        saved_models.append(done_epoch)
                        if len(saved_models) > last_n_model:
                            remove_model = saved_models[0]
                            saved_models = saved_models[-last_n_model:]
                            os.remove(os.path.join(config.saved_path, "{}-model".format(remove_model)))
    
                    best_valid_loss = valid_loss
    
                if done_epoch >= config.max_epoch \
                        or config.early_stop and patience <= done_epoch:
                    if done_epoch < config.max_epoch:
                        logger.info('!!!!! Early stop due to run out of patience !!!!!')
                    print('Best validation loss = %f' % (best_valid_loss, ))
                    return best_epoch
    
                # exit eval model
                model.train()
                train_loss.clear()
                logger.info('\n***** Epoch {}/{} *****'.format(done_epoch, config.max_epoch))
                sys.stdout.flush()


def validate(model, val_data, config, batch_cnt=None, use_py=None):
    model.eval()
    val_data.epoch_init(config, shuffle=False, verbose=False)
    losses = LossManager()
    while True:
        batch = val_data.next_batch()
        if batch is None:
            break
        if use_py is not None:
            loss = model(batch, mode=TEACH_FORCE, use_py=use_py)
        else:
            loss = model(batch, mode=TEACH_FORCE)

        losses.add_loss(loss)
        losses.add_backward_loss(model.model_sel_loss(loss, batch_cnt))

    valid_loss = losses.avg_loss()
    # print('Validation finished at {}'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    logger.info(losses.pprint(val_data.name))
    logger.info('Total valid loss = {}'.format(valid_loss))
    sys.stdout.flush()
    return valid_loss


def generate(model, data, config, evaluator, num_batch, dest_f=None):
    
    def write(msg):
        if msg is None or msg == '':
            return
        if dest_f is None:
            print(msg)
        else:
            dest_f.write(msg + '\n')

    model.eval()
    de_tknize = get_detokenize()
    data.epoch_init(config, shuffle=num_batch is not None, verbose=False)
    evaluator.initialize()
    logger.info('Generation: {} batches'.format(data.num_batch
                                          if num_batch is None
                                          else num_batch))
    batch_cnt = 0
    print_cnt = 0
    while True:
        batch_cnt += 1
        batch = data.next_batch()
        if batch is None or (num_batch is not None and data.ptr > num_batch):
            break
        outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)

        # move from GPU to CPU
        labels = labels.cpu()
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1) # (batch_size, max_dec_len)
        true_labels = labels.data.numpy() # (batch_size, output_seq_len)

        # get attention if possible
        if config.dec_use_attn:
            pred_attns = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_ATTN_SCORE]]
            pred_attns = np.array(pred_attns, dtype=float).squeeze(2).swapaxes(0, 1) # (batch_size, max_dec_len, max_ctx_len)
        else:
            pred_attns = None
        # get context
        ctx = batch.get('contexts') # (batch_size, max_ctx_len, max_utt_len)
        ctx_len = batch.get('context_lens') # (batch_size, )

        for b_id in range(pred_labels.shape[0]):
            # TODO attn
            pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id) 
            true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
            prev_ctx = ''
            if ctx is not None:
                ctx_str = []
                for t_id in range(ctx_len[b_id]):
                    temp_str = get_sent(model.vocab, de_tknize, ctx[:, t_id, :], b_id, stop_eos=False)
                    # print('temp_str = %s' % (temp_str, ))
                    # print('ctx[:, t_id, :] = %s' % (ctx[:, t_id, :], ))
                    ctx_str.append(temp_str)
                ctx_str = '|'.join(ctx_str)[-200::]
                prev_ctx = 'Source context: {}'.format(ctx_str)

            evaluator.add_example(true_str, pred_str)

            if num_batch is None or batch_cnt < 2:
                print_cnt += 1
                write('prev_ctx = %s' % (prev_ctx, ))
                write('True: {}'.format(true_str, ))
                write('Pred: {}'.format(pred_str, ))
                write('='*30)
                if num_batch is not None and print_cnt > 10:
                    break

    write(evaluator.get_report())
    write('Generation Done')


def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        # TODO EOT
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)


def generate_with_name(model, data, config):
    model.eval()
    de_tknize = get_detokenize()
    data.epoch_init(config, shuffle=False, verbose=False)
    logger.info('Generation With Name: {} batches.'.format(data.num_batch))

    from collections import defaultdict
    res = defaultdict(dict)
    while True:
        batch = data.next_batch()
        if batch is None:
            break
        keys, outputs, labels = model(batch, mode=GEN, gen_type=config.gen_type)
        
        pred_labels = [t.cpu().data.numpy() for t in outputs[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1) # (batch_size, max_dec_len)
        true_labels = labels.cpu().data.numpy() # (batch_size, output_seq_len)

        for b_id in range(pred_labels.shape[0]):
            pred_str = get_sent(model.vocab, de_tknize, pred_labels, b_id) 
            true_str = get_sent(model.vocab, de_tknize, true_labels, b_id)
            dlg_name, dlg_turn = keys[b_id]
            res[dlg_name][dlg_turn] = {'pred': pred_str, 'true': true_str}

    return res
