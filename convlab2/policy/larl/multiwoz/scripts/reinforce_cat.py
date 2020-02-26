import time
import os
import sys
sys.path.append('../')
import json
import torch as th
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.utils import Pack, set_seed
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import NormMultiWozCorpus
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.models_task import SysPerfectBD2Cat
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.agent_task import OfflineLatentRlAgent
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.main import OfflineTaskReinforce
from experiments_woz.dialog_utils import task_generate


def main():
    start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[START]', start_time, '='*30)
    # RL configuration
    env = 'gpu'
    pretrained_folder = '2019-09-07-01-03-54-sl_cat'
    # pretrained_folder = '2019-06-20-22-49-55-sl_cat'
    # pretrained_model_id = 41
    pretrained_model_id = 35

    exp_dir = os.path.join('sys_config_log_model', pretrained_folder, 'rl-'+start_time)
    # create exp folder
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    rl_config = Pack(
        train_path='../data/norm-multi-woz/train_dials.json',
        valid_path='../data/norm-multi-woz/val_dials.json',
        test_path='../data/norm-multi-woz/test_dials.json',

        sv_config_path = os.path.join('sys_config_log_model', pretrained_folder, 'config.json'),
        sv_model_path = os.path.join('sys_config_log_model', pretrained_folder, '{}-model'.format(pretrained_model_id)),

        rl_config_path = os.path.join(exp_dir, 'rl_config.json'),
        rl_model_path = os.path.join(exp_dir, 'rl_model'),

        ppl_best_model_path = os.path.join(exp_dir, 'ppl_best.model'),
        reward_best_model_path = os.path.join(exp_dir, 'reward_best.model'),
        record_path = exp_dir,
        record_freq = 200,
        sv_train_freq= 0,  # TODO pay attention to main.py, cuz it is also controlled there
        use_gpu = env == 'gpu',
        nepoch = 10,
        nepisode = 0,
        tune_pi_only=False,
        max_words = 100,
        temperature = 1.0,
        episode_repeat = 1.0,
        rl_lr = 0.01,
        momentum = 0.0,
        nesterov = False,
        gamma = 0.99,
        rl_clip = 5.0,
        random_seed = 100,
    )

    # save configuration
    with open(rl_config.rl_config_path, 'w') as f:
        json.dump(rl_config, f, indent=4)

    # set random seed
    set_seed(rl_config.random_seed)

    # load previous supervised learning configuration and corpus
    sv_config = Pack(json.load(open(rl_config.sv_config_path)))
    sv_config['dropout'] = 0.0
    sv_config['use_gpu'] = rl_config.use_gpu
    corpus = NormMultiWozCorpus(sv_config)

    # TARGET AGENT
    sys_model = SysPerfectBD2Cat(corpus, sv_config)
    if sv_config.use_gpu:
        sys_model.cuda()
    sys_model.load_state_dict(th.load(rl_config.sv_model_path, map_location=lambda storage, location: storage))
    sys_model.eval()
    sys = OfflineLatentRlAgent(sys_model, corpus, rl_config, name='System', tune_pi_only=rl_config.tune_pi_only)

    # start RL
    reinforce = OfflineTaskReinforce(sys, corpus, sv_config, sys_model, rl_config, task_generate)
    reinforce.run()

    end_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    print('[END]', end_time, '='*30)


if __name__ == '__main__':
    main()
