import torch.nn as nn
import torch.optim as optim
import numpy as np
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.utils import LONG, FLOAT, Pack


class OfflineRlAgent(object):
    def __init__(self, model, corpus, args, name, tune_pi_only):
        self.model = model
        self.corpus = corpus
        self.args = args
        self.name = name
        self.raw_goal = None
        self.vec_goals_list = None
        self.logprobs = None
        print("Do we only tune the policy: {}".format(tune_pi_only))
        self.opt = optim.SGD(
            [p for n, p in self.model.named_parameters() if 'c2z' in n or not tune_pi_only],
            lr=self.args.rl_lr,
            momentum=self.args.momentum,
            nesterov=(self.args.nesterov and self.args.momentum > 0))
        # self.opt = optim.Adam(self.model.parameters(), lr=0.01)
        # self.opt = optim.RMSprop(self.model.parameters(), lr=0.0005)
        self.all_rewards = []
        self.all_grads = []
        self.model.train()

    def print_dialog(self, dialog, reward, stats):
        for t_id, turn in enumerate(dialog):
            if t_id % 2 == 0:
                print("Usr: {}".format(' '.join([t for t in turn if t != '<pad>'])))
            else:
                print("Sys: {}".format(' '.join(turn)))
        report = ['{}: {}'.format(k, v) for k, v in stats.items()]
        print("Reward {}. {}".format(reward, report))

    def run(self, batch, evaluator, max_words=None, temp=0.1):
        self.logprobs = []
        self.dlg_history =[]
        batch_size = len(batch['keys'])
        logprobs, outs = self.model.forward_rl(batch, max_words, temp)
        if batch_size == 1:
            logprobs = [logprobs]
            outs = [outs]

        key = batch['keys'][0]
        sys_turns = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))

        for log_prob in logprobs:
            self.logprobs.extend(log_prob)
        # compute reward here
        generated_dialog = {key: sys_turns}
        return evaluator.evaluateModel(generated_dialog, mode="offline_rl")

    def update(self, reward, stats):
        self.all_rewards.append(reward)
        # standardize the reward
        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        # compute accumulated discounted reward
        g = self.model.np2var(np.array([r]), FLOAT).view(1, 1)
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        # estimate the loss using one MonteCarlo rollout
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        # for name, p in self.model.named_parameters():
        #    print(name)
        #    print(p.grad)
        self.opt.step()


class OfflineLatentRlAgent(OfflineRlAgent):
    def run(self, batch, evaluator, max_words=None, temp=0.1):
        self.logprobs = []
        self.dlg_history =[]
        batch_size = len(batch['keys'])
        logprobs, outs, logprob_z, sample_z = self.model.forward_rl(batch, max_words, temp)
        if batch_size == 1:
            outs = [outs]
        key = batch['keys'][0]
        sys_turns = []
        # construct the dialog history for printing
        for turn_id, turn in enumerate(batch['contexts']):
            user_input = self.corpus.id2sent(turn[-1])
            self.dlg_history.append(user_input)
            sys_output = self.corpus.id2sent(outs[turn_id])
            self.dlg_history.append(sys_output)
            sys_turns.append(' '.join(sys_output))

        for b_id in range(batch_size):
            self.logprobs.append(logprob_z[b_id])
        # compute reward here
        generated_dialog = {key: sys_turns}
        return evaluator.evaluateModel(generated_dialog, mode="offline_rl")