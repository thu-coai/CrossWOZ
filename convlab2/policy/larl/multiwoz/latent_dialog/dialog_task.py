from convlab.modules.word_policy.multiwoz.larl.latent_dialog.metric import MetricsContainer
from convlab.modules.word_policy.multiwoz.larl.latent_dialog.corpora import EOD, EOS
from convlab.modules.word_policy.multiwoz.larl.latent_dialog import evaluators


class Dialog(object):
    """Dialogue runner."""
    def __init__(self, agents, args):
        assert len(agents) == 2
        self.agents = agents
        self.system, self.user = agents
        self.args = args
        self.metrics = MetricsContainer()
        self.dlg_evaluator = evaluators.MultiWozEvaluator('SYS_WOZ')
        self._register_metrics()

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_average('reward')
        self.metrics.register_time('time')

    def _is_eod(self, out):
        return len(out) == 2 and out[0] == EOD and out[1] == EOS

    def _eval_dialog(self, conv, g_key, goal):
        generated_dialog = dict()
        generated_dialog[g_key] = {'goal': goal, 'log': list()}
        for t_id, (name, utt) in enumerate(conv):
            # assert utt[-1] == EOS, utt
            if t_id % 2 == 0:
                assert name == 'User'
            utt = ' '.join(utt[:-1])
            if utt == EOD:
                continue
            generated_dialog[g_key]['log'].append({'text': utt})
        report, success_r, match_r = self.dlg_evaluator.evaluateModel(generated_dialog, mode='rollout')
        return success_r + match_r

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, g_key, goal):
        """Runs one instance of the dialogue."""
        # initialize agents by feeding in the goal
        # initialize BOD utterance for each agent
        for agent in self.agents:
            agent.feed_goal(goal)
            agent.bod_init()

        # role assignment
        reader, writer = self.system, self.user
        begin_name = writer.name
        print('begin_name = {}'.format(begin_name))

        conv = []
        # reset metrics
        self.metrics.reset()
        nturn = 0
        while True:
            nturn += 1
            # produce an utterance
            out_words = writer.write() # out: list of word, str, len = max_words
            print('\t{} out_words = {}'.format(writer.name, ' '.join(out_words)))

            self.metrics.record('sent_len', len(out_words))
            # self.metrics.record('%s_unique' % writer.name, out_words)

            # append the utterance to the conversation
            conv.append((writer.name, out_words))
            # make the other agent to read it
            reader.read(out_words)
            # check if the end of the conversation was generated
            if self._is_eod(out_words):
                break

            if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
                # return conv, 0
                break

            writer, reader = reader, writer

        # evaluate dialog and produce success
        reward = self._eval_dialog(conv, g_key, goal)
        print('Reward = {}'.format(reward))
        # perform update
        self.system.update(reward)
        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('reward', int(reward))

        print('='*50)
        print(self.show_metrics())
        print('='*50)
        return conv, reward


class DialogEval(Dialog):
    def run(self, g_key, goal):
        """Runs one instance of the dialogue."""
        # initialize agents by feeding in the goal
        # initialize BOD utterance for each agent
        for agent in self.agents:
            agent.feed_goal(goal)
            agent.bod_init()

        # role assignment
        reader, writer = self.system, self.user
        conv = []
        nturn = 0
        while True:
            nturn += 1
            # produce an utterance
            out_words = writer.write()  # out: list of word, str, len = max_words
            conv.append((writer.name, out_words))
            # make the other agent to read it
            reader.read(out_words)
            # check if the end of the conversation was generated
            if self._is_eod(out_words):
                break

            writer, reader = reader, writer
            if self.args.max_nego_turn > 0 and nturn >= self.args.max_nego_turn:
                return conv, 0

        # evaluate dialog and produce success
        reward = self._eval_dialog(conv, g_key, goal)
        return conv, reward
