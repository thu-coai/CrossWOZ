import re
import random
import json


def get_domain(name):
    if name == 'object_division':
        return ObjectDivisionDomain()
    raise()


class ObjectDivisionDomain(object):
    def __init__(self):
        self.item_pattern = re.compile('^item([0-9])=([0-9\-])+$')

    def input_length(self):
        return 3

    def selection_length(self):
        return 6

    def generate_choices(self, inpt):
        cnts, _ = self.parse_context(inpt)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                right_choice = ['item%d=%d' % (i, n - c) for i, (n, c) in enumerate(zip(cnts, choice))]
                return [left_choice + right_choice]
            choices = []
            for c in range(cnts[idx] + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()
            return choices
        choices = gen(cnts)
        choices.append(['<no_agreement>'] * self.selection_length())
        choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def parse_context(self, ctx):
        cnts = [int(n) for n in ctx[0::2]]
        vals = [int(v) for v in ctx[1::2]]
        return cnts, vals

    def _to_int(self, x):
        try:
            return int(x)
        except:
            return 0

    def score_choices(self, choices, ctxs):
        assert len(choices) == len(ctxs)
        # print('choices = {}'.format(choices))
        # print('ctxs = {}'.format(ctxs))
        cnts = [int(x) for x in ctxs[0][0::2]]
        agree, scores = True, [0 for _ in range(len(ctxs))]
        for i, n in enumerate(cnts):
            for agent_id, (choice, ctx) in enumerate(zip(choices, ctxs)):
                # taken = self._to_int(choice[i+3][-1])
                taken = self._to_int(choice[i][-1])
                n -= taken
                scores[agent_id] += int(ctx[2 * i + 1]) * taken
            agree = agree and (n == 0)
        return agree, scores


class ContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file."""
    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, 'r') as f:
            ctx_pair = []
            for line in f:
                ctx = line.strip().split()
                ctx_pair.append(ctx)
                if len(ctx_pair) == 2:
                    self.ctxs.append(ctx_pair)
                    ctx_pair = []

    def sample(self):
        return random.choice(self.ctxs)

    def iter(self, nepoch=1):
        for e in range(nepoch):
            random.shuffle(self.ctxs)
            for ctx in self.ctxs:
                yield ctx

    def total_size(self, nepoch):
        return nepoch*len(self.ctxs)


class ContextGeneratorEval(object):
    """Dialogue context generator. Generates contexes from the file."""
    def __init__(self, context_file):
        self.ctxs = []
        with open(context_file, 'r') as f:
            ctx_pair = []
            for line in f:
                ctx = line.strip().split()
                ctx_pair.append(ctx)
                if len(ctx_pair) == 2:
                    self.ctxs.append(ctx_pair)
                    ctx_pair = []


class TaskGoalGenerator(object):
    def __init__(self, goal_file):
        self.goals = []
        data = json.load(open(goal_file))
        for key, raw_dlg in data.items():
            self.goals.append((key, raw_dlg['goal']))

    def sample(self):
        return random.choice(self.goals)

    def iter(self, nepoch=1):
        for e in range(nepoch):
            random.shuffle(self.goals)
            for goal in self.goals:
                yield goal


