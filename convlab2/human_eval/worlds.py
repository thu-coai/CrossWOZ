# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import random
# from goal_generator import GoalGenerator
import sys
import time
from copy import deepcopy
from pprint import pprint

import numpy as np
import spacy
from parlai.core.worlds import validate
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.mturk.core.worlds import MTurkTaskWorld

sys.path.append('../../')
from convlab2.task.multiwoz.goal_generator import GoalGenerator

nlp = spacy.load('en_core_web_sm')

# Instruction messages
ONBOARD_MSG = '\nWelcome! Below is your persona \
        (you can find it on the left side of the chat)\n \
        When you are ready to start your conversation, \
        click the "I am ready, continue" button below\n'
START_MSG = '\nNow speak to the clerk to plan your trip! \n\
        <b>You can track your goal description on the left.</b> \n\
        You need to cover all the items to end the chat.\n \
        Say one item or two each time.\n \
        <span style="color:blue"><b>Please try to speak naturally and do not trivially copy \
        the goal descriptions into the message.</b></span>\n \
        <span style="color:red"><b>Say "Success" when you have accomplished your goal or \
        "Fail" when it feels like impossible to accomplish your goal with the clerk.</b></span>'
CHAT_NOT_DONE_MSG = 'Sorry, we need at least <b>{} more turn(s)</b> to finish. \
       Please send a new message:'
TRY_MORE_MSG = 'Sorry, we need you to try <b>more turns</b> to finish. \
       Please send a new message:'
NO_TEXT_MSG = 'Sorry, we received an empty message. \
       Please send a new message:'
TIMEOUT_MSG = '<b> The other person has timed out. \
        Please click the "Done with this HIT" button below to finish this HIT.\
        </b>'
EXCEED_MIN_TURNS_MSG = '\n {} chat turns finished! \n \
        You can click the "Done" button to end the chat if it\'s your turn \
        or keep chatting.'
UNEXPECTED_DISCONNECTION_MSG = 'The other worker unexpectedly diconnected. \n \
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
CHAT_ENDED_MSG = 'Thanks for your time! \n\
        Please click <span style="color:blue"><b>Done with this HIT</b>\
        </span> button below to finish this HIT.'
WAITING_MSG = 'Please wait while we match you with another worker...'
NAN_MSG = 'The score you entered must be in [1, 2, 3, 4, 5]. Please \
        try again:'
NAN_PERSONA_MSG = 'The score you entered must be in [1, 2]. Remember to \
        click the <b>SEND</b> button and not the <b>DONE</b> button. Please \
        try again:'
TOO_SHORT_MSG = 'Your message is too short, please make it more than \
        <b><span style="color:red">{} words</span></b>.'
TOO_LONG_MSG = 'Your message is too long, please make it less than \
        <b><span style="color:red">{} words</span></b>.'
COPIED_CHARACTER_MSG = 'We found that you <b><span style="color:red">trivially \
        copied character descriptions</span></b>. Please rephrase your \
        message again.'

# Evaluation messages
FAIL_REASON_MSG = 'Please give a <b>reason for the <span style="color:red">failure</span></b> in detail.'
UNDERSTANDING_MSG = 'Now please evaluate the bot\'s \
        <span style="color:blue"><b>language understanding</b></span> \
        during this conversation by <b>entering a score \
        from [1, 2, 3, 4, 5]</b> below: (1 means "doesn\'t understand your utterance at all" \
        and 5 means "understand very well", e.g., You can enter 3 for an OK dialog) \
        <span style="color:red"><b>NOTE: following this you will \
        be asked to give a reason for the score you choose.</b></span>'
UNDERSTANDING_REASON_MSG = 'Please give a <b>reason for the language understanding \
        score</b> you gave above. Please try to give concrete examples.'
APPROPRIATENESS_MSG = 'Now please evaluate the \
        <span style="color:blue"><b>appropriateness of bot\'s response</b></span> \
        during this conversation by <b>entering a score \
        from [1, 2, 3, 4, 5]</b> below: (1 means "not appropriate at all" and 5 \
        means "extremely appropriate", e.g., You can enter 3 for an OK dialog) \
        <span style="color:red"><b>NOTE: following this you will \
        be asked to give a reason for the score you choose.</b></span>'
APPROPRIATENESS_REASON_MSG = 'Please give a <b>reason for the appropriateness \
        score</b> you gave above. Please try to give concrete examples.'

import requests


class CambridgeBot(object):
    def __init__(self):
        self.history = [['null']]
        self.prev_state = None
        self.prev_active_domain = None

    def observe(self, acts):
        self.history[-1].append(acts['text'])
        print(self.history)

    def act(self):
        print(self.history)
        resp = requests.post('http://localhost:10002', json={'history': self.history,
                                                             'prev_state': self.prev_state,
                                                             'prev_active_domain': self.prev_active_domain})
        if resp.status_code != 200:
            raise Exception('POST /tasks/ {}'.format(resp.status_code))
        else:
            response = resp.json()["response"]
            if response == 'What did you say?':
                self.history = [['null']]
                self.prev_state = None
                self.prev_active_domain = None
            else:
                self.history.append([response])
                self.prev_state = resp.json()['state']
                self.prev_active_domain = resp.json()['active_domain']
        print('Response: {}'.format(response))
        print(self.history)
        return {'text': response}


class SequicityBot(object):
    def __init__(self):
        self.state = {}
        self.input = ''

    def observe(self, acts):
        self.input = acts['text']
        print(self.input)

    def act(self):
        resp = requests.post('http://localhost:10001', json={'input': self.input,
                                                             'state': self.state})
        if resp.status_code != 200:
            raise Exception('POST /tasks/ {}'.format(resp.status_code))
        else:
            response = resp.json()["response"]
            if response == 'What did you say?':
                self.state = {}
            else:
                self.state = resp.json()['state']
        print('Response: {}'.format(response))
        return {'text': response}


class RuleBot(object):
    def __init__(self):
        self.state = {}
        self.input = ''
        self.rule_bot_recommend_flag = -1

    def observe(self, acts):
        self.input = acts['text']
        print(self.input)

    def act(self):
        resp = requests.post('http://localhost:10003', json={'input': self.input,
                                                             'state': self.state,
                                                             'recommend_flag':self.rule_bot_recommend_flag})
        if resp.status_code != 200:
            raise Exception('POST /tasks/ {}'.format(resp.status_code))
        else:
            response = resp.json()["response"]
            if response == 'What did you say?':
                self.state = {}
            else:
                self.state = resp.json()['state']
            self.rule_bot_recommend_flag = resp.json()["recommend_flag"]
        print('Response: {}'.format(response))
        return {'text': response}


class DQNBot(object):
    def __init__(self):
        self.state = {}
        self.input = ''

    def observe(self, acts):
        self.input = acts['text']
        print(self.input)

    def act(self):
        resp = requests.post('http://localhost:10004', json={'input': self.input,
                                                             'agent_state': self.state})
        if resp.status_code != 200:
            raise Exception('POST /tasks/ {}'.format(resp.status_code))
        else:
            response = resp.json()["response"]
            if response == 'What did you say?':
                self.state = {}
            else:
                self.state = resp.json()['agent_state']
        print('Response: {}'.format(response))
        return {'text': response}


class MultiWozEvalWorld(MTurkTaskWorld):
    def __init__(self, opt, agent,
                 num_extra_trial=2,
                 max_turn=50,
                 max_resp_time=120,
                 model_agent_opt=None,
                 world_tag='',
                 agent_timeout_shutdown=120):
        self.opt = opt
        self.agent = agent
        self.turn_idx = 1
        self.hit_id = None
        self.max_turn = max_turn
        self.num_extra_trial = num_extra_trial
        self.dialog = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.eval_done = False
        self.chat_done = False
        self.success = False
        self.success_attempts = []
        self.fail_attempts = []
        self.fail_reason = None
        self.understanding_score = -1
        self.understanding_reason = None
        self.appropriateness_score = -1
        self.appropriateness_reason = None
        self.world_tag = world_tag
        self.ratings = ['1', '2', '3', '4', '5']
        super().__init__(opt, agent)

        # set up model agent
        self.model_agents = {
            # "cambridge": CambridgeBot(),
            # "sequicity": SequicityBot(),
            # "RuleBot": RuleBot(),
            "DQNBot": DQNBot()
        }
        # self.model_agent = RuleBot()
        # self.model_agent = DQNBot()
        self.model_name = random.choice(list(self.model_agents.keys()))
        self.model_agent = self.model_agents[self.model_name]
        print("Bot is loaded")

        # below are timeout protocols
        self.max_resp_time = max_resp_time  # in secs
        self.agent_timeout_shutdown = agent_timeout_shutdown

        # set up personas
        self.goal = None
        goal_generator = GoalGenerator(boldify=True)
        num_goal_trials = 0
        while num_goal_trials < 100 and self.goal == None:
            try:
                self.goal = goal_generator.get_user_goal()
            except Exception as e:
                print(e)
                num_goal_trials += 1
        self.goal_message = goal_generator.build_message(self.goal)
        self.goal_text = '<ul>'
        for m in self.goal_message:
            self.goal_text += '<li>' + m + '</li>'
        self.goal_text += '</ul>'
        print(self.goal_text)

        self.state = deepcopy(self.goal)

    def _track_state(self, inp):
        def appear(words, sent):
            for word in words:
                if word in sent:
                    return True
            return False

        doc = nlp(inp)
        inp = ' '.join([token.text for token in doc])
        inp = inp.strip().lower()
        print('input: ', inp)
        for domain in self.goal:
            if domain == 'domain_ordering':
                continue
            if 'info' in self.goal[domain]:
                for slot in self.goal[domain]['info']:
                    if slot == 'parking' and slot in inp and slot in self.state[domain]['info']:
                        print("check out info ", slot)
                        del self.state[domain]['info'][slot]
                    elif slot == 'internet' and (slot in inp or 'wifi' in inp) and slot in self.state[domain]['info']:
                        print("check out info ", slot)
                        del self.state[domain]['info'][slot]
                    elif self.goal[domain]['info'][slot].lower() in inp and slot in self.state[domain]['info']:
                        print("check out info ", self.state[domain]['info'][slot])
                        del self.state[domain]['info'][slot]
            if 'reqt' in self.goal[domain]:
                for slot in self.goal[domain]['reqt']:
                    if slot.lower() in inp and slot in self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'internet' and 'wifi' in inp and slot in self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'trainid' and 'id' in inp and slot in self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'pricerange' and 'price' in inp and slot in self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'arriveBy' and appear(['arrive', 'get', 'when'], inp) and slot in \
                            self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'leaveAt' and appear(['leave', 'depart', 'when'], inp) and slot in \
                            self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
                    elif slot.lower() == 'duration' and appear(['long', 'time', 'duration'], inp) and slot in \
                            self.state[domain]['reqt']:
                        print("check out reqt ", slot)
                        self.state[domain]['reqt'].remove(slot)
            if 'book' in self.goal[domain]:
                for slot in self.goal[domain]['book']:
                    if self.goal[domain]['book'][slot].lower() in inp and slot in self.state[domain]['book']:
                        print("check out book ", self.state[domain]['book'][slot])
                        del self.state[domain]['book'][slot]

    def _get_num_items(self):
        num_items = 0
        goal = self.state
        for domain in goal:
            if domain == 'domain_ordering':
                continue
            if 'info' in goal[domain]:
                num_items += len(goal[domain]['info'])
            if 'reqt' in goal[domain]:
                num_items += len(goal[domain]['reqt'])
            if 'book' in goal[domain]:
                num_items += len(goal[domain]['book'])
        pprint(goal)
        print("Num of remaining items:", num_items)
        return num_items

    def parley(self):

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))

        """If at first turn, we need to give the turker a brief instruction"""
        if self.turn_idx == 1:
            control_msg['goal_text'] = self.goal_text
            control_msg['text'] = self.get_instruction(
                tag='start', agent_id=self.agent.id
            )
            self.agent.observe(validate(control_msg))
            time.sleep(3)

        """Otherwise, we proceed accordingly"""
        acts = []
        # MTurk evaluating agent turn
        idx = 0
        agent = self.agent

        acts.append(agent.act(timeout=self.max_resp_time))

        if acts[idx] is None:
            raise Exception("None from turker")

        # MTurk agent said something
        if acts[idx]['text'].strip() != '':
            self.check_disconnects(acts[idx])

            """If turker said success and covered all items, end the chat"""
            if acts[idx]['text'].strip().lower() == 'success':
                self.success_attempts.append(self.turn_idx)
                if self._get_num_items() < self.num_extra_trial:  # or self.turn_idx >= len(self.goal_message) - self.num_extra_trial:
                    self.success = True
                    self.chat_done = True
                else:
                    control_msg['text'] = TRY_MORE_MSG
                    agent.observe(validate(control_msg))
                    return

            """If turker said fail and tried hard, end the chat"""
            if acts[idx]['text'].strip().lower() == 'fail':
                self.fail_attempts.append(self.turn_idx)
                if self.turn_idx >= len(self.goal_message) + self.num_extra_trial:
                    self.success = False
                    self.chat_done = True
                else:
                    control_msg['text'] = TRY_MORE_MSG
                    agent.observe(validate(control_msg))
                    return

            if self.chat_done:
                """evaluation"""
                # Fail reason
                if acts[idx]['text'].strip().lower() == 'fail':
                    control_msg['text'] = FAIL_REASON_MSG
                    agent.observe(validate(control_msg))
                    acts[idx] = agent.act(timeout=self.max_resp_time)
                    while acts[idx]['text'] == '':
                        control_msg['text'] = 'Please try again.'
                        agent.observe(validate(control_msg))
                        acts[idx] = agent.act(timeout=self.max_resp_time)
                    if 'text' in acts[idx] and \
                            acts[idx]['text'] != '':
                        self.fail_reason = acts[idx]['text']

                # Language Understanding Check
                control_msg['text'] = UNDERSTANDING_MSG
                agent.observe(validate(control_msg))
                acts[idx] = agent.act(timeout=self.max_resp_time)
                while acts[idx]['text'] not in self.ratings:
                    control_msg['text'] = NAN_MSG
                    agent.observe(validate(control_msg))
                    acts[idx] = agent.act(timeout=self.max_resp_time)
                if 'text' in acts[idx] and \
                        acts[idx]['text'] in self.ratings:
                    self.understanding_score = int(acts[idx]['text'])

                # Language Understanding reason 
                control_msg['text'] = UNDERSTANDING_REASON_MSG
                agent.observe(validate(control_msg))
                acts[idx] = agent.act(timeout=self.max_resp_time)
                while acts[idx]['text'] == '':
                    control_msg['text'] = 'Please try again.'
                    agent.observe(validate(control_msg))
                    acts[idx] = agent.act(timeout=self.max_resp_time)
                if 'text' in acts[idx] and \
                        acts[idx]['text'] != '':
                    self.understanding_reason = acts[idx]['text']

                # Response Appropriateness Check
                control_msg['text'] = APPROPRIATENESS_MSG
                agent.observe(validate(control_msg))
                acts[idx] = agent.act(timeout=self.max_resp_time)
                while acts[idx]['text'] not in self.ratings:
                    control_msg['text'] = NAN_MSG
                    agent.observe(validate(control_msg))
                    acts[idx] = agent.act(timeout=self.max_resp_time)
                if 'text' in acts[idx] and \
                        acts[idx]['text'] in self.ratings:
                    self.appropriateness_score = int(acts[idx]['text'])

                # Response Appropriateness reason 
                control_msg['text'] = APPROPRIATENESS_REASON_MSG
                agent.observe(validate(control_msg))
                acts[idx] = agent.act(timeout=self.max_resp_time)
                while acts[idx]['text'] == '':
                    control_msg['text'] = 'Please try again.'
                    agent.observe(validate(control_msg))
                    acts[idx] = agent.act(timeout=self.max_resp_time)
                if 'text' in acts[idx] and \
                        acts[idx]['text'] != '':
                    self.appropriateness_reason = acts[idx]['text']

                self.eval_done = True

            if self.eval_done:
                control_msg['text'] = CHAT_ENDED_MSG
                agent.observe(validate(control_msg))
                return

            self._track_state(acts[idx]['text'])

            self.dialog.append((idx, acts[idx]['text']))
            self.turn_idx += 1

            self.model_agent.observe(acts[idx])

            # Model_agent turn
            idx = 1
            act = self.model_agent.act()
            # # echo
            # act = {'text': acts[0]['text']}

            # NOTE: model agent may or may not need to observe itself here,
            # depending on how your model handles this, uncomment for that
            # self.model_agent.observe(act)

            acts.append({'text': act['text']})

            # for (sb_0, sb_1) in [
            #     (' .', '.'),
            #     (' ,', ','),
            #     (' ?', '?'),
            #     (' !', '!'),
            #     ('i ', 'I ')
            # ]:
            #     acts[idx]['text'] = acts[idx]['text'].replace(sb_0, sb_1)
            # acts[idx]['text'].capitalize()
            acts[idx]['id'] = 'Clerk'
            acts[idx]['message_id'] = acts[0]['message_id'][:-1] + '0' if \
                acts[0]['message_id'][-1] != '0' else \
                acts[0]['message_id'][:-1] + '1'

            self.dialog.append((idx, acts[idx]['text']))
            # time.sleep(len(acts[idx]['text'].split(' ')) * 0.5)
            agent.observe(acts[idx])
        else:
            """If the message is None, ask turker to speak again"""
            control_msg['text'] = NO_TEXT_MSG
            agent.observe(validate(control_msg))
            return

    def episode_done(self):
        return self.eval_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return START_MSG

        if tag == 'try_more':
            return TRY_MORE_MSG

        if tag == 'timeout':
            return TIMEOUT_MESSAGE

        if tag == 'exceed_min_turns':
            return EXCEED_MIN_TURNS_MSG.format(self.n_turn)

    def save_data(self):
        convo_finished = True
        bad_workers = []
        if (self.agent.hit_is_abandoned or self.agent.hit_is_returned or
                self.agent.disconnected or self.agent.hit_is_expired):
            bad_workers.append(self.agent.worker_id)
            convo_finished = False
        if (not convo_finished or self.dialog == [] or
                self.understanding_score == -1 or
                self.appropriateness_score == -1):
            self.agent.not_approve = True
            convo_finished = False

        data_path = self.opt['datapath']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(
                data_path, '{}_{}_{}_{}_withreasons.json'.format(
                    self.model_name,
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        else:
            filename = os.path.join(
                data_path,
                '{}_{}_{}_{}_incomplete_withreasons.json'.format(
                    self.model_name,
                    time.strftime("%Y%m%d-%H%M%S"),
                    np.random.randint(0, 1000),
                    self.task_type
                )
            )
        result = {'goal': self.goal,
                  'goal_text': self.goal_text,
                  'dialog': self.dialog,
                  'workers': self.agent.worker_id,
                  'hit_id': self.agent.hit_id,
                  'assignment_id': self.agent.assignment_id,
                  'bad_workers': bad_workers,
                  'success': self.success,
                  'success_attempts': self.success_attempts,
                  'fail_attempts': self.fail_attempts,
                  'fail_reason': self.fail_reason,
                  'understanding_score': self.understanding_score,
                  'understanding_reason': self.understanding_reason,
                  'appropriateness_score': self.appropriateness_score,
                  'appropriateness_reason': self.appropriateness_reason,
                  'model': self.model_name}
        pprint(result)
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
            print(
                self.world_tag,
                ': Data successfully saved at {}.'.format(os.path.abspath(filename))
            )
            # check = json.load(open(filename, 'r'))
            # pprint(check)

    def is_exact_match(self, act, ag, tolerance=0):
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        text = act['text']
        if text not in ['', ' ', '  ', '   ']:
            n_word_match = 0
            for per in self.agentpersona_data:
                per_parse = per.split(' ')
                regular_words = ['', ' ', 'I', 'I\'m', 'My', 'i']
                for r_w in regular_words:
                    if r_w in per_parse:
                        per_parse.remove(r_w)
                per_subseq = [' '.join(per_parse[i:i + len(per_parse) -
                                                   tolerance]) for i in range(tolerance + 1)]
                for pp in per_subseq:
                    if pp in ['', ' ', '  ', '   ']:
                        per_subseq.remove(pp)
                n_word_match += sum([(paa in text) for paa in per_subseq])
            if n_word_match > 0:
                control_msg['text'] = COPIED_CHARACTER_MSG
                self.agentobserve(validate(control_msg))
                return True
            else:
                return False

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
        if act['episode_done']:
            return False

        control_msg = {'episode_done': False}
        control_msg['id'] = 'SYSTEM'

        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = TOO_SHORT_MSG.format(th_min)
            self.agentobserve(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = TOO_LONG_MSG.format(th_max)
            self.agentobserve(validate(control_msg))
            return True
        return False

    def reset_random(self):
        self.n_turn = np.random.randint(
            self.range_turn[0],
            self.range_turn[1]
        ) + 1

    def check_disconnects(self, act):
        if (
                act['text'] == '[TIMEOUT]' or
                act['text'] == '[RETURNED]' or
                act['text'] == '[DISCONNECT]'
        ):
            control_msg = {'episode_done': True}
            control_msg['id'] = 'SYSTEM'
            control_msg['text'] = self.get_instruction(
                agent_id=act['id'],
                tag='timeout'
            )
            self.chat_done = True
            return True
        else:
            return False

    def shutdown(self):
        self.agent.shutdown()
        # global shutdown_agent

        # def shutdown_agent(mturk_agent):
        #     mturk_agent.shutdown()

        # Parallel(
        #     n_jobs=len(self.agents),
        #     backend='threading'
        # )(delayed(shutdown_agent)(agent) for agent in self.agents)
