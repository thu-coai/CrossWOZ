import sys
import os

import numpy as np
import copy
from flask import Flask, request, jsonify
from queue import PriorityQueue
from threading import Thread


# Agent
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.nlu.milu.multiwoz import MILU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
import random
import numpy as np
from pprint import pprint

rgi_queue = PriorityQueue(maxsize=0)
rgo_queue = PriorityQueue(maxsize=0)

app = Flask(__name__)

sys_nlu = MILU()
sys_dst = RuleDST()
sys_policy = RulePolicy(character='sys')
sys_nlg = TemplateNLG(is_user=False)

agent = PipelineAgent(sys_nlu,sys_dst,sys_policy, sys_nlg,'amt')

print(agent.response('I am looking for a hotel'))


@app.route('/', methods=['GET', 'POST'])
def process():
    try:
        in_request = request.json
        print(in_request)
    except:
        return "invalid input: {}".format(in_request)
    rgi_queue.put(in_request)
    rgi_queue.join()
    output = rgo_queue.get()
    print(output['response'])
    rgo_queue.task_done()
    # return jsonify({'response': response})
    return jsonify(output)


def generate_response(in_queue, out_queue):
    while True:
        # pop input
        last_action = 'null'
        in_request = in_queue.get()
        obs = in_request['input']
        if in_request['agent_state'] == {}:
            agent.init_session()
        else:
            encoded_state, dst_state, last_action = in_request['agent_state']
            agent.dst.state = copy.deepcopy(dst_state)
        try:
            action = agent.response(obs)
            print(f'obs:{obs}; action:{action}')
            dst_state = copy.deepcopy(agent.dst.state)
            encoded_state = None
        except Exception as e:
            print('agent error', e)

        try:
            if action == '':
                response = 'Sorry I do not understand, can you paraphrase?'
            else:
                response = action
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'

        last_action = action
        out_queue.put({'response': response, 'agent_state': (encoded_state, dst_state, last_action)})
        in_queue.task_done()
        out_queue.join()


if __name__ == '__main__':
    worker = Thread(target=generate_response, args=(rgi_queue, rgo_queue,))
    worker.setDaemon(True)
    worker.start()

    app.run(host='0.0.0.0', port=10004)