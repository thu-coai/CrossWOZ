from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.nlu.milu.multiwoz import MILU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch

sys_nlu = BERTNLU(mode='all', config_file='multiwoz_all.json',
                  model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip')
# sys_nlu = SVMNLU(mode='sys')
# simple rule DST
sys_dst = RuleDST()
# rule policy
sys_policy = RulePolicy(character='sys')
# template NLG
sys_nlg = TemplateNLG(is_user=False)
# assemble
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')

# user_nlu = sys_nlu
# user_nlu = SVMNLU(mode='all')
user_nlu = MILU(model_file="https://convlab.blob.core.windows.net/models/milu.tar.gz")
# not use dst
user_dst = None
# rule policy
user_policy = RulePolicy(character='usr')
# template NLG
user_nlg = TemplateNLG(is_user=True)
# assemble
user_agent = PipelineAgent(user_nlu, None, user_policy, user_nlg, 'user')

evaluator = MultiWozEvaluator()
sess = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=evaluator)

random.seed(20200131)
np.random.seed(20190827)
torch.manual_seed(20200131)
sys_response = ''
sess.init_session()
print('init goal:')
pprint(sess.evaluator.goal)
print('-'*50)
for i in range(40):
    sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
    print('user:', user_response)
    print('sys:', sys_response)
    print()
    if session_over is True:
        print('task complete:', user_policy.policy.goal.task_complete())
        print('task success:', sess.evaluator.task_success())
        print('book rate:', sess.evaluator.book_rate())
        print('inform precision/recall/f1:', sess.evaluator.inform_F1())
        print('-'*50)
        print('final goal:')
        pprint(sess.evaluator.goal)
        print('='*100)
        break

total_dialog = 10
random.seed(20200131)
goal_seeds = [random.randint(1,100000) for _ in range(total_dialog)]
precision = 0
recall = 0
f1 = 0
suc_num = 0
complete_num = 0
for j in range(total_dialog):
    sys_response = ''
    random.seed(goal_seeds[0])
    np.random.seed(goal_seeds[0])
    torch.manual_seed(goal_seeds[0])
    goal_seeds.pop(0)
    sess.init_session()
    # print('init goal:')
    # pprint(sess.evaluator.goal)
    # print('-'*50)
    for i in range(40):
        sys_response, user_response, session_over, reward = sess.next_turn(
            sys_response)
        # print('user:', user_response)
        # print('sys:', sys_response)
        if session_over is True:
            if sess.evaluator.task_success() == 1:
                suc_num = suc_num+1
            if user_policy.policy.goal.task_complete():
                complete_num += 1
            print('task complete:', user_policy.policy.goal.task_complete())
            print('task success:', sess.evaluator.task_success())
            print('book rate:', sess.evaluator.book_rate())
            print('inform precision/recall/f1:', sess.evaluator.inform_F1())
            stats = sess.evaluator.inform_F1()
            if(stats[0] != None):
                precision = precision+stats[0]
            if(stats[1] != None):
                recall = recall+stats[1]
            if(stats[2] != None):
                f1 = f1+stats[2]
            else:
                suc_num = suc_num-1
            # print('-'*50)
            # print('final goal:')
            # pprint(sess.evaluator.goal)
            # print('='*100)
            break
print("complete number of dialogs/tot:", complete_num/total_dialog)
print("success number of dialogs/tot:", suc_num/total_dialog)
print("average precision:", precision/total_dialog)
print("average recall:", recall/total_dialog)
print("average f1:", f1/total_dialog)