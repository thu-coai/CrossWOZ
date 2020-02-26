import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from convlab2.nlu.svm.multiwoz import SVMNLU
from convlab2.nlu.jointBERT.multiwoz import BERTNLU
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.nlg.template.multiwoz import TemplateNLG
from convlab2.dialog_agent import PipelineAgent, BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from helper import Reporter

class Analyzer():
    def __init__(self, user_agent, use_nlu=True, dataset='multiwoz'):
        self.user_agent = user_agent
        self.use_nlu = use_nlu
        self.dataset = dataset

    def build_sess(self, sys_agent):
        if self.dataset == 'multiwoz':
            evaluator = MultiWozEvaluator()
        else:
            evaluator = None
        
        if evaluator is None:
            self.sess = None
        else:
            self.sess = BiSession(sys_agent=sys_agent, user_agent=self.user_agent, kb_query=None, evaluator=evaluator)
        return self.sess

    def sample_dialog(self, sys_agent):
        sess = self.build_sess(sys_agent)
        random.seed(20200131)
        np.random.seed(20190827)
        torch.manual_seed(20200131)
        sys_response = [] if not self.use_nlu else ''
        sess.init_session()
        print('init goal:')
        pprint(sess.evaluator.goal)
        print('-'*50)
        for i in range(40):
            sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
            print('user:', user_response)
            # print('user in da:', sess.user_agent.get_in_da())
            # print('user out da:', sess.user_agent.get_out_da())
            print('sys:', sys_response)
            # print('sys in da:', sess.sys_agent.get_in_da())
            # print('sys out da:', sess.sys_agent.get_out_da())
            print()
            if session_over is True:
                print('task complete:', sess.user_agent.policy.policy.goal.task_complete())
                print('task success:', sess.evaluator.task_success())
                print('book rate:', sess.evaluator.book_rate())
                print('inform precision/recall/f1:', sess.evaluator.inform_F1())
                print('-'*50)
                print('final goal:')
                pprint(sess.evaluator.goal)
                print('='*100)
                break

    def comprehensive_analyze(self, sys_agent, total_dialog=100):
        sess = self.build_sess(sys_agent)

        goal_seeds = [random.randint(1,100000) for _ in range(total_dialog)]
        precision = 0
        recall = 0
        f1 = 0
        suc_num = 0
        complete_num = 0
        turn_num = 0
        turn_suc_num = 0
        model_name = sys_agent.name

        reporter = Reporter(model_name)

        for j in range(total_dialog):
            sys_response = [] if not self.use_nlu else ''
            random.seed(goal_seeds[0])
            np.random.seed(goal_seeds[0])
            torch.manual_seed(goal_seeds[0])
            goal_seeds.pop(0)
            sess.init_session()

            usr_da_list = []
            failed_da_sys = []
            failed_da_usr = []
            last_sys_da = None
        
            step = 0


            for i in range(40):
                sys_response, user_response, session_over, reward = sess.next_turn(
                    sys_response)
                
                # print('user in', sess.user_agent.get_in_da())
                # print('user out', sess.user_agent.get_out_da())

                # print('sys in', sess.sys_agent.get_in_da())
                # print('sys out', sess.sys_agent.get_out_da())

                step += 2

                if sess.user_agent.get_out_da() != [] and sess.user_agent.get_out_da() != sess.sys_agent.get_in_da():
                    for da1 in sess.user_agent.get_out_da():
                        for da2 in sess.sys_agent.get_in_da():
                            if da1 != da2 and da1 != None and da2 != None and (da1, da2) not in failed_da_sys:
                                failed_da_sys.append((da1, da2))

                

                if last_sys_da != None and last_sys_da != [] and sess.user_agent.get_in_da() != last_sys_da: 
                    for da1 in last_sys_da:
                        for da2 in sess.user_agent.get_in_da():
                            if da1 != da2 and da1 != None and da2 != None and (da1, da2) not in failed_da_usr:
                                failed_da_usr.append((da1, da2))

                last_sys_da = sess.sys_agent.get_out_da()
                usr_da_list.append(sess.user_agent.get_out_da())


                if session_over is True:
                    if sess.evaluator.task_success() == 1:
                        suc_num = suc_num+1
                        turn_suc_num += step
                    if sess.user_agent.policy.policy.goal.task_complete():
                        complete_num += 1
                    print('task complete:', sess.user_agent.policy.policy.goal.task_complete())
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

                    domain_set = []
                    for da in sess.evaluator.usr_da_array:
                        if da.split('-')[0] != 'general' and da.split('-')[0] not in domain_set:
                            domain_set.append(da.split('-')[0])

                    break
            
            turn_num += step

            da_list = usr_da_list
            cycle_start = []
            for da in usr_da_list:
                if len(da) == 1 and da[0][2] == 'general':
                    continue

                if usr_da_list.count(da) > 1 and da not in cycle_start:
                    cycle_start.append(da)

            domain_turn = []
            for da in usr_da_list:
                if len(da) > 0 and da[0] is not None and len(da[0]) > 2:
                    domain_turn.append(da[0][1].lower())
            

            for domain in domain_set:
                reporter.record(domain, sess.evaluator.domain_success(domain), sess.evaluator.domain_reqt_inform_analyze(domain), failed_da_sys, failed_da_usr, cycle_start, domain_turn)

            

        tmp = 0 if suc_num == 0 else turn_suc_num / suc_num
        print('=' * 100)
        print("complete number of dialogs/tot:", complete_num/total_dialog)
        print("success number of dialogs/tot:", suc_num/total_dialog)
        print("average precision:", precision/total_dialog)
        print("average recall:", recall/total_dialog)
        print("average f1:", f1/total_dialog)
        print("average turn (succ):", tmp)
        print("average turn (all):", turn_num / total_dialog)
        print('=' * 100)

        
        reporter.report(complete_num/total_dialog, suc_num/total_dialog, precision/total_dialog, recall/total_dialog, f1/total_dialog, tmp, turn_num / total_dialog)

        return suc_num/total_dialog, precision/total_dialog, recall/total_dialog, f1/total_dialog,  turn_num / total_dialog


    def compare_model(self, agent_list, total_dialog=100):

        model_name = [agent.name for agent in agent_list]

        if len(agent_list) != len(model_name):
            return
        if len(agent_list) <= 0:
            return

        seed = random.randint(1, 100000)
        
        y1, y2, y3, y4, y5 = [], [], [], [], []
        for i in range(len(agent_list)):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            suc, pre, rec, f1, turn = self.comprehensive_analyze(agent_list[i], total_dialog)
            y1.append(suc)
            y2.append(pre)
            y3.append(rec)
            y4.append(f1)
            y5.append(turn)

        x1 = list(range(1, 1 + len(model_name)))
        x1 = np.array(x1)
        x2 = x1 + 0.1
        x3 = x2 + 0.1
        x4 = x3 + 0.1

        plt.figure(figsize=(12, 7), dpi=300)

        font1 = {'weight' : 'normal','size' : 20}

        font2 = {'weight' : 'bold','size' : 22}

        font3 = {'weight' : 'bold','size' : 35}
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=22)
        plt.ylabel('score', font2)   
        plt.ylim(0, 1)
        plt.xlabel('system', font2)
        plt.title('Comparison of different systems', font3, pad=16)
        
        
        plt.bar(x1, y1, width=0.1, align='center', label='Success rate')
        plt.bar(x2, y2, width=0.1, align='center', tick_label=model_name, label='Precision')
        plt.bar(x3, y3, width=0.1, align='center', label='Recall')
        plt.bar(x4, y4, width=0.1, align='center', label='Inform F1')
        plt.legend(loc=2,prop=font1)
        if not os.path.exists('results/'):
                os.mkdir('results')
        plt.savefig('results/compare_results.jpg')
        plt.close()
        

if __name__ == "__main__":

    sys_nlu = SVMNLU(mode='all')
    

    sys_dst = RuleDST()
    # rule policy
    sys_policy = RulePolicy(character='sys')
    # template NLG
    sys_nlg = TemplateNLG(is_user=False)
    # assemble
    if self.use_nlu:
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    else:
        sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, 'sys')
    

    user_nlu = SVMNLU(mode='all')
    
    # not use dst
    user_dst = None
    # rule policy
    user_policy = RulePolicy(character='usr')
    # template NLG
    user_nlg = TemplateNLG(is_user=True)
    # assemble
    if self.use_nlu:
        user_agent = PipelineAgent(user_nlu, None, user_policy, user_nlg, 'user')
    else:
        user_agent = PipelineAgent(None, None, user_policy, None, 'user')

    compare_model(user_agent, [sys_agent, sys_agent, sys_agent, sys_agent], ['m1', 'm2', 'm3', 'm4'],  2)



