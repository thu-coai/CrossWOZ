from analyzer import Analyzer
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



def build_user_agent_svmnlu(use_nlu=True):
    user_nlu = SVMNLU(mode='all')
    
    user_dst = None
    
    user_policy = RulePolicy(character='usr')
    
    user_nlg = TemplateNLG(is_user=True)
    
    if use_nlu:
        user_agent = PipelineAgent(user_nlu, None, user_policy, user_nlg, 'user')
    else:
        user_agent = PipelineAgent(None, None, user_policy, None, 'user')

    return user_agent

def build_sys_agent_svmnlu(use_nlu=True):
    sys_nlu = SVMNLU(mode='all')
    
    sys_dst = RuleDST()
    
    sys_policy = RulePolicy(character='sys')
    
    sys_nlg = TemplateNLG(is_user=False)
    
    if use_nlu:
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    else:
        sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, 'sys')
    return sys_agent


def build_user_agent_bertnlu(use_nlu=True):
    user_nlu = BERTNLU(mode='all', config_file='multiwoz_all.json',
                       model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip')

    user_dst = None

    user_policy = RulePolicy(character='usr')

    user_nlg = TemplateNLG(is_user=True)

    if use_nlu:
        user_agent = PipelineAgent(user_nlu, None, user_policy, user_nlg, 'user')
    else:
        user_agent = PipelineAgent(None, None, user_policy, None, 'user')

    return user_agent

def build_sys_agent_bertnlu(use_nlu=True):
    sys_nlu = BERTNLU(mode='all', config_file='multiwoz_all.json',
                      model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip')
    sys_dst = RuleDST()

    sys_policy = RulePolicy(character='sys')

    sys_nlg = TemplateNLG(is_user=False)

    if use_nlu:
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    else:
        sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, 'sys')
    return sys_agent

def build_sys_agent_bertnlu_context(use_nlu=True):
    sys_nlu = BERTNLU(mode='all', config_file='multiwoz_all_context.json',
                      model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all_context.zip')
    sys_dst = RuleDST()

    sys_policy = RulePolicy(character='sys')

    sys_nlg = TemplateNLG(is_user=False)

    if use_nlu:
        sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')
    else:
        sys_agent = PipelineAgent(None, sys_dst, sys_policy, None, 'sys')
    return sys_agent

if __name__ == "__main__":
    #user agent for simulator
    # user_agent = build_user_agent_bertnlu(True)
    user_agent = build_user_agent_svmnlu(True)

    #build your own sys agent, modify the func to change the settings
    sys_agent_svm = build_sys_agent_svmnlu(True)
    # sys_agent_bert = build_sys_agent_bertnlu(True)
    # sys_agent_bert_context = build_sys_agent_bertnlu_context(True)

    #build analyzer, temporarily only for multiwoz
    analyzer = Analyzer(user_agent=user_agent, use_nlu=True, dataset='multiwoz')

    #sample dialog
    # analyzer.sample_dialog(sys_agent)

    #analyze and generate test report
    analyzer.comprehensive_analyze(sys_agent=sys_agent_svm, total_dialog=10)
    # analyzer.comprehensive_analyze(sys_agent=sys_agent_bert, model_name='bertnlu', total_dialog=1000)
    # analyzer.comprehensive_analyze(sys_agent=sys_agent_bert_context, model_name='bertnlu', total_dialog=1000)

    #compare multiple model
    # analyzer.compare_model(agent_list = [sys_agent_svm, sys_agent_svm], total_dialog=10)
