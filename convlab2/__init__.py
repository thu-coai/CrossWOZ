from convlab2.nlu import NLU
from convlab2.dst import DST
from convlab2.policy import Policy
from convlab2.nlg import NLG
from convlab2.dialog_agent import Agent, PipelineAgent
from convlab2.dialog_agent import Session, BiSession, DealornotSession

from os.path import abspath, dirname


def get_root_path():
    return dirname(dirname(abspath(__file__)))
