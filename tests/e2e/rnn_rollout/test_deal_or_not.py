import argparse

from convlab2.e2e.rnn_rollout.deal_or_not import DealornotAgent
from convlab2.e2e.rnn_rollout.deal_or_not.model import get_context_generator
from convlab2 import DealornotSession
import convlab2.e2e.rnn_rollout.utils as utils
import numpy as np

session_num = 20

def rnn_model_args():
    parser = argparse.ArgumentParser(description='selfplaying script')
    parser.add_argument('--nembed_word', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=64,
                        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
            help='size of the hidden state for the language module')
    parser.add_argument('--nhid_cluster', type=int, default=256,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_ctx', type=int, default=64,
        help='size of the hidden state for the context module')
    parser.add_argument('--nhid_strat', type=int, default=64,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=64,
        help='size of the hidden state for the attention module')
    parser.add_argument('--nhid_sel', type=int, default=128,
        help='size of the hidden state for the selection module')
    parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-07,
        help='min threshold for learning rate annealing')
    parser.add_argument('--decay_rate', type=float,  default=5.0,
        help='decrease learning rate by this factor')
    parser.add_argument('--decay_every', type=int,  default=1,
        help='decrease learning rate after decay_every epochs')
    parser.add_argument('--momentum', type=float, default=0.1,
        help='momentum for sgd')
    parser.add_argument('--clip', type=float, default=2.0,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.1,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.2,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=30,
        help='max number of epochs')
    parser.add_argument('--num_clusters', type=int, default=50,
        help='number of clusters')
    parser.add_argument('--partner_ctx_weight', type=float, default=0.0,
        help='selection weight')
    parser.add_argument('--sel_weight', type=float, default=0.6,
        help='selection weight')
    parser.add_argument('--prediction_model_file', type=str,  default='',
        help='path to save the prediction model')
    parser.add_argument('--cluster_model_file', type=str,  default='',
        help='path to save the cluster model')
    parser.add_argument('--lang_model_file', type=str,  default='',
        help='path to save the language model')
    parser.add_argument('--model_file', type=str,
                        help='model file (use algorithm/dataset/configs as root path)',
                        default="models/rnn_model_state_dict.th")
    parser.add_argument('--alice_forward_model_file', type=str,
                        help='Alice forward model file')
    parser.add_argument('--bob_model_file', type=str,
                        help='Bob model file')
    parser.add_argument('--context_file', type=str, default='data/deal_or_not/selfplay.txt',
                        help='context file')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
                        help='temperature')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='print out converations')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--score_threshold', type=int, default=6,
                        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--max_turns', type=int, default=20,
                        help='maximum number of turns in a dialog')
    parser.add_argument('--log_file', type=str, default='',
                        help='log successful dialogs to file for training')
    parser.add_argument('--smart_alice', action='store_true', default=False,
                        help='make Alice smart again')
    parser.add_argument('--diverse_alice', action='store_true', default=False,
                        help='make Alice smart again')
    parser.add_argument('--rollout_bsz', type=int, default=3,
                        help='rollout batch size')
    parser.add_argument('--rollout_count_threshold', type=int, default=3,
                        help='rollout count threshold')
    parser.add_argument('--smart_bob', action='store_true', default=False,
                        help='make Bob smart again')
    parser.add_argument('--selection_model_file', type=str, default='models/selection_model.th',
                        help='path to save the final model')
    parser.add_argument('--rollout_model_file', type=str, default='',
                        help='path to save the final model')
    parser.add_argument('--diverse_bob', action='store_true', default=False,
                        help='make Alice smart again')
    parser.add_argument('--ref_text', type=str,
                        help='file with the reference text')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use CUDA')
    parser.add_argument('--domain', type=str, default='object_division',
                        help='domain for the dialogue')
    parser.add_argument('--visual', action='store_true', default=False,
                        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='eps greedy')
    parser.add_argument('--data', type=str, default='data/deal_or_not',
                        help='location of the data corpus (use project path root path)')
    parser.add_argument('--unk_threshold', type=int, default=20,
                        help='minimum word frequency to be in dictionary')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch size')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='plot graphs')
    parser.add_argument('--sep_sel', action='store_true', default=True,
            help='use separate classifiers for selection')
    args = parser.parse_args()
    return args

def sel_model_args():
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--data', type=str, default='data/negotiate',
        help='location of the data corpus')
    parser.add_argument('--nembed_word', type=int, default=128,
        help='size of word embeddings')
    parser.add_argument('--nembed_ctx', type=int, default=128,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_cluster', type=int, default=256,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_ctx', type=int, default=64,
        help='size of the hidden state for the context module')
    parser.add_argument('--nhid_strat', type=int, default=256,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=128,
        help='size of the hidden state for the attention module')
    parser.add_argument('--nhid_sel', type=int, default=128,
        help='size of the hidden state for the selection module')
    parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-5,
        help='min threshold for learning rate annealing')
    parser.add_argument('--decay_rate', type=float,  default=5.0,
        help='decrease learning rate by this factor')
    parser.add_argument('--decay_every', type=int,  default=1,
        help='decrease learning rate after decay_every epochs')
    parser.add_argument('--momentum', type=float, default=0.1,
        help='momentum for sgd')
    parser.add_argument('--clip', type=float, default=0.2,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.1,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.2,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=7,
        help='max number of epochs')
    parser.add_argument('--num_clusters', type=int, default=50,
        help='number of clusters')
    parser.add_argument('--bsz', type=int, default=25,
        help='batch size')
    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--partner_ctx_weight', type=float, default=0.0,
        help='selection weight')
    parser.add_argument('--sel_weight', type=float, default=0.6,
        help='selection weight')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='',
        help='path to save the final model')
    parser.add_argument('--prediction_model_file', type=str,  default='',
        help='path to save the prediction model')
    parser.add_argument('--selection_model_file', type=str,  default='models/selection_model_state_dict.th',
        help='path to save the selection model')
    parser.add_argument('--cluster_model_file', type=str,  default='',
        help='path to save the cluster model')
    parser.add_argument('--lang_model_file', type=str,  default='',
        help='path to save the language model')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--skip_values', action='store_true', default=True,
        help='skip values in ctx encoder')
    parser.add_argument('--model_type', type=str, default='selection_model',
        help='model type')
    parser.add_argument('--domain', type=str, default='object_division',
        help='domain for the dialogue')
    parser.add_argument('--clustering', action='store_true', default=False,
        help='use clustering')
    parser.add_argument('--sep_sel', action='store_true', default=True,
        help='use separate classifiers for selection')
    args = parser.parse_args()
    return args

# agent
alice_agent = DealornotAgent('Alice', rnn_model_args(), sel_model_args())
bob_agent = DealornotAgent('Bob', rnn_model_args(), sel_model_args())
agents = [alice_agent, bob_agent]
context_generator = get_context_generator(rnn_model_args().context_file)

# session
session = DealornotSession(alice_agent, bob_agent)

session_idx = 0
rewards = [[], []]
for ctxs in context_generator.iter():
    print('session_idx', session_idx)
    for agent, ctx, partner_ctx in zip(agents, ctxs, reversed(ctxs)):
        agent.feed_context(ctx)
        agent.feed_partner_context(partner_ctx)
    last_observation = None
    while True:
        response = session.next_response(last_observation)
        print('\t', ' '.join(response))
        session_over = session.is_terminated()
        if session_over:
            break
        last_observation = response
    agree, [alice_r, bob_r] = session.get_rewards(ctxs)
    print('session [{}] alice vs bos: {:.1f}/{:.1f}'.format(session_idx, alice_r, bob_r))
    rewards[0].append(alice_r)
    rewards[1].append(bob_r)
    session.init_session()
    session_idx += 1
# print(np.mean(rewards, axis=1))
