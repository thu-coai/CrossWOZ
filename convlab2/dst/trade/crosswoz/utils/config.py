import os
import logging 
import argparse
from tqdm import tqdm
import torch

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

MODE = 'cn'
data_version = 'init'  # processed

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='TRADE Multi-Domain DST')

# Training Setting
parser.add_argument('-ds','--dataset', help='dataset', required=False, default="multiwoz")
parser.add_argument('-t','--task', help='Task Number', required=False, default="dst")
parser.add_argument('-path','--path', help='path of the file to load', required=False)
parser.add_argument('-sample','--sample', help='Number of Samples', required=False,default=None)
parser.add_argument('-patience','--patience', help='', required=False, default=6, type=int)
parser.add_argument('-es','--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
parser.add_argument('-all_vocab','--all_vocab', help='', required=False, default=1, type=int)
parser.add_argument('-imbsamp','--imbalance_sampler', help='', required=False, default=0, type=int)
parser.add_argument('-data_ratio','--data_ratio', help='', required=False, default=100, type=int)
parser.add_argument('-um','--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-bsz','--batch', help='Batch_size', required=False, type=int)

# Testing Setting
parser.add_argument('-rundev','--run_dev_testing', help='', required=False, default=0, type=int)
parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)
## model predictions
parser.add_argument('-gs','--genSample', help='Generate Sample', type=int, required=False, default=0)  #### change this when testing
parser.add_argument('-evalp','--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an','--addName', help='An add name for the model folder', required=False, default='')
parser.add_argument('-eb','--eval_batch', help='Evaluation Batch_size', required=False, type=int, default=0)

# Model architecture
parser.add_argument('-gate','--use_gate', help='', required=False, default=1, type=int)
parser.add_argument('-le','--load_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-femb','--fix_embedding', help='', required=False, default=0, type=int)
parser.add_argument('-paral','--parallel_decode', help='', required=False, default=0, type=int)

# Model Hyper-Parameters
parser.add_argument('-dec','--decoder', help='decoder model', required=False)
parser.add_argument('-hdd','--hidden', help='Hidden size', required=False, type=int, default=100)
parser.add_argument('-lr','--learn', help='Learning Rate', required=False, type=float)
parser.add_argument('-dr','--drop', help='Drop Out', required=False, type=float)
parser.add_argument('-lm','--limit', help='Word Limit', required=False,default=-10000)
parser.add_argument('-clip','--clip', help='gradient clipping', required=False, default=10, type=int) 
parser.add_argument('-tfr','--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False, default=0.5)
# parser.add_argument('-l','--layer', help='Layer Number', required=False)

# Unseen Domain Setting
parser.add_argument('-l_ewc','--lambda_ewc', help='regularization term for EWC loss', type=float, required=False, default=0.01)
parser.add_argument('-fisher_sample','--fisher_sample', help='number of sample used to approximate fisher mat', type=int, required=False, default=0)
parser.add_argument("--all_model", action="store_true")
parser.add_argument("--domain_as_task", action="store_true")
parser.add_argument('--run_except_4d', help='', required=False, default=1, type=int)
parser.add_argument("--strict_domain", action="store_true")
parser.add_argument('-exceptd','--except_domain', help='', required=False, default="", type=str)
parser.add_argument('-onlyd','--only_domain', help='', required=False, default="", type=str)


args = vars(parser.parse_known_args(args=[])[0])
if args["load_embedding"]:
    args["hidden"] = 100
if args["fix_embedding"]:
    args["addName"] += "FixEmb"
if args["except_domain"] != "":
    args["addName"] += "Except"+args["except_domain"]
if args["only_domain"] != "":
    args["addName"] += "Only"+args["only_domain"]

